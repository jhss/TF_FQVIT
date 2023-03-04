import tfimm
import tensorflow as tf
import argparse
from datasets import load_dataset

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from functools import partial

def generator(dataset):
    for idx, data in enumerate(dataset):

        img, label = data['image'], data['label']
        img = (lambda x: x.convert('RGB') if x.mode != 'RGB' else x)(img)
        img = img.resize((224, 224))

        yield img, label

def calibration_fn(train_loader, args):
    preprocess = tfimm.create_preprocessing(args.model_name, dtype="float32")

    for idx, (data, label) in enumerate(train_loader.batch(100).prefetch(1)):
        if idx == 10: break
        image = preprocess(data)
        yield image

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'vit_base_patch16_224')

args = parser.parse_args()

model = tfimm.create_model(args.model_name, pretrained="timm")
model.save("tf_model")

conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.INT8)
converter = trt.TrtGraphConverterV2(input_saved_model_dir="./tf_model", conversion_params=conversion_params)

imagenet = load_dataset('imagenet-1k', streaming = True)

train_dataset, valid_dataset = imagenet['train'], imagenet['validation']
train_generator = partial(generator, train_dataset)
valid_generator = partial(generator, valid_dataset)

train_loader    = tf.data.Dataset.from_generator(train_generator,
                                                 output_shapes = ((224, 224, 3), ()),
                                                 output_types  = (tf.float32, tf.int32)
                                                 )

valid_loader   = tf.data.Dataset.from_generator(valid_generator,
                                                output_shapes = ((224, 224, 3), ()),
                                                output_types = (tf.float32, tf.int32)
                                               )

preprocess = tfimm.create_preprocessing(args.model_name, dtype="float32")
trt_model = converter.convert(calibration_input_fn=partial(calibration_fn, train_loader, args))
top1_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
cur_top1    = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
cur_top5    = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

for iter_idx, data in enumerate(valid_loader.batch(100).prefetch(1)):
    images, labels = data
    images = preprocess(images)

    output = trt_model(images)
    pred = output['output_1']

    cur_top1.reset_state()
    cur_top5.reset_state()
    cur_top1.update_state(labels.numpy(), pred.numpy())
    cur_top5.update_state(labels.numpy(), pred.numpy())
    top1_metric.update_state(labels.numpy(), pred.numpy())
    top5_metric.update_state(labels.numpy(), pred.numpy())

    print(f"[{iter_idx}/10]: Currrent Top1 {cur_top1.result().numpy():.5f} / Top5 {cur_top5.result().numpy():.5f}")

    if iter_idx == 10: break

print(f"Average Top1 {top1_metric.result().numpy():.5f} / Top5 {top5_metric.result().numpy():.5f}")
print("Finish")
