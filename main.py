import argparse
from functools import partial

import tensorflow as tf
import tfimm

from model.factory import create_model
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type = str, default = 'qdeit_small_patch16_224')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--buffer_size', type = int, default = 1000, 
                        help = 'shuffle buffer size')
    args = parser.parse_args()
    
    return args

def generator(dataset):
    for data in valid_dataset:
        img, label = data['image'], data['label']
        img = (lambda x: x.convert('RGB') if x.mode != 'RGB' else x)(img)
        img = img.resize((224, 224))

        yield img, label
        
def calibration(model, train_loader, args):
    
    preprocess = tfimm.create_preprocessing(args.model, dtype="float32")
    model.open_calibrate()
    for iter, data in enumerate(train_loader.shuffle(args.buffer_size).batch(args.batch_size).prefetch(1)):
        #print(data)
        images, labels = data
        images = preprocess(images)
        
        preds = model(images)
        break
    model.close_calibrate()
    print("[DEBUG] calibration finish")

def validation(model, valid_loader, args):
    
    preprocess = tfimm.create_preprocessing(args.model, dtype="float32")
    top1_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    
    for iter, data in enumerate(valid_loader.batch(args.batch_size).prefetch(1)):
        
        images, labels = data
        images = preprocess(images)
        
        preds = model(images)
        
        top1_metric.reset_state()
        top5_metric.reset_state()
        top1_metric.update_state(labels.numpy(), preds.numpy())
        top5_metric.update_state(labels.numpy(), preds.numpy())

        print(f"[{iter}]: Top1 {top1_metric.result().numpy():.5f} / Top5 {top5_metric.result().numpy():.5f}")

        if iter == 0:
            break

if __name__ == '__main__':
    
    # Set config
    args = parse_args()
    
    # load ImageNet
      
    imagenet = load_dataset('imagenet-1k', num_proc = 8)
    train_dataset, valid_dataset = imagenet['train'], imagenet['validation']
    
    valid_generator = partial(generator, valid_dataset)
    valid_loader    = tf.data.Dataset.from_generator(valid_generator,
                                                     output_shapes = ((224, 224, 3), ()),
                                                     output_types  = (tf.float32, tf.int32)
                                                    )
       
    # Preprocess
    model = create_model(args.model, pretrained="timm")
    
    #for layer in model.layers:
    #    print(layer.name)
        
    calibration(model, valid_loader, args)
    #validation(model, valid_loader, args)
    
    #model.summary()
                               
                               