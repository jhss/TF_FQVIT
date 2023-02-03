import argparse, time
from functools import partial

import numpy as np
import tensorflow as tf
import tfimm

from model.factory import create_model
from datasets import load_dataset
import datasets

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type = str, default = 'qdeit_small_patch16_224')
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--buffer_size', type = int, default = 1000, 
                        help = 'shuffle buffer size')
    parser.add_argument('--ptf', default = False, action = 'store_true')
    parser.add_argument('--lis', default = False, action = 'store_true')
    args = parser.parse_args()
    
    return args

def generator(dataset):
    for data in dataset:
        img, label = data['image'], data['label']
        img = (lambda x: x.convert('RGB') if x.mode != 'RGB' else x)(img)
        img = img.resize((224, 224))

        yield img, label
        
def calibration(model, train_loader, args):
    devices = tf.config.list_physical_devices('GPU')
    print("GPU: ", devices)
    #sys.exit()
    print("[DEBUG] calibration start")
    preprocess = tfimm.create_preprocessing(args.model_name, dtype="float32")
    model.open_calibrate()
    image_list = []
    
    for iter, data in enumerate(train_loader.batch(args.batch_size).prefetch(1)):
        #print(data[1])
        image_list.append(preprocess(data[0]))
        if iter == 9:
            break
    
    start = time.time()
    
    for iter, images in enumerate(image_list):
        if iter == 9:
            model.open_last_calibrate()
        #images, labels = data
        #images = preprocess(images)
        preds = model(images)
            
    end = time.time() - start
    print(f"[DEBUG] calibration time: {end} sec")
    model.close_calibrate()
    print("[DEBUG] calibration finish")

def validation(model, valid_loader, args):
    
    preprocess = tfimm.create_preprocessing(args.model_name, dtype="float32")
    top1_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    cur_top1    = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    cur_top5    = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    
    for iter, data in enumerate(valid_loader.batch(args.batch_size).prefetch(1)):
        
        images, labels = data
        images = preprocess(images)
        
        preds = model(images)
        
        #top1_metric.reset_state()
        #top5_metric.reset_state()
        cur_top1.reset_state()
        cur_top5.reset_state()
        cur_top1.update_state(labels.numpy(), preds.numpy())
        cur_top5.update_state(labels.numpy(), preds.numpy())
        top1_metric.update_state(labels.numpy(), preds.numpy())
        top5_metric.update_state(labels.numpy(), preds.numpy())
        
        print(f"[{iter}]: Currrent Top1 {cur_top1.result().numpy():.5f} / Top5 {cur_top5.result().numpy():.5f}")
        print(f"[{iter}]: Top1 {top1_metric.result().numpy():.5f} / Top5 {top5_metric.result().numpy():.5f}")

        if iter == 10:
            break

if __name__ == '__main__':
    
    # Set config
    args = parse_args()
    
    # load ImageNet
      
    #imagenet = load_dataset('imagenet-1k', streaming = True)
    imagenet = load_dataset('imagenet-1k', streaming = True)
    #imagenet = load_dataset('imagenet-1k', num_proc = 20, split = datasets.Split.TEST)
    train_dataset, valid_dataset = imagenet['train'], imagenet['validation']
    #valid_dataset = imagenet['test']
    
    train_generator = partial(generator, train_dataset)
    train_loader    = tf.data.Dataset.from_generator(train_generator,
                                                     output_shapes = ((224, 224, 3), ()),
                                                     output_types  = (tf.float32, tf.int32)
                                                    )
    
    valid_generator = partial(generator, valid_dataset)
    valid_loader    = tf.data.Dataset.from_generator(valid_generator,
                                                     output_shapes = ((224, 224, 3), ()),
                                                     output_types  = (tf.float32, tf.int32)
                                                    )
       
    # Preprocess
    print("[DEBUG] create model start")
    model = create_model(args, pretrained="timm")
    print("[DEBUG] create model finish")
    #for layer in model.layers:
    #    print(layer.name)
    print("[DEBUG] calibration start")
    calibration(model, train_loader, args)
    #sys.exit()
    model.quant()
    
    valid_start_time = time.time()
    validation(model, valid_loader, args)
    valid_end_time = time.time() - valid_start_time
    print(f"[DEBUG] validation time: {valid_end_time}")
    #model.summary()
                               
                               