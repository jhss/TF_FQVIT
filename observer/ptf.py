# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf

from .base import BaseObserver

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(pred-tgt), p), axis = 1))
    else:
        return tf.reduce_mean(tf.pow(tf.abs(pred-tgt), p))

class PtfObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver, self).__init__(module_type, bit_type,
                                          calibration_mode)

    def update(self, v, channel_first):
        v = self.reshape_tensor(v, channel_first)
        cur_max = tf.reduce_max(v, axis = 1)
        
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = tf.math.maximum(cur_max, self.max_val)
        cur_min = tf.reduce_min(v, axis = 1)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = tf.math.minimum(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = tf.reduce_max(self.max_val)
            self.min_val = tf.reduce_min(self.min_val)

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = tf.reduce_max(max_val)
        min_val_t = tf.reduce_min(min_val)
        scale8 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8 = tf.clip_by_value(scale8, self.eps, 10000.)
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - tf.round(min_val_t / scale8)
        zero_point = tf.clip_by_value(zero_point, qmin, qmax)
        #scale_mask = tf.ones_like(max_val)
        #print("[DEBUG] scale_mask.shape: ", scale_mask.shape)
        print("[DEBUG] inputs.shape[2]: ", inputs.shape[2])
        scale_mask = []
        for j in range(inputs.shape[2]):
            data = tf.expand_dims(inputs[..., j], axis = -1)
            data_q1 = (tf.clip_by_value(tf.round(data / scale1 + zero_point),
                                        qmin, 
                                        qmax) -
                       zero_point) * scale1
            data_q2 = (tf.clip_by_value(tf.round(data / scale2 + zero_point),
                                        qmin, 
                                        qmax) -
                       zero_point) * scale2
            data_q4 = (tf.clip_by_value(tf.round(data / scale4 + zero_point),
                                        qmin, 
                                        qmax) -
                       zero_point) * scale4
            data_q8 = (tf.clip_by_value(tf.round(data / scale8 + zero_point),
                                        qmin, 
                                        qmax) -
                       zero_point) * scale8
            
            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score = [score1, score2, score4, score8]
            #scale_mask[j] *= 2**score.index(min(score))
            scale_mask.append(2**score.index(min(score)))
        scale_mask = tf.Variable(scale_mask, dtype = tf.float32)
        print("[DEBUG] scale1.dtype: ", scale1.dtype, " scale_mask.dtype: ", scale_mask.dtype)
        scale = scale1 * scale_mask
        print("[DEBUG] scale.shape: ", scale.shape, " zp.shape: ", zero_point.shape)
        return scale, zero_point
