import tensorflow as tf

from .base import BaseObserver


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed
        self.max = 1000.

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
        
    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        
        scale = tf.ones_like(max_val, dtype=tf.float32)
        zero_point = tf.zeros_like(max_val, dtype=tf.float32)
        
        if self.symmetric:
            max_val = tf.math.maximum(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale = tf.clip_by_value(scale, clip_value_min = self.eps, clip_value_max = self.max)
            zero_point = tf.zeros_like(max_val, dtype=tf.float32)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = tf.clip_by_value(scale, clip_value_min = self.eps, clip_value_max = self.max)
            zero_point = qmin - tf.round(min_val / scale)
            zero_point = tf.clip_by_value(zero_point, qmin, qmax)
            
        return scale, zero_point