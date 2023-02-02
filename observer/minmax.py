import tensorflow as tf

from .base import BaseObserver


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed
        self.max = 1000.

    def update(self, v, channel_first):
        #print("[DEBUG] before v shape: ", v.shape)
        v = self.reshape_tensor(v, channel_first)
        #print("[DEBUG] update v shape: ", v.shape)
        #cur_max = v.max(axis=1).values
        cur_max = tf.reduce_max(v, axis = 1)
        #print("[DEBUG] cur_max.shape: ", cur_max.shape)
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = tf.math.maximum(cur_max, self.max_val)
        #cur_min = v.min(axis=1).values
        cur_min = tf.reduce_min(v, axis = 1)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = tf.math.minimum(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = tf.reduce_max(self.max_val)
            self.min_val = tf.reduce_min(self.min_val)
            #print("[DEBUG] layerwise self.max_val.shape: ", self.max_val.shape)
        
        print("[DEBUG] module type: ", self.module_type, " max_val.shape: ", self.max_val.shape)
        #if self.module_type == 'conv_weight':
        #    sys.exit()

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        
        print("[DEBUG] max_val.shape: ", max_val.shape)
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        
        #print("[DEBUG] max_val.shape: ", max_val.shape)
        scale = tf.ones_like(max_val, dtype=tf.float32)
        #zero_point = tf.zeros_like(max_val, dtype=tf.int64)
        zero_point = tf.zeros_like(max_val, dtype=tf.float32)
        
        if self.symmetric:
            max_val = tf.math.maximum(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale = tf.clip_by_value(scale, clip_value_min = self.eps, clip_value_max = self.max)
            #scale.clamp_(self.eps)
            #zero_point = tf.zeros_like(max_val, dtype=tf.int64)
            zero_point = tf.zeros_like(max_val, dtype=tf.float32)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = tf.clip_by_value(scale, clip_value_min = self.eps, clip_value_max = self.max)
            #scale.clamp_(self.eps)
            zero_point = qmin - tf.round(min_val / scale)
            zero_point = tf.clip_by_value(zero_point, qmin, qmax)
            
        print("[DEBUG] get_quant_params scale.shape: ", scale.shape, " zero_p.shape: ", zero_point.shape)

        return scale, zero_point