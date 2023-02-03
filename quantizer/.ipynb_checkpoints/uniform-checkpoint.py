import sys
import tensorflow as tf

from .base import BaseQuantizer

class UniformQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale = None
        self.zero_point = None

    def update_quantization_params(self, *args, **kwargs):
        #print("[DEBUG] update_quantization_params")
        self.scale, self.zero_point = self.observer.get_quantization_params(
            *args, **kwargs)

    def quant(self, inputs, scale=None, zero_point=None):
        print("***************[DEBUG] quant start*****************")
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
            
        range_shape = self.get_reshape_range(inputs)
        #print("[DEBUG] module_type: ", self.module_type)
        print("[DEBUG] inputs.shape: ", inputs.shape)
        print("[DEBUG] before scale shape: ", scale.shape)
        print("[DEBUG] range_shape: ", range_shape)
        scale = tf.reshape(scale, shape = range_shape)
        #scale = scale.reshape(range_shape)
        print("[DEBUG] after scale shape: ", scale.shape)
        zero_point = tf.reshape(zero_point, shape = range_shape)
        #zero_point = zero_point.reshape(range_shape)
        inter = inputs / scale
        print("[DEBUG] inputs.shape: ", inputs.shape, " scale.shape: ", scale.shape, " zero_point.shape: ", zero_point.shape)
        print("[DEBUG] zero_point: ", zero_point)
        print("[DEBUG] scale: ", scale)
        print("[DEBUG] inter.shape: ", inter.shape)
        outputs = inputs / scale + zero_point
        print("[DEBUG] outputs.shape: ", outputs.shape)
        #sys.exit()
        #outputs = inputs / scale + zero_point
        outputs = tf.clip_by_value(tf.round(outputs),
                                   self.bit_type.lower_bound,
                                   self.bit_type.upper_bound)
        
        print("***************[DEBUG] quant end*****************")
        
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = tf.reshape(scale, shape = range_shape)
        zero_point = tf.reshape(zero_point, shape = range_shape)
        #zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs