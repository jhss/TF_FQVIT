import tensorflow as tf

from .base import BaseQuantizer

class Log2Quantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(Log2Quantizer, self).__init__(
            bit_type,
            observer,
            module_type,
        )
        self.softmax_mask = None

    def quant(self, inputs):
        rounds = tf.round(-1 * tf.experimental.numpy.log2(inputs))
        self.softmax_mask = rounds >= 2**self.bit_type.bits
        outputs = tf.clip_by_value(rounds, 0, 2**self.bit_type.bits - 1)
        return outputs

    def dequantize(self, inputs):
        outputs = 2**(-1 * inputs)
        outputs = tf.where(self.softmax_mask, 0, outputs)
        #outputs[self.softmax_mask] = 0
        return outputs