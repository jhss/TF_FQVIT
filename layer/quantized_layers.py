import sys

import tensorflow as tf

from quantizer import BIT_TYPE_DICT
from observer import build_observer
from quantizer import build_quantizer

class QAct(tf.keras.layers.Layer):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def call(self, x, channel_first = False):
        #print("[DEBUG] QAct calibrate: ", self.calibrate)
        if self.calibrate:
            #print("[DEBUG] calibrate start")
            self.quantizer.observer.update(x, channel_first)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x, channel_first)
        if not self.quant:
            return x
        x = self.quantizer(x, channel_first)
        return x
    
    
class QLinear(tf.keras.layers.Dense):

    def __init__(self,
                 units, 
                 use_bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 name = None,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        #print("[DEBUG] QLinear name: ", name)
        #sys.exit()
        super(QLinear, self).__init__(
                units = units, 
                use_bias = use_bias,
                name = name)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def call(self, x, channel_first = False):
        #print("[DEBUG] QLinear calibrate: ", self.calibrate)
        if self.calibrate:
            #print("------------------------[DEBUG] QLinear calibrate start------: ", self.calibrate)
            self.quantizer.observer.update(self.kernel, channel_first)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x, channel_first)
        if not self.quant:
            return super().call(x)
        self.kernel = self.quantizer(self.kernel, channel_first)
        
        return super().call(x)
    
class QConv2d(tf.keras.layers.Conv2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer= None,
                 name= None,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name = name,
            groups=groups,
            use_bias=use_bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def call(self, x, channel_first = False):
        if self.calibrate:
            self.quantizer.observer.update(self.kernel, channel_first)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x, channel_first)
        if not self.quant:
            return super().call(x)
        self.kernel = self.quantizer(self.kernel, channel_first)
        return super().call(x)


class QuantizedLayerNorm(tf.keras.layers.LayerNormalization):
    def __init__(self, name, eps):
        super(QuantizedLayerNorm, self).__init__(name = name, epsilon = eps)
        self.quant = 'normal'
        #self.cnt = 0

    def get_MN(self, x):
        bit = 7
        N = tf.clip_by_value(bit - tf.floor(tf.experimental.numpy.log2(x)),
                             0, 31)
        M = tf.clip_by_value(tf.floor(x * tf.pow(2, N)),
                             0, 2**(bit+1) - 1)
        return M, N

    def call(self, x, training = False, in_quantizer = None, out_quantizer = None, in_scale_expand = 1, last = False):
        if self.quant == 'normal':
            print("--------------------[DEBUG] layernorm normal mode-----------------")
            outputs = super().call(x)
            #if last == True and x.shape[0] == 128:
            #    sys.exit()
            #self.cnt += 1
        else:
            print("-------------------[DEBUG] layer_norm int mode---------------")
            #sys.exit()
            print("-------------------[DEBUG] layer_norm quant start---------------")
            in_scale = in_quantizer.scale
            out_scale = out_quantizer.scale

            channel_nums = x.shape[-1]
            print("[DEBUG] layer norm first in_scale.shape: ", in_scale.shape)
            print("[DEBUG] layer norm out_scale.shape: ", out_scale.shape)
            in_scale = in_scale[tf.newaxis, tf.newaxis, :]
            print("[DEBUG] layer norm last in_scale.shape: ", in_scale.shape)
            print("[DEBUG] in_scale: ", in_scale)
            #out_scale = out_scale[tf.newaxis, tf.newaxis, :]
            out_scale = tf.reshape(out_scale, shape = (1, 1, -1) )
            print("[DEBUG] layer norm last out_scale.shape: ", out_scale.shape)
            x_q = tf.round(x / in_scale)
            in_scale_min = tf.reduce_min(in_scale)
            in_scale_mask = tf.round(in_scale / in_scale_min)

            x_q = x_q * in_scale_mask

            x_q_square_sum = tf.reduce_sum(tf.pow(x_q, 2), axis = -1)
            x_q_sum_square = tf.pow(tf.math.reduce_sum(x_q, axis = -1), 2)

            x_q_mean = tf.reduce_mean(x_q, axis = -1) * in_scale_min
            x_q_std  = (in_scale_min / channel_nums) * tf.sqrt(
                            channel_nums * x_q_square_sum - x_q_sum_square)
            
            inter = (in_scale_min / x_q_std)[:, :, tf.newaxis]
            print("[DEBUG] in_scale_min.shape: ", in_scale_min.shape, " x_q_std.shape: ", x_q_std.shape)
            print("[DEBUG] inter.shape: ", inter.shape)
            print("[DEBUG] self.gamma.shape: " , self.gamma.shape, " newaxis: ", self.gamma[tf.newaxis, tf.newaxis, :].shape)
            A = (in_scale_min / x_q_std)[:, :, tf.newaxis] * self.gamma[tf.newaxis, tf.newaxis, :] / out_scale
            
            A_sign = tf.sign(A)
            M, N = self.get_MN(tf.abs(A))

            B = tf.round((self.beta[tf.newaxis, tf.newaxis, :] - (x_q_mean / x_q_std)[:,:,tf.newaxis] *
                          self.gamma[tf.newaxis, tf.newaxis, :]) / out_scale * tf.pow(2, N))
            
            print("[DEBUG] A_sign: ", A_sign[0,0,0:3])
            print("[DEBUG] M: ", M[0,0,0:3])
            print("[DEBUG] B: ", B[0,0,0:3])
            print("[DEBUG] x_q: ", x_q[0,0,0:3])
            x_q = tf.round((A_sign * M * x_q + B) / tf.pow(2, N))
            
            print("[DEBUG] x_q.shape: ", x_q.shape)
            outputs = x_q * out_scale
            print("[DEBUG] out_scale ", out_scale)
            print("[DEBUG] outputs: ", outputs.shape)
            print("[DEBUG] outputs: ", outputs[0,0,0:3])
            #sys.exit()
            print("-------------------[DEBUG] layer_norm quant end---------------")
        return outputs
    