import tensorflow as tf

from quantizer import BIT_TYPE_DICT
from observer import build_observer
from quantizer import build_quantizer

@tf.custom_gradient
def ste_round(x):
    x = tf.math.round(x)
    def grad(dy):
        return dy
    return x, grad

@tf.custom_gradient
def ste_floor(x):
    x = tf.math.floor(x)
    def grad(dy):
        return dy
    return x, grad

class QuntizedAct(tf.keras.layers.Layer):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QunaitzedAct, self).__init__()

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

    def call(self, x):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x


class QuantizedLinear(tf.keras.layers.Dense):

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
        super(QuantizedLinear, self).__init__(
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

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return super().call(x)
        self.kernel = self.quantizer(self.kernel)

        return super().call(x)

class QuantizedConv2d(tf.keras.layers.Conv2D):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QuantizedConv2d, self).__init__(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            groups=groups,
            use_bias=bias,
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

    def call(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.kernel)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return super().call(x)
        self.kernel = self.quantizer(self.kernel)
        return super().call(x)

class QuantizedLayerNorm(tf.keras.layers.LayerNormalization):
    def __init__(self, eps):
        super(QuantizedLayerNorm, self).__init__(epsilon = eps)
        self.mode = 'normal'

    def get_MN(self, x):
        bit = 7
        N = tf.clip_by_value(bit - tf.floor(tf.experimental.numpy.log2(x)),
                             0, 31)
        M = tf.clip_by_value(tf.floor(x * tf.pow(2, N)),
                             0, 2**(bit+1) - 1)
        return M, N

    def call(self, x, in_quantizer = None, out_quantizer = None, in_scale_expand = 1):
        if self.mode == 'normal':
            outputs = super().call(x)
        else:
            in_scale = in_quantizer.scale
            out_scale = out_quantizer.scale

            channel_nums = x.shape[-1]
            in_scale = in_scale[tf.newaxis, tf.newaxis, :]
            out_scale = out_scale[tf.newaxis, tf.newaxis, :]

            x_q = tf.round(x / in_scale)
            in_scale_min = tf.reduce_min(in_scale)

            x_q_square_sum = tf.reduce_sum(tf.pow(x_q, 2), axis = -1)
            x_q_sum_square = tf.pow(tf.sum(x_q, axis = -1), 2)

            x_q_mean = tf.reduce_mean(x_q, axis = -1) * in_scale_min
            x_q_std  = (in_scale_min / channel_nums) * tf.sqrt(
                            channel_nums * x_q_square_sum - x_q_sum_square)

            A = (in_scale_min / x_q_std)[:, tf.newaxis] * self.gamma[tf.newaxis, tf.newaxis, :] / out_scale
            A_sign = tf.sign(A)
            M, N = self.get_MN(tf.abs(A))

            B = tf.round(self.beta[tf.newaxis, tf.newaxis, :] - (x_q_mean / x_q_std)[:,:,tf.newaxis]) *
                         self.gamma[tf.newaxis, tf.newaxis, :] / out_scale * tf.pow(2, N))

            x_q = tf.round((A_sign * M * x_q + B) / tf.pow(2, N))
            outputs = x_q * out_scale

        return outputs

class QuantizedSoftmax(tf.keras.layers.Layer):
    def __init__(self, ):
        super(QuantizedSoftmax, self).__init__()
        self.max_bits = 32

        self.log2 = tf.math.log([2.])
        self.coeff = [0.35815147, 0.96963238, 1.0]
        self.m_ln2 = -0.6931 # -ln2
        self.bound = 30

        self.coeff[1] /= self.coeff[0]
        self.coeff[2] /= self.coeff[0]

    # [Stage]
    def log2_quant(self, x):

        log_x = tf.math.log(x)
        # [Confused] whether use tf.floor or ste_floor
        msb = tf.floor(tf.math.divide(log_x, self.log2))
        remainder = (x - 2**msb) >= 2 **(msb-1)
        int_log_x = tf.where(remainder, msb+1, msb)

        return int_log_x

    # [Stage]
    def int_polynomial(self, x_int, scale_factor):

        b_int = tf.stop_gradient(tf.floor(self.coeff[1] / scale_factor))
        c_int = tf.stop_gradient(tf.floor(self.coeff[2] / scale_factor**2))

        z = (x_int + b_int) * x_int + c_int
        scale_factor = self.coeff[0] * scale_factor**2

        return z, scale_factor

    # [Stage]
    def int_exp(self, x_int, scale_factor):
        scale_ln2 = tf.stop_gradient(tf.floor(self.m_ln2 / scale_factor))
        x_int = tf.math.maximum(x_int, self.bound * scale_ln2)

        q = ste_floor(x_int / scale_ln2)
        r = x_int - scale_ln2 * q

        # approximate 'exp(p)' with a polynomial function.
        exp_int, exp_scale_factor = self.int_polynomial(r, scale_factor)

        exp_int = ste_floor(exp_int * 2 ** (self.bound - q))
        scale_factor = exp_scale_factor / (2 ** self.bound)

        return exp_int, scale_factor

    # [Stage]
    def int_inverse_softmax(self, x, scale_factor):

        x = x / scale_factor
        x_hat = x - tf.math.reduce_max(x, axis = -1, keepdims = True)

        x_exp, scale_exp = self.int_exp(x_hat, scale_factor)
        x_exp_sum = tf.math.reduce_sum(x_exp, axis = -1, keepdims = True)

        inverse_softmax = ste_round(tf.math.divide(x_exp_sum, x_exp))

        return inverse_softmax

    # [Stage]
    def call(self, inputs, scale):

        inv_softmax = self.int_inverse_softmax(inputs, scale)
        log2_quant_inv_softmax = self.log2_quant(inv_softmax)

        mask = log2_quant_inv_softmax >= 2**self.max_bits
        clamped = tf.clip_by_value(log2_quant_inv_softmax, 0, 2**self.max_bits - 1)
        softmax = 2**(-clamped)
        softmax = torch.where(mask, clamped, 0)

        return softmax
