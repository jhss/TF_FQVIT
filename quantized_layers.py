import tensorflow as tf

class QuantizedLinear(tf.keras.layers.Dense):
    def __init__(self, units, use_bias = True, use_quant = False):
        super(QuantizedLinear, self).__init__(units = units, use_bias = use_bias)

        self.use_bias = use_bias
        self.use_quant = use_quant

    def call(self, inputs):
        # bias quantization은 안하나??


        if self.use_quant:
            kernel = self.quantizer(self.kernel)
        else:
            kernel = self.kernel

        outputs = tf.matmul(a = inputs, b = kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

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


    def call(self, inputs, scale):

        
