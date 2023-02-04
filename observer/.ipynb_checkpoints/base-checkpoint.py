import tensorflow as tf

class BaseObserver:

    def __init__(self, module_type, bit_type, calibration_mode):
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.max_val = None
        self.min_val = None
        self.eps = 1e-9
        
    def reshape_tensor(self, v, channel_first):
        if not isinstance(v, tf.Tensor):
            v = tf.convert_to_tensor(v)
        v = tf.stop_gradient(v)
        if self.module_type == 'linear_weight':
            v = tf.reshape(v, shape = (v.shape[0], -1))
            v = tf.transpose(v, [1, 0])
        elif self.module_type == 'conv_weight':
            v = tf.reshape(v, shape = (-1, v.shape[-1]))
            v = tf.transpose(v, [1,0])
        elif self.module_type == 'activation':
            if channel_first == True:
                v = tf.transpose(v, [1,0,2,3])
                v = tf.reshape(v, shape = (v.shape[0], -1))
            else:
                v = tf.reshape(v, shape = (-1, v.shape[-1]))
                v = tf.transpose(v, [1, 0])
        else:
            raise NotImplementedError
        return v

    def update(self, v, channel_first):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError