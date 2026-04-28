import tensorflow as tf
import tensorflow.keras as keras

weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)

def Dense(units, activation=None):
    return tf.keras.layers.Dense(units=units, activation=activation, use_bias=True,
                                  kernel_initializer=weights_initializer, bias_initializer='zeros')

def Conv1D(filters, kernel_size, strides=1, padding='valid', activation=None, use_bias=True):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides),
                                   padding=padding, data_format='channels_last', dilation_rate=1,
                                   activation=None, use_bias=use_bias,
                                   kernel_initializer=weights_initializer, bias_initializer='zeros')

Conv2D = tf.keras.layers.Conv2D

def DeConv1D(filters, kernel_size, strides=1, padding='valid', use_bias=True):
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size),
                                            strides=(1, strides), padding=padding,
                                            output_padding=None, data_format=None,
                                            dilation_rate=(1, 1), activation=None, use_bias=use_bias,
                                            kernel_initializer=weights_initializer, bias_initializer='zeros')

def BatchNormalization(trainable=True, virtual_batch_size=None):
    return tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                               center=True, scale=True,
                                               beta_initializer='zeros', gamma_initializer='ones',
                                               moving_mean_initializer='zeros',
                                               moving_variance_initializer='ones',
                                               trainable=trainable,
                                               virtual_batch_size=virtual_batch_size)

def Activation(x, activation):
    if activation == 'relu':
        return tf.keras.layers.ReLU()(x)
    elif activation == 'leaky_relu':
        return tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
    elif activation == 'sigmoid':
        return tf.keras.layers.Activation('sigmoid')(x)
    elif activation == 'softmax':
        return tf.keras.layers.Activation('softmax')(x)
    elif activation == 'tanh':
        return tf.keras.layers.Activation('tanh')(x)
    else:
        raise ValueError('please check the name of the activation')

def Dropout(rate):
    return tf.keras.layers.Dropout(rate=rate)

def flatten():
    return tf.keras.layers.Flatten(data_format=None)

# ── InstanceNormalization replacement (no tfa required) ────────────────────
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],), initializer='ones',  name='gamma')
        self.beta  = self.add_weight(shape=(input_shape[-1],), initializer='zeros', name='beta')
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(var + self.epsilon) + self.beta

def normalization(name):
    if name == 'none':
        return lambda x: x
    elif name == 'batch_norm':
        return keras.layers.BatchNormalization()
    elif name == 'instance_norm':
        return InstanceNormalization()
    elif name == 'layer_norm':
        return keras.layers.LayerNormalization()

def attention_block_1d(curr_layer, conn_layer):
    inter_channel = curr_layer.get_shape().as_list()[3]
    theta_x = Conv1D(inter_channel, 1, 1)(conn_layer)
    phi_g   = Conv1D(inter_channel, 1, 1)(curr_layer)
    f       = Activation(keras.layers.add([theta_x, phi_g]), 'relu')
    psi_f   = Conv1D(1, 1, 1)(f)
    rate    = Activation(psi_f, 'sigmoid')
    att_x   = keras.layers.multiply([conn_layer, rate])
    return att_x