import tensorflow as tf
from tensorflow import keras


def build_input(filters, kernel_size, input_layer):
    conv = keras.layers.Conv2D(
        filters=filters
        , kernel_size=kernel_size
        , padding='same'
        , use_bias=False
        , activation='linear'
    )(input_layer)
    norm = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.LeakyReLU()(norm)
    residual1 = ResidualLayer(filters, kernel_size)(relu)
    residual2 = ResidualLayer(filters, kernel_size)(residual1)
    residual3 = ResidualLayer(filters, kernel_size)(residual2)
    residual4 = ResidualLayer(filters, kernel_size)(residual3)
    residual5 = ResidualLayer(filters, kernel_size)(residual4)
    residual6 = ResidualLayer(filters, kernel_size)(residual5)
    residual7 = ResidualLayer(filters, kernel_size)(residual6)
    residual8 = ResidualLayer(filters, kernel_size)(residual7)
    return residual8


def build_policy(filters, kernel_size, hidden_size, num_actions, input_layer, base_input):
    p_conv = keras.layers.Conv2D(
        filters=filters
        , kernel_size=kernel_size
        , padding='same'
        , use_bias=False
        , activation='linear'
    )(input_layer)
    p_norm = keras.layers.BatchNormalization()(p_conv)
    p_relu = keras.layers.LeakyReLU()(p_norm)
    p_flatten = keras.layers.Flatten()(p_relu)
    p_dense1 = keras.layers.Dense(hidden_size * 2, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(p_flatten)
    p_dropout = keras.layers.Dropout(.2)(p_dense1)
    p_dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(p_dropout)
    p_out = keras.layers.Dense(num_actions, activation='linear',
                               kernel_initializer=keras.initializers.he_normal(), name="policy_output")(
        p_dense2)
    return p_out


def build_value(filters, kernel_size, hidden_size, input_layer):
    v_conv = keras.layers.Conv2D(
        filters=filters
        , kernel_size=kernel_size
        , padding='same'
        , use_bias=False
        , activation='linear'
    )(input_layer)
    v_norm = keras.layers.BatchNormalization()(v_conv)
    v_relu = keras.layers.LeakyReLU()(v_norm)
    v_flatten = keras.layers.Flatten()(v_relu)
    v_dense1 = keras.layers.Dense(hidden_size * 2, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(v_flatten)
    v_dropout = keras.layers.Dropout(.2)(v_dense1)
    v_dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(v_dropout)
    v_out = keras.layers.Dense(1, activation='tanh', kernel_initializer=keras.initializers.he_normal(),
                               name='value_output')(v_dense2)
    return v_out


def get_net(filters, kernel_size, hidden_size, out_filters, out_kernel_size, num_action, input_shape):
    input_tensor = keras.Input(shape=input_shape)
    base_model = build_input(filters, kernel_size, input_tensor)
    model = keras.Model(input_tensor,
                        [build_policy(out_filters, out_kernel_size, hidden_size, num_action, base_model, input_tensor),
                         build_value(out_filters, out_kernel_size, hidden_size, base_model)])
    return model


class ResidualLayer(keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ResidualLayer, self).__init__()
        self.conv1 = keras.layers.Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , padding='same'
            , use_bias=False
            , activation='linear'
        )
        self.norm1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.LeakyReLU()
        self.conv2 = keras.layers.Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , padding='same'
            , use_bias=False
            , activation='linear'
        )
        self.norm2 = keras.layers.BatchNormalization()
        self.add = keras.layers.Add()
        self.relu2 = keras.layers.LeakyReLU()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.add([x, inputs])
        x = self.relu2(x)
        return x

    def get_config(self):
        return {"conv_filters": self.conv1.filters, "kernel_size": self.conv1.kernel_size}


def cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true
    minus_ones = tf.fill(dims=tf.shape(pi), value=-1.)
    where = tf.equal(pi, minus_ones)

    negatives = tf.fill(tf.shape(pi), -100.0)
    zeros = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    p = tf.where(where, negatives, p)
    pi = tf.where(where, zeros, pi)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss
