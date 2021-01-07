import tensorflow as tf
from tensorflow import keras


class Residual_Model(keras.Model):
    def __init__(self, hidden_size: int, num_actions: int):
        super(Residual_Model, self).__init__()
        self.conv = keras.layers.Conv2D(
            filters=96
            , kernel_size=3
            , padding='same'
            , use_bias=False
            , activation='linear'
        )
        self.norm = keras.layers.BatchNormalization(axis=1)
        self.relu = keras.layers.LeakyReLU()
        self.residual1 = ResidualLayer(96, 3)
        self.residual2 = ResidualLayer(96, 3)
        self.residual3 = ResidualLayer(96, 3)
        self.residual4 = ResidualLayer(96, 3)
        self.residual5 = ResidualLayer(96, 3)
        self.residual6 = ResidualLayer(96, 3)
        self.residual7 = ResidualLayer(96, 3)
        self.residual8 = ResidualLayer(96, 3)
        self.residual9 = ResidualLayer(96, 3)
        self.residual10 = ResidualLayer(96, 3)
        self.p_conv = keras.layers.Conv2D(
            filters=4
            , kernel_size=1
            , padding='same'
            , use_bias=False
            , activation='linear'
        )
        self.p_norm = keras.layers.BatchNormalization(axis=1)
        self.p_relu = keras.layers.LeakyReLU()
        self.p_flatten = keras.layers.Flatten()
        self.p_dense1 = keras.layers.Dense(hidden_size * 2, activation='relu',
                                           kernel_initializer=keras.initializers.he_normal())
        self.p_dropout = keras.layers.Dropout(.2)
        self.p_dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                           kernel_initializer=keras.initializers.he_normal())
        self.p_out = keras.layers.Dense(num_actions, activation='softmax',
                                        kernel_initializer=keras.initializers.he_normal())
        self.v_conv = keras.layers.Conv2D(
            filters=4
            , kernel_size=1
            , padding='same'
            , use_bias=False
            , activation='linear'
        )
        self.v_norm = keras.layers.BatchNormalization(axis=1)
        self.v_relu = keras.layers.LeakyReLU()
        self.v_flatten = keras.layers.Flatten()
        self.v_dense1 = keras.layers.Dense(hidden_size * 2, activation='relu',
                                           kernel_initializer=keras.initializers.he_normal())
        self.v_dropout = keras.layers.Dropout(.2)
        self.v_dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                           kernel_initializer=keras.initializers.he_normal())
        self.v_out = keras.layers.Dense(1, activation='tanh', kernel_initializer=keras.initializers.he_normal())

    def call(self, input, **kwargs):
        x = self.conv(input)
        x = self.norm(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.residual6(x)
        x = self.residual7(x)
        x = self.residual8(x)
        x = self.residual9(x)
        x = self.residual10(x)
        pol = self.p_conv(x)
        pol = self.p_norm(pol)
        pol = self.p_relu(pol)
        pol = self.p_flatten(pol)
        pol = self.p_dense1(pol)
        pol = self.p_dropout(pol)
        pol = self.p_dense2(pol)
        pol = self.p_out(pol)
        v = self.v_conv(x)
        v = self.v_norm(v)
        v = self.v_relu(v)
        v = self.v_flatten(v)
        v = self.v_dense1(v)
        v = self.v_dropout(v)
        v = self.v_dense2(v)
        v = self.v_out(v)
        return pol, v

    @tf.function
    def traceable(self, input, **kwargs):
        return self(input, **kwargs)


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
        self.norm1 = keras.layers.BatchNormalization(axis=1)
        self.relu1 = keras.layers.LeakyReLU()
        self.conv2 = keras.layers.Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , padding='same'
            , use_bias=False
            , activation='linear'
        )
        self.norm2 = keras.layers.BatchNormalization(axis=1)
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
