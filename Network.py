import tensorflow as tf
from tensorflow import keras


class Residual_Model(keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, input_shape: (int, int, int) = None):
        super(Residual_Model, self).__init__()
        self.conv = keras.layers.Conv2D(2, 3, activation='relu', input_shape=input_shape)
        self.norm = keras.layers.BatchNormalization()
        self.residual = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights=None,
                                                                input_shape=(input_shape[1], input_shape[2], 2),
                                                                pooling='max')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(hidden_size * 3, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(hidden_size * 4, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense3 = keras.layers.Dense(hidden_size * 5, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense4 = keras.layers.Dense(hidden_size * 4, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_dense1 = keras.layers.Dense(hidden_size * 3, activation='relu',
                                             kernel_initializer=keras.initializers.he_normal())
        self.adv_dense2 = keras.layers.Dense(hidden_size * 4, activation='relu',
                                             kernel_initializer=keras.initializers.he_normal())
        self.adv_dense3 = keras.layers.Dense(hidden_size * 3, activation='relu',
                                             kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(num_actions, activation='softmax',
                                          kernel_initializer=keras.initializers.he_normal())
        self.v_dense1 = keras.layers.Dense(hidden_size * 3, activation='relu',
                                           kernel_initializer=keras.initializers.he_normal())
        self.v_dense2 = keras.layers.Dense(hidden_size * 4, activation='relu',
                                           kernel_initializer=keras.initializers.he_normal())
        self.v_out = keras.layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.he_normal())

    def call(self, input, **kwargs):
        x = self.conv(input)
        x = self.norm(x)
        x = self.residual(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        adv = self.adv_dense1(x)
        adv = self.adv_dense2(adv)
        adv = self.adv_dense3(adv)
        adv = self.adv_out(adv)
        v = self.v_dense1(x)
        v = self.v_dense2(v)
        v = self.v_out(v)
        return adv, v

    @tf.function
    def traceable(self, input, **kwargs):
        return self(input, **kwargs)
