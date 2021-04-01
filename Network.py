import tensorflow as tf
from tensorflow import keras


def build_input(filters, graph, num_residual, input_layer):
    conv = CustomConvLayer(filters, graph)(input_layer)
    norm = keras.layers.BatchNormalization(axis=1)(conv)
    residual = keras.layers.LeakyReLU()(norm)
    for i in range(num_residual):
        residual = ResidualLayer(filters, graph)(residual)
    return residual


def build_policy(filters, graph, hidden_size, num_actions, input_layer, base_input):
    p_conv = CustomConvLayer(filters, graph)(input_layer)
    p_norm = keras.layers.BatchNormalization(axis=1)(p_conv)
    p_relu = keras.layers.LeakyReLU()(p_norm)
    p_flatten = keras.layers.Flatten()(p_relu)
    p_dense1 = keras.layers.Dense(hidden_size * 2, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(p_flatten)
    p_dropout = keras.layers.Dropout(.2)(p_dense1)
    p_dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(p_dropout)
    p_all_actions = keras.layers.Dense(num_actions, activation='linear',
                                       kernel_initializer=keras.initializers.he_normal())(
        p_dense2)

    p_out = SliceLayer(name="policy_output")(
        [p_all_actions, base_input[:, 0, 3]])
    return p_out


def build_value(filters, graph, hidden_size, input_layer):
    v_conv = CustomConvLayer(filters, graph)(input_layer)
    v_norm = keras.layers.BatchNormalization(axis=1)(v_conv)
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


def get_net(filters, hidden_size, out_filters, num_action, input_shape, graph, num_residual):
    input_tensor = keras.Input(shape=input_shape)
    base_model = build_input(filters, graph, num_residual, input_tensor)
    model = keras.Model(input_tensor,
                        [build_policy(out_filters, graph, hidden_size, num_action, base_model, input_tensor),
                         build_value(out_filters, graph, hidden_size, base_model)])
    return model


class ResidualLayer(keras.layers.Layer):
    def __init__(self, filters, graph):
        super(ResidualLayer, self).__init__()
        self.conv1 = CustomConvLayer(filters, graph)
        self.norm1 = keras.layers.BatchNormalization(axis=1)
        self.relu1 = keras.layers.LeakyReLU()
        self.conv2 = CustomConvLayer(filters, graph)
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

    def get_config(self):
        return {}


class SliceLayer(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        moveNeeded = inputs[1]
        inputs = inputs[0]
        out_tensor = inputs[:, 0:24]
        pos_zeros = tf.reshape(tf.equal(moveNeeded, tf.constant(0.)), (-1, 1))
        pos_ones = tf.reshape(tf.equal(moveNeeded, tf.constant(1.)), (-1, 1))
        pos_twos = tf.reshape(tf.equal(moveNeeded, tf.constant(2.)), (-1, 1))
        pos_threes = tf.reshape(tf.equal(moveNeeded, tf.constant(3.)), (-1, 1))
        out_tensor = tf.where(pos_zeros, inputs[:, 0:24], out_tensor)
        out_tensor = tf.where(pos_ones, inputs[:, 24:48], out_tensor)
        out_tensor = tf.where(pos_twos, tf.concat([inputs[:, 72:76], out_tensor[:, 0:20]], axis=1), out_tensor)
        out_tensor = tf.where(pos_threes, inputs[:, 48:72], out_tensor)
        return out_tensor


class CustomConvLayer(keras.layers.Layer):
    def __init__(self, filters, graph, **kwargs):
        import stellargraph
        super(CustomConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.conv = stellargraph.layer.GraphConvolution(units=filters, activation='linear', use_bias=False)
        self.graph = tf.constant(graph) / 5

    def call(self, inputs, **kwargs):
        graph = tf.multiply(tf.ones(shape=(tf.shape(inputs)[0], 24, 24)), self.graph)
        return self.conv([inputs, graph])

    def get_config(self):
        config = super(CustomConvLayer, self).get_config()
        config.update({"filters": self.filters, "graph": self.graph.numpy()})
        return config




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
