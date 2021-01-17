def build_input(filters, kernel_size, input_layer):
    from tensorflow import keras
    class ResidualLayer(keras.layers.Layer):
        def __init__(self, filters, kernel_size):
            from tensorflow import keras
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

    conv = keras.layers.Conv2D(
        filters=filters
        , kernel_size=kernel_size
        , padding='same'
        , use_bias=False
        , activation='linear'
    )(input_layer)
    norm = keras.layers.BatchNormalization(axis=1)(conv)
    relu = keras.layers.LeakyReLU()(norm)
    residual1 = ResidualLayer(filters, kernel_size)(relu)
    residual2 = ResidualLayer(filters, kernel_size)(residual1)
    residual3 = ResidualLayer(filters, kernel_size)(residual2)
    residual4 = ResidualLayer(filters, kernel_size)(residual3)
    residual5 = ResidualLayer(filters, kernel_size)(residual4)
    residual6 = ResidualLayer(filters, kernel_size)(residual5)
    residual7 = ResidualLayer(filters, kernel_size)(residual6)
    residual8 = ResidualLayer(filters, kernel_size)(residual7)
    residual9 = ResidualLayer(filters, kernel_size)(residual8)
    residual10 = ResidualLayer(filters, kernel_size)(residual9)
    return residual10


def build_policy(filters, kernel_size, hidden_size, num_actions, input_layer):
    from tensorflow import keras
    p_conv = keras.layers.Conv2D(
        filters=filters
        , kernel_size=kernel_size
        , padding='same'
        , use_bias=False
        , activation='linear'
    )(input_layer)
    p_norm = keras.layers.BatchNormalization(axis=1)(p_conv)
    p_relu = keras.layers.LeakyReLU()(p_norm)
    p_flatten = keras.layers.Flatten()(p_relu)
    p_dense1 = keras.layers.Dense(hidden_size * 2, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(p_flatten)
    p_dropout = keras.layers.Dropout(.2)(p_dense1)
    p_dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                  kernel_initializer=keras.initializers.he_normal())(p_dropout)
    p_out = keras.layers.Dense(num_actions, activation='softmax',
                               kernel_initializer=keras.initializers.he_normal(), name='policy_output')(p_dense2)
    return p_out


def build_value(filters, kernel_size, hidden_size, input_layer):
    from tensorflow import keras
    v_conv = keras.layers.Conv2D(
        filters=filters
        , kernel_size=kernel_size
        , padding='same'
        , use_bias=False
        , activation='linear'
    )(input_layer)
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


def get_net(filters, kernel_size, hidden_size, out_filters, out_kernel_size, num_action, input_shape):
    from tensorflow import keras
    input_tensor = keras.Input(shape=input_shape)
    base_model = build_input(filters, kernel_size, input_tensor)
    model = keras.Model(input_tensor,
                        [build_policy(out_filters, out_kernel_size, hidden_size, num_action, base_model),
                         build_value(out_filters, out_kernel_size, hidden_size, base_model)])
    model.compile(optimizer='adam', loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mse'},
                  loss_weights=[0.5, 0.5],
                  metrics=['accuracy'])
    return model
