# Author: Soheil Khorram
# License: Simplified BSD

from network import Network
from keras.layers import Input, Conv1D, Add, multiply, add, Activation
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam


class DilatedNetwork(Network):
    def __init__(self):
        super().__init__()

    @staticmethod
    def load_parameters(parser):
        parser.add_argument('-layer-num', default=5, type=int)
        parser.add_argument('-kernel-size', default=7, type=int)
        parser.add_argument('-kernel-num', default=64, type=int)
        parser.add_argument('-l2', default=0.0, type=float)
        parser.add_argument('-apply-end-relu', default=1, type=int)
        parser.add_argument('-bridge', default='mul', type=str)
        parser.add_argument('-padding', default='causal', type=str)

    @staticmethod
    def get_name():
        return 'dilated'

    @staticmethod
    def dilated_conv(prm, x, dilation_rate):
        return Conv1D(filters=prm.kernel_num,
                    kernel_size=prm.kernel_size,
                    strides=1,
                    padding=prm.padding,
                    data_format='channels_last',
                    dilation_rate=dilation_rate,
                    activation=prm.activation,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=l2(prm.l2),
                    bias_regularizer=l2(prm.l2),
                    activity_regularizer=None)(x)

    @staticmethod
    def last_conv(prm, x):
        return Conv1D(filters=prm.inp_dim,
                    kernel_size=prm.kernel_size,
                    strides=1,
                    padding=prm.padding,
                    data_format='channels_last',
                    dilation_rate=1,
                    activation='linear',
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=l2(prm.l2),
                    bias_regularizer=l2(prm.l2),
                    activity_regularizer=None)(x)

    def construct(self, prm):
        """
        Constructs the network graph and stores it in self.net
        """
        inp = Input(shape=(prm.inp_len, prm.inp_dim))
        x = inp
        x = DilatedNetwork.dilated_conv(prm, x, 1)
        for i in range(prm.feature_layer_num - 1):
            y = DilatedNetwork.dilated_conv(prm, x, 1)
            if prm.skip == 1:
                x = Add()([x, y])
            else:
                x = y
        for i in range(prm.dilated_layer_num):
            y = DilatedNetwork.dilated_conv(prm, x, 2 ** (i + 1))
            if prm.skip == 1:
                x = Add()([x, y])
            else:
                x = y
        x = DilatedNetwork.last_conv(prm, x)
        outp = None
        if prm.bridge == 'nothing':
            outp = x
        elif prm.bridge == 'add':
            outp = add([x, inp])
        elif prm.bridge == 'mul':
            outp = multiply([x, inp])
        else:
            print('ERROR in bridge param')
        if prm.apply_end_relu == 1:
            outp = Activation('relu')(outp)
        net = Model(inputs=inp, outputs=outp)
        net.compile(optimizer=Adam(lr=prm.step_size),
                    loss='mean_squared_error')
        self.net = net
