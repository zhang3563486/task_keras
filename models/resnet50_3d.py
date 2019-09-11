import keras
from keras.layers import GlobalAveragePooling3D
from keras.layers import ZeroPadding3D
from keras.layers import MaxPooling3D
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import Add

from keras.models import Model

from .layers import *

def ResNet503D(
    input_shape,
    norm='bn',
    activation='relu',
    noise,
    base_filter
):
    def _conv_block(inputs, filters, strides=(2, 2, 2), norm='bn', activation='relu'):
        x = Conv3D(filters[0], (1, 1, 1), strides=strides)(inputs)
        x = _normalization(x, norm=norm)
        x = _activation(x, activation=activation)

        x = Conv3D(filters[1], (3, 3, 3), padding='same')(x)
        x = _normalization(x, norm=norm)
        x = _activation(x, activation=activation)

        x = Conv3D(filters[2], (1, 1, 1))(x)
        x = _normalization(x, norm=norm)

        shortcut = Conv3D(filters[2], (1, 1, 1), strides=strides)(inputs)
        shortcut = _normalization(x, norm=norm)

        x = Add()([x, shortcut])
        x = _activation(x, activation=activation)
        
        return x

    def _identity_block(inputs, filters):
        x = Conv3D(filters[0], (1, 1, 1))(inputs)
        x = _normalization(x, norm=norm)
        x = _activation(x, activation=activation)

        x = Conv3D(filters[1], (3, 3, 3), padding='same')(x)
        x = _normalization(x, norm=norm)
        x = _activation(x, activation=activation)

        x = Conv3D(filters[2], (1, 1, 1))(x)
        x = _normalization(x, norm=norm)

        x = Add()([x, inputs])
        x = _activation(x, activation=activation)
        
        return x

    img_input = Input(shape=input_shape)
    c0 = GaussianNoise(noise)(img_input)

    x = ZeroPadding3D(padding=(3, 3, 3))(c0)
    x = Conv3D(base_filter, (7, 7, 7), strides=(2, 2, 2), padding='valid')(x)
    x = _normalization(x, norm=norm)
    x = _activation(x, activation=activation)
    x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)

    x = _conv_block(x, [base_filter, base_filter, base_filter*4], strides=(1, 1, 1))
    x = _identity_block(x, [base_filter, base_filter, base_filter*4])
    x = _identity_block(x, [base_filter, base_filter, base_filter*4])

    x = _conv_block(x, [base_filter*2, base_filter*2, base_filter*8])
    x = _identity_block(x, [base_filter*2, base_filter*2, base_filter*8])
    x = _identity_block(x, [base_filter*2, base_filter*2, base_filter*8])
    x = _identity_block(x, [base_filter*2, base_filter*2, base_filter*8])

    x = _conv_block(x, [base_filter*4, base_filter*4, base_filter*16])
    x = _identity_block(x, [base_filter*4, base_filter*4, base_filter*16])
    x = _identity_block(x, [base_filter*4, base_filter*4, base_filter*16])
    x = _identity_block(x, [base_filter*4, base_filter*4, base_filter*16])
    x = _identity_block(x, [base_filter*4, base_filter*4, base_filter*16])
    x = _identity_block(x, [base_filter*4, base_filter*4, base_filter*16])

    x = _conv_block(x, [base_filter*8, base_filter*8, base_filter*32])
    x = _identity_block(x, [base_filter*8, base_filter*8, base_filter*32])
    x = _identity_block(x, [base_filter*8, base_filter*8, base_filter*32])

    dense = GlobalAveragePooling3D()(x)
    logits = Dense(2, activation='softmax')(dense)

    model = Model(img_input, logits, name=model)

    return model