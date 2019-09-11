# -*- coding: utf-8 -*-
from keras.layers import Input
from keras.layers import GaussianNoise
from keras.layers import MaxPooling3D
from keras.layers import Conv3D
from keras.layers import ZeroPadding3D
from keras.layers import Conv3DTranspose


def Conv3D_block(inputs, filters, mode='unet', downsizing=None, loop=2):
    if downsizing:
        if downsizing == 'pooling':
            inputs = MaxPooling3D(pool_size=(2, 2, 2))(inputs)
        elif downsizing == 'conv2':
            inputs = Conv3D(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(inputs)
        elif downsizing == 'conv3':
            inputs = Conv3D(filters, (3, 3, 3), strides=(2, 2, 2), use_bias=False, padding='same')(inputs)

    x = inputs
    if 'unet' in mode:
        x = _base_block(x, filters, loop=loop)
    elif 'res' in mode:
        x = _residual_block(x, filters, mode=mode, loop=loop)
    else:
        pass
    return x

def UpConv3D_block(inputs, skip_input, filters, mode='unet', loop=2):
    if skip == 'attention':
        skip_input = _atten_gate(inputs, skip_input, filters)

    x = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(inputs)
    x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(x)
    x = self._norm(x)
    x = self._activation(x)
    x = self._crop_concat()([x, skip_input])
    x = self._conv3d(x, filters, mode=mode, downsizing=False, loop=loop)

def Unet(args):
    img_input = Input(shape=args.input_shape)

    if args.noise > 0:
        img_input = GaussianNoise(args.noise)(img_input)

    d1 = 