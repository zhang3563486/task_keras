import keras
import tensorflow as tf

import keras.backend as K
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import MaxPooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import ZeroPadding3D
from keras.layers import Conv3D
from keras.layers import Conv3DTranspose
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Multiply
from keras.layers import BatchNormalization
from keras_contrib.layers import InstanceNormalization
from keras_contrib.layers import GroupNormalization

__all__ = ['_basic_block', '_unet_upconv_block']

def _normalization(inputs, norm='bn'):
    if norm == 'bn':
        return BatchNormalization()(inputs)
    elif norm == 'in':
        return InstanceNormalization()(inputs)
    elif norm == 'gn':
        return GroupNormalization()(inputs)

def _activation(inputs, activation='relu'):
    if activation in ['relu', 'sigmoid', 'softmax']:
        return Activation(activation)(inputs)
    elif activation == 'leakyrelu':
        return LeakyReLU(alpha=.3)(inputs)

def _downsizing(inputs, filters, downsizing='pooling', norm='bn', activation='relu'):
    if downsizing == 'pooling':
        return MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(inputs)
    elif downsizing == 'conv':
        x = Conv3D(filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(inputs)
        x = _activation(x, activation=activation)
        x = _normalization(x, norm=norm)
        return x

def _se_block(inputs, filters, se_ratio=16):
    x = GlobalAveragePooling3D()(inputs)
    x = Dense(filters//se_ratio, activation='relu')(x)
    x = Dense(filters, activation='sigmoid')(x)
    x = Reshape([1, 1, 1, filters])(x)
    x = Multiply()([inputs, x])
    return x


def _basic_block(inputs, filters, norm='bn', activation='relu', is_seblock=False):
    x = Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same')(inputs)
    x = _normalization(x, norm=norm)
    x = _activation(x, activation=activation)
    if is_seblock:
        x = _se_block(x, filters)
    return x

def _resnet_conv_block(inputs, filters, strides=(2, 2, 2), norm='bn', activation='relu'):
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

def _resnet_identity_block(inputs, filters, is_seblock=False):
    x = Conv3D(filters[0], (1, 1, 1))(inputs)
    x = _normalization(x, norm=norm)
    x = _activation(x, activation=activation)

    x = Conv3D(filters[1], (3, 3, 3), padding='same')(x)
    x = _normalization(x, norm=norm)
    x = _activation(x, activation=activation)

    x = Conv3D(filters[2], (1, 1, 1))(x)
    x = _normalization(x, norm=norm)
    if is_seblock:
        x = _se_block(x, filters[2])

    x = Add()([x, inputs])
    x = _activation(x, activation=activation)
    
    return x

def _atten_gate(inputs, skip, filters):
        def __expend_as(tensor, rep):
            my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep})(tensor)
            return my_repeat

        gating = Conv3D(K.int_shape(inputs)[-1], (1, 1, 1), use_bias=False, padding='same')(inputs)
        gating = _norm(gating)
        shape_skip = K.int_shape(skip)
        shape_gating = K.int_shape(gating)

        #
        theta = Conv3D(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(skip)
        shape_theta = K.int_shape(theta)

        phi = Conv3D(filters, (1, 1, 1), use_bias=False, padding='same')(gating)
        phi = Conv3DTranspose(filters, (3, 3, 3), 
                              strides=(shape_theta[1]//shape_gating[1], 
                                       shape_theta[2]//shape_gating[2], 
                                       shape_theta[3]//shape_gating[3]),
                              padding='same')(phi)

        add_xg = Add()([phi, theta])
        act_xg = Activation(activation='relu')(add_xg)
        psi = Conv3D(1, (1, 1, 1), use_bias=False, padding='same')(act_xg)
        sigmoid_xg = Activation(activation='sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)

        upsample_psi = UpSampling3D(size=(shape_skip[1]//shape_sigmoid[1], 
                                          shape_skip[2]//shape_sigmoid[2],
                                          shape_skip[3]//shape_sigmoid[3]))(sigmoid_xg)
        upsample_psi = __expend_as(upsample_psi, shape_skip[4])

        result = Multiply()([skip, attention])
        result = Conv3D(shape_skip[3], (3, 3, 3), padding='same')(result)
        result = _norm(result)
        return result

def _crop_concat(mode='concat'):
    def crop(concat_layers):
        big, small = concat_layers[0], concat_layers[1:]
        big_shape, small_shape = tf.shape(big), tf.shape(small[0])
        sh, sw, sd = small_shape[1], small_shape[2], small_shape[3]
        bh, bw, bd = big_shape[1], big_shape[2] ,big_shape[3]
        dh, dw, dd = bh-sh, bw-sw, bd-sd
        big_crop = big[:,:-dh,:-dw,:-dd,:]
        return K.concatenate([big_crop] + small, axis=-1)
    return Lambda(crop)

def _unet_upconv_block(inputs, skip_input, filters, skip='unet', top_down='unet', norm='bn', activation='relu'):
    if 'attention' in skip:
        skip_input = [_atten_gate(inputs, skip_input, filters)]
    elif 'dense' in skip:
        raise ValueError()
    elif 'unet' in skip:
        skip_input = [skip_input]

    x = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(inputs)
    x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(x)
    x = _normalization(x, norm=norm)
    x = _activation(x, activation=activation)
    x = _crop_concat()([x]+skip_input)

    if 'unet' in top_down:
        x = _basic_block(x, filters, norm=norm, activation=activation, is_seblock=True if 'se' in top_down else False)
        x = _basic_block(x, filters, norm=norm, activation=activation, is_seblock=True if 'se' in top_down else False)

    return x