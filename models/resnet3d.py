import keras
from keras.layers import GlobalAveragePooling3D
from keras.layers import ZeroPadding3D
from keras.layers import MaxPooling3D
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import Add

from keras.models import Model

from . import Backbone
from .layers import *
from .segmentation import *


class ResNetBackbone(Backbone):
    def unet(self, *args, **kwargs):
        return unet_structure(ResNet3D(task=self.task, 
                                       sub_args=self.sub_args, 
                                       base_filter=int(32//self.sub_args['mode']['divide'])))

    def validate(self):
        allowed_backbones = ['resnet50']

        if self.sub_args['mode']['bottom_up'] not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.sub_args['mode']['bottom_up'], 
                                                                                       allowed_backbones))


def ResNet3D(task, sub_args, base_filter=32):
    model_name = sub_args['mode']['bottom_up'] if main_args.mode == 'segmentation' else sub_args['mode']['model']
    img_input = Input(shape=sub_args['hyperparameter']['input_shape'])

    c1 = ZeroPadding3D(padding=(3, 3, 3))(img_input)
    c1 = Conv3D(base_filter, (7, 7, 7), strides=(2, 2, 2), padding='valid')(c1)
    c1 = _normalization(c1, norm=sub_args['mode']['norm'])
    c1 = _activation(c1, activation=sub_args['mode']['activation'])

    c2 = ZeroPadding3D(padding=(1, 1, 1))(c1)
    c2 = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(c2)
    c2 = _resnet_conv_block(c2, [base_filter, base_filter, base_filter*4], strides=(1, 1, 1))
    c2 = _resnet_identity_block(c2, [base_filter, base_filter, base_filter*4])
    c2 = _resnet_identity_block(c2, [base_filter, base_filter, base_filter*4])

    c3 = _resnet_conv_block(c2, [base_filter*2, base_filter*2, base_filter*8])
    c3 = _resnet_identity_block(c3, [base_filter*2, base_filter*2, base_filter*8])
    c3 = _resnet_identity_block(c3, [base_filter*2, base_filter*2, base_filter*8])
    c3 = _resnet_identity_block(c3, [base_filter*2, base_filter*2, base_filter*8])

    c4 = _resnet_conv_block(c3, [base_filter*4, base_filter*4, base_filter*16])
    c4 = _resnet_identity_block(c4, [base_filter*4, base_filter*4, base_filter*16])
    c4 = _resnet_identity_block(c4, [base_filter*4, base_filter*4, base_filter*16])
    c4 = _resnet_identity_block(c4, [base_filter*4, base_filter*4, base_filter*16])
    c4 = _resnet_identity_block(c4, [base_filter*4, base_filter*4, base_filter*16])
    c4 = _resnet_identity_block(c4, [base_filter*4, base_filter*4, base_filter*16])

    c5 = _resnet_conv_block(c4, [base_filter*8, base_filter*8, base_filter*32])
    c5 = _resnet_identity_block(c5, [base_filter*8, base_filter*8, base_filter*32])
    c5 = _resnet_identity_block(c5, [base_filter*8, base_filter*8, base_filter*32])
    
    if task == 'segmentation':
        model = Model([img_input], [c1, c2, c3, c4, c5], name='resnet')

    elif task == 'classification':
        dense = GlobalAveragePooling3D()(c5)
        logits = Dense(sub_args['hyperparameter']['classes'], activation='softmax')(dense)
        model = Model([img_input], [logits], name='resnet')

        return model