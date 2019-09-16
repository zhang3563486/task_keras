from .layers import *
from keras.layers import Conv3D
from keras.layers import Activation
from keras.models import Model

def unet_structure(backbone_output, sub_args):
    if len(backbone_output[1]) == 5:
        img_input, [c1, c2, c3, c4, c5], base_filter = backbone_output
        u4 = _unet_upconv_block(c5, c4, base_filter*8, 
                                skip=sub_args['mode']['skip'],
                                top_down=sub_args['mode']['top_down'],
                                norm=sub_args['mode']['norm'],
                                activation=sub_args['mode']['activation'])

        u3 = _unet_upconv_block(u4, c3, base_filter*4, 
                                skip=sub_args['mode']['skip'],
                                top_down=sub_args['mode']['top_down'],
                                norm=sub_args['mode']['norm'],
                                activation=sub_args['mode']['activation'])

    elif len(backbone_output[1]) == 4:
        img_input, [c1, c2, c3, c4], base_filter = backbone_output
        u3 = _unet_upconv_block(c4, c3, base_filter*4, 
                                skip=sub_args['mode']['skip'],
                                top_down=sub_args['mode']['top_down'],
                                norm=sub_args['mode']['norm'],
                                activation=sub_args['mode']['activation'])
        
    u2 = _unet_upconv_block(u3, c2, base_filter*2, 
                            skip=sub_args['mode']['skip'],
                            top_down=sub_args['mode']['top_down'],
                            norm=sub_args['mode']['norm'],
                            activation=sub_args['mode']['activation'])

    u1 = _unet_upconv_block(u2, c1, base_filter, 
                            skip=sub_args['mode']['skip'],
                            top_down=sub_args['mode']['top_down'],
                            norm=sub_args['mode']['norm'],
                            activation=sub_args['mode']['activation'])

    img_output = Conv3D(sub_args['hyperparameter']['classes'], (1, 1, 1), strides=(1, 1, 1), padding='same')(u1)
    img_output = Activation(activation='softmax')(img_output)

    model = Model(img_input, img_output, name='{}-{}-{}'.format(sub_args['mode']['bottom_up'],
                                                                sub_args['mode']['top_down'],
                                                                sub_args['mode']['skip']))

    return model