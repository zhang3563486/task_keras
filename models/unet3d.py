from . import Backbone
from .layers import *
from .segmentation import *
from keras.layers import Input


class UnetBackbone(Backbone):
    def unet(self, *args, **kwargs):
        return unet_structure(Unet(task=self.task, 
                                   sub_args=self.sub_args, 
                                   base_filter=int(32//self.sub_args['mode']['divide'])),
                              sub_args=self.sub_args)

    def validate(self):
        allowed_backbones = ['unet', 'unetse']
        if self.sub_args['mode']['bottom_up'] not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.sub_args['mode']['bottom_up'], 
                                                                                       allowed_backbones))

def Unet(task, sub_args, base_filter=32):
    img_input = Input(shape=sub_args['hyperparameter']['input_shape'])

    c1 = _basic_block(img_input, base_filter, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)
    c1 = _basic_block(c1, base_filter, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)

    c2 = _basic_block(c1, base_filter*2, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)
    c2 = _basic_block(c2, base_filter*2, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)
    
    c3 = _basic_block(c2, base_filter*4, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)
    c3 = _basic_block(c3, base_filter*4, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)

    c4 = _basic_block(c3, base_filter*8, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)
    c4 = _basic_block(c4, base_filter*8, norm=sub_args['mode']['norm'], 
                      activation=sub_args['mode']['activation'], 
                      is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)

    if sub_args['mode']['depth'] == 4:
        c5 = _basic_block(c4, base_filter*16, norm=sub_args['mode']['norm'], 
                          activation=sub_args['mode']['activation'], 
                          is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)
        c5 = _basic_block(c5, base_filter*16, norm=sub_args['mode']['norm'], 
                          activation=sub_args['mode']['activation'], 
                          is_seblock=True if 'se' in sub_args['mode']['bottom_up'] else False)
        return img_input, [c1, c2, c3, c4, c5], base_filter

    elif sub_args['mode']['depth'] == 3:
        return img_input, [c1, c2, c3, c4], base_filter

    else:
        raise Exception('Depth size must be 3 or 4. You put ', sub_args['mode']['depth'])