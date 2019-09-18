from . import Backbone
from .layers import *
from .segmentation import *

from keras.layers import Input
from keras.layers import GlobalAveragePooling3D
from keras.layers import Dense
from keras.layers import Activation


class UnetBackbone(Backbone):
    def unet(self, *args, **kwargs):
        return unet_structure(VGG3D(main_args=self.main_args,
                                    sub_args=self.sub_args, 
                                    base_filter=int(32//self.sub_args['hyperparameter']['divide'])),
                              sub_args=self.sub_args)

    def validate(self):
        allowed_backbones = ['vggA', 'vggB', 'vggC', 'vggD', 'vggE', 'vggAse', 'vggBse', 'vggCse', 'vggDse', 'vggEse']
        if self.sub_args['mode']['bottom_up'] not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.sub_args['mode']['bottom_up'], 
                                                                                       allowed_backbones))

def VGG3D(main_args, sub_args, base_filter=32):
    model_name = sub_args['mode']['bottom_up'] if main_args.mode == 'segmentation' else sub_args['mode']['model']

    img_input = Input(shape=sub_args['hyperparameter']['input_shape'])

    # Block 1
    c1 = _basic_block(img_input, base_filter, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggB', 'vggC', 'vggD', 'vggE', 'vggBse', 'vggCse', 'vggDse', 'vggEse']:
        c1 = _basic_block(c1, base_filter, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)

    # Block 2
    c2 = _basic_block(c1, base_filter*2, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    
    if model_name in ['vggB', 'vggC', 'vggD', 'vggE', 'vggBse', 'vggCse', 'vggDse', 'vggEse']:
        c2 = _basic_block(c2, base_filter*2, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
    
    # Block 3
    c3 = _basic_block(c2, base_filter*4, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    c3 = _basic_block(c3, base_filter*4, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggC', 'vggCse']:
        c3 = _basic_block(c3, base_filter*4, (1, 1, 1),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)

    elif model_name in ['vggD', 'vggE', 'vggDse', 'vggEse']:
        c3 = _basic_block(c3, base_filter*4, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
        if model_name in ['vggE', 'vggEse']:
            c3 = _basic_block(c3, base_filter*4, (3, 3, 3),
                              downsizing=sub_args['hyperparameter']['downsizing'],
                              norm=sub_args['hyperparameter']['norm'], 
                              activation=sub_args['hyperparameter']['activation'], 
                              is_seblock=True if 'se' in model_name else False,
                              is_downsizing=False)

    # Block 4
    c4 = _basic_block(c3, base_filter*8, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    c4 = _basic_block(c4, base_filter*8, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggC', 'vggCse']:
        c4 = _basic_block(c4, base_filter*8, (1, 1, 1),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)

    elif model_name in ['vggD', 'vggE', 'vggDse', 'vggEse']:
        c4 = _basic_block(c4, base_filter*8, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
        if model_name in ['vggE', 'vggEse']:
            c4 = _basic_block(c4, base_filter*8, (3, 3, 3),
                              downsizing=sub_args['hyperparameter']['downsizing'],
                              norm=sub_args['hyperparameter']['norm'], 
                              activation=sub_args['hyperparameter']['activation'], 
                              is_seblock=True if 'se' in model_name else False,
                              is_downsizing=False)

    # Block 5
    c5 = _basic_block(c4, base_filter*16, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    c5 = _basic_block(c5, base_filter*16, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggC', 'vggCse']:
        c5 = _basic_block(c5, base_filter*16, (1, 1, 1),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)

    elif model_name in ['vggD', 'vggE', 'vggDse', 'vggEse']:
        c5 = _basic_block(c5, base_filter*16, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
        if model_name in ['vggE', 'vggEse']:
            c5 = _basic_block(c5, base_filter*16, (3, 3, 3),
                              downsizing=sub_args['hyperparameter']['downsizing'],
                              norm=sub_args['hyperparameter']['norm'], 
                              activation=sub_args['hyperparameter']['activation'], 
                              is_seblock=True if 'se' in model_name else False,
                              is_downsizing=False)

    if main_args.mode == 'segmentation':
        if sub_args['mode']['depth'] == 4:
            return img_input, [c1, c2, c3, c4, c5], base_filter

        elif sub_args['mode']['depth'] == 3:
            return img_input, [c1, c2, c3, c4], base_filter

    elif main_args.mode == 'classification':
        flatten = GlobalAveragePooling3D()(c5)
        dense1 = Dense(int(base_filter*2.625))(flatten)
        dense2 = Dense(int(base_filter*2.625))(dense1)
        dense3 = Dense(sub_args['hyperparameter']['classes'])(dense2)
        img_output = Activation(activation='softmax')(dense3)
        
        model = Model(img_input, img_output, name=model_name)
        return model