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
        allowed_backbones = ['vggA', 'vggB', 'vggC', 'vggD', 'vggE', 
                             'vggAse', 'vggBse', 'vggCse', 'vggDse', 'vggEse']

        if self.sub_args['mode']['bottom_up'] not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.sub_args['mode']['bottom_up'], 
                                                                                       allowed_backbones))

def VGG3D(main_args, sub_args, base_filter=32):
    model_name = sub_args['mode']['bottom_up'] if main_args.mode == 'segmentation' else sub_args['mode']['model']

    img_input = Input(shape=sub_args['hyperparameter']['input_shape'])
    img_output = []

    # Block 1
    x = _basic_block(img_input, base_filter, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggB', 'vggC', 'vggD', 'vggE', 'vggBse', 'vggCse', 'vggDse', 'vggEse']:
        x = _basic_block(x, base_filter, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
    
    if main_args.mode == 'segmentation':
        img_output.append(x)

    # Block 2
    x = _basic_block(x, base_filter*2, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    
    if model_name in ['vggB', 'vggC', 'vggD', 'vggE', 'vggBse', 'vggCse', 'vggDse', 'vggEse']:
        x = _basic_block(x, base_filter*2, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
    
    if main_args.mode == 'segmentation':
        img_output.append(x)

    # Block 3
    x = _basic_block(x, base_filter*4, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    x = _basic_block(x, base_filter*4, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggC', 'vggCse']:
        x = _basic_block(x, base_filter*4, (1, 1, 1),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)

    elif model_name in ['vggD', 'vggE', 'vggDse', 'vggEse']:
        x = _basic_block(x, base_filter*4, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
        if model_name in ['vggE', 'vggEse']:
            x = _basic_block(x, base_filter*4, (3, 3, 3),
                              downsizing=sub_args['hyperparameter']['downsizing'],
                              norm=sub_args['hyperparameter']['norm'], 
                              activation=sub_args['hyperparameter']['activation'], 
                              is_seblock=True if 'se' in model_name else False,
                              is_downsizing=False)

    if main_args.mode == 'segmentation':
        img_output.append(x)
        
    # Block 4
    x = _basic_block(x, base_filter*8, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    x = _basic_block(x, base_filter*8, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggC', 'vggCse']:
        x = _basic_block(x, base_filter*8, (1, 1, 1),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)

    elif model_name in ['vggD', 'vggE', 'vggDse', 'vggEse']:
        x = _basic_block(x, base_filter*8, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
        if model_name in ['vggE', 'vggEse']:
            x = _basic_block(x, base_filter*8, (3, 3, 3),
                              downsizing=sub_args['hyperparameter']['downsizing'],
                              norm=sub_args['hyperparameter']['norm'], 
                              activation=sub_args['hyperparameter']['activation'], 
                              is_seblock=True if 'se' in model_name else False,
                              is_downsizing=False)

    if main_args.mode == 'segmentation':
        img_output.append(x)
    
    # Block 5
    x = _basic_block(x, base_filter*16, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=True)
    x = _basic_block(x, base_filter*16, (3, 3, 3),
                      downsizing=sub_args['hyperparameter']['downsizing'],
                      norm=sub_args['hyperparameter']['norm'], 
                      activation=sub_args['hyperparameter']['activation'], 
                      is_seblock=True if 'se' in model_name else False,
                      is_downsizing=False)
    
    if model_name in ['vggC', 'vggCse']:
        x = _basic_block(x, base_filter*16, (1, 1, 1),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)

    elif model_name in ['vggD', 'vggE', 'vggDse', 'vggEse']:
        x = _basic_block(x, base_filter*16, (3, 3, 3),
                          downsizing=sub_args['hyperparameter']['downsizing'],
                          norm=sub_args['hyperparameter']['norm'], 
                          activation=sub_args['hyperparameter']['activation'], 
                          is_seblock=True if 'se' in model_name else False,
                          is_downsizing=False)
        if model_name in ['vggE', 'vggEse']:
            x = _basic_block(x, base_filter*16, (3, 3, 3),
                              downsizing=sub_args['hyperparameter']['downsizing'],
                              norm=sub_args['hyperparameter']['norm'], 
                              activation=sub_args['hyperparameter']['activation'], 
                              is_seblock=True if 'se' in model_name else False,
                              is_downsizing=False)

    if main_args.mode == 'segmentation' and sub_args['mode']['depth'] == 4:
        img_output.append(x)

    if main_args.mode == 'segmentation':
        return img_input, img_output, base_filter

    elif main_args.mode == 'classification':
        flatten = GlobalAveragePooling3D()(x)
        dense1 = Dense(int(base_filter*2.625))(flatten)
        dense2 = Dense(int(base_filter*2.625))(dense1)
        dense3 = Dense(sub_args['hyperparameter']['classes'])(dense2)
        img_output = Activation(activation='softmax')(dense3)
        
        model = Model(img_input, img_output, name=model_name)
        return model