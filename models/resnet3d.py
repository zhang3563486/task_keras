from . import Backbone
from .layers import *
from .segmentation import *

from keras.layers import Input
from keras.layers import ZeroPadding3D
from keras.layers import MaxPooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import Dense
from keras.layers import Activation


class ResNetBackbone(Backbone):
    def unet(self, *args, **kwargs):
        return unet_structure(ResNet3D(main_args=self.main_args, 
                                       sub_args=self.sub_args, 
                                       base_filter=int(32//self.sub_args['hyperparameter']['divide'])),
                              sub_args=self.sub_args)

    def validate(self):
        allowed_backbones = ['resnet18', 'resnet18se', 'resnet34', 'resnet34se',
                             'resnet50', 'resnet50se', 'resnet101', 'resnet101se',
                             'resnet152', 'resnet152se']

        if self.sub_args['mode']['bottom_up'] not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.sub_args['mode']['bottom_up'], 
                                                                                       allowed_backbones))


def ResNet3D(main_args, sub_args, base_filter=32):
    model_name = sub_args['mode']['bottom_up'] if main_args.mode == 'segmentation' else sub_args['mode']['model']
    loop = {18: [2, 2, 2, 2], 
            34: [3, 4, 6, 3], 
            50: [3, 4, 6, 3], 
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]}
            
    if 'se' in model_name:
        is_seblock = True
        model_name = model_name[:-2]
    else:
        is_seblock = False


    img_input = Input(shape=sub_args['hyperparameter']['input_shape'])
    img_output = []

    x = ZeroPadding3D(padding=(3, 3, 3))(img_input)
    x = Conv3D(base_filter, (7, 7, 7), strides=(2, 2, 2), padding='valid')(x)
    x = _normalization(x, norm=sub_args['hyperparameter']['norm'])
    x = _activation(x, activation=sub_args['hyperparameter']['activation'])
    if main_args.mode == 'segmentation':
        img_output.append(x)

    x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)

    if int(model_name[3:]) in [50, 101, 152]:
        for i, l in enumerate(loop[int(model_name[3:])]):
            for j in range(l):
                if j == 0:
                    x = _resnet_bottleneck_block(x, [base_filter*(2**i), base_filter*(2**i), base_filter*(2**(i+2))], 
                                                 strides=(2, 2, 2),
                                                 norm=sub_args['hyperparameter']['norm'],
                                                 activation=sub_args['hyperparameter']['activation'],
                                                 is_seblock=is_seblock)
                else:
                    x = _resnet_bottleneck_block(x, [base_filter*(2**i), base_filter*(2**i), base_filter*(2**(i+2))],
                                                 norm=sub_args['hyperparameter']['norm'],
                                                 activation=sub_args['hyperparameter']['activation'],
                                                 is_identity=True,
                                                 is_seblock=is_seblock)
            
            if main_args.mode == 'segmentation':
                img_output.append(x)

    elif int(model_name[3:]) in [18, 34]:
        for i, l in enumerate(loop[int(model_name[3:])]):
            for j in range(l):
                if j == 0:
                    x = _resnet_conv_block(x, [base_filter*(2**i), base_filter*(2**i)], 
                                           strides=(2, 2, 2),
                                           norm=sub_args['hyperparameter']['norm'],
                                           activation=sub_args['hyperparameter']['activation'],
                                           is_seblock=is_seblock)
                else:
                    x = _resnet_conv_block(x, [base_filter*(2**i), base_filter*(2**i)],
                                           norm=sub_args['hyperparameter']['norm'],
                                           activation=sub_args['hyperparameter']['activation'],
                                           is_identity=True,
                                           is_seblock=is_seblock)
            
            if main_args.mode == 'segmentation':
                img_output.append(x)

    else:
        raise ValueError('You must enter the model name following guideline.')
    
    if main_args.mode == 'segmentation':
        model = Model(img_input, img_output, name='resnet')

    elif main_args.mode == 'classification':
        x = GlobalAveragePooling3D()(x)
        img_output = Dense(sub_args['hyperparameter']['classes'])(x)
        img_output = Activation(activation='softmax')(img_output)

        model = Model(img_input, img_output, name='resnet')
        return model