from __future__ import print_function

import keras
from . import classification
from .. import losses
from .. import metrics

class Backbone(object):
    def __init__(self, main_args, sub_args):
        self.custom_objects = {
            'dice_loss'     : losses.dice_loss,
            'dice_loss_wo'  : losses.dice_loss_wo,
            'crossentropy'  : losses.crossentropy,
            'focal'         : losses.focal()
        }

        self.main_args = main_args
        self.sub_args = sub_args

    def unet(self, *args, **kwargs):
        raise NotImplementedError('U-Net method is not implemented.')

    def fpn(self, *args, **kwargs):
        raise NotImplementedError('FPN method is not implemented.')

    def validate(self):
        raise NotImplementedError('validate method is not implemented.')

def backbone(main_args, sub_args):
    if 'unet' in sub_args['mode']['bottom_up']:
        from .unet3d import UnetBackbone as b

    elif 'vgg' in sub_args['mode']['bottom_up']:
        from .vgg3d import VGGBackbone as b

    elif 'resnet' in sub_args['mode']['bottom_up']:
        from .resnet3d import ResNetBackbone as b
    
    return b(main_args, sub_args)

def compile(main_args, sub_args):
    if sub_args['hyperparameter']['optimizer'] == 'adam':
        optimizer = keras.optimizers.adam(lr=sub_args['hyperparameter']['lr'], clipnorm=.001)
    elif sub_args['hyperparameter']['optimizer'] == 'sgd':
        optimizer = keras.optimizers.sgd(lr=sub_args['hyperparameter']['lr'], clipnorm=.001)
    
    if main_args.mode == 'segmentation':
        if sub_args['hyperparameter']['lossfn'] == 'dice':
            loss = losses.dice_loss
        elif sub_args['hyperparameter']['lossfn'] == 'crossentropy':
            loss = losses.crossentropy
        elif sub_args['hyperparameter']['lossfn'] == 'focal':
            loss = losses.focal
        elif sub_args['hyperparameter']['lossfn'] == 'cedice':
            loss = losses.ce_dice_loss
        elif sub_args['hyperparameter']['lossfn'] == 'focaldice':
            loss = losses.focal_dice_loss
        elif sub_args['hyperparameter']['lossfn'] == 'celogdice':
            loss = losses.ce_logdice_loss
        elif sub_args['hyperparameter']['lossfn'] == 'focallogdice':
            loss = losses.focal_logdice_loss
        else:
            raise ValueError()
    
    elif main_args.mode == 'classification':
        if sub_args['hyperparameter']['lossfn'] == 'crossentropy':
            loss = losses.crossentropy
        elif sub_args['hyperparameter']['lossfn'] == 'focal':
            loss = losses.focal
        else:
            raise ValueError()

    if main_args.mode == 'segmentation':
        metric = [metrics.dice]
        metric.append(metrics.dice_target)
        if sub_args['hyperparameter']['classes'] == 3:
            if sub_args['task']['subtask'] in ['Vessel']:
                metric.append(metrics.dice_cancer)
    
    elif main_args.mode == 'classification':
        metric = ['acc']

    return optimizer, loss, metric