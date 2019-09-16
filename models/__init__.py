from __future__ import print_function

import keras
from .. import losses
from .. import metrics

class Backbone(object):
    def __init__(self, task, sub_args):
        self.custom_objects = {
            'dice_loss'     : losses.dice_loss,
            'dice_loss_wo'  : losses.dice_loss_wo,
            'crossentropy'  : losses.crossentropy,
            'focal'         : losses.focal()
        }

        self.task = task
        self.sub_args = sub_args

    def unet(self, *args, **kwargs):
        raise NotImplementedError('U-Net method is not implemented.')

    def fpn(self, *args, **kwargs):
        raise NotImplementedError('FPN method is not implemented.')

    def validate(self):
        raise NotImplementedError('validate method is not implemented.')

def backbone(task, sub_args):
    if 'unet' in sub_args['mode']['bottom_up']:
        from .unet3d import UnetBackbone as b

    elif 'vgg' in sub_args['mode']['bottom_up']:
        from .vgg3d import VGGBackbone as b

    elif 'resnet' in sub_args['mode']['bottom_up']:
        from .resnet3d import ResNetBackbone as b
    
    return b(task, sub_args)

def compile(sub_args):
    if sub_args['hyperparameter']['optimizer'] == 'adam':
        optimizer = keras.optimizers.adam(lr=sub_args['hyperparameter']['lr'], clipnorm=.001)
    
    if sub_args['hyperparameter']['lossfn'] == 'dice':
        loss = losses.dice_loss

    metric = [metrics.dice]
    metric.append(metrics.dice_target)
    if sub_args['hyperparameter']['classes'] == 3:
        if sub_args['task']['subtask'] in ['Vessel']:
            metric.append(metrics.dice_cancer)

    return optimizer, loss, metric