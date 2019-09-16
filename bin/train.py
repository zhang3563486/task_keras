# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import json
import argparse
import numpy as np
from collections import OrderedDict

import keras
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

SEED = 42
np.random.seed(SEED)

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import src_sungchul_keras.bin  # noqa: F401
    __package__ = "src_sungchul_keras.bin"

from . import get_session
from .. import models
from .. import callbacks
from .. import preprocessing

def set_cbdir(main_args, sub_args):
    if not os.path.isdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task)):
        os.mkdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task))

    if not os.path.isdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode)):
        os.mkdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode))

    if not os.path.isdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode, sub_args['task']['subtask'])):
        os.mkdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode, sub_args['task']['subtask']))

    if not os.path.isdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode, sub_args['task']['subtask'], sub_args['hyperparameter']['stamp'])):
        os.mkdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode, sub_args['task']['subtask'], sub_args['hyperparameter']['stamp']))
    
    for i in ['checkpoint', 'history', 'logs']:
        if not os.path.isdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode, sub_args['task']['subtask'], sub_args['hyperparameter']['stamp'], i)):
            os.mkdir(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode, sub_args['task']['subtask'], sub_args['hyperparameter']['stamp'], i))
    
    new_args = OrderedDict()
    new_args['task'] = {main_args.task: sub_args['task']}
    new_args['mode'] = {main_args.mode: sub_args['mode']}
    new_args['hyperparameter'] = sub_args['hyperparameter']
    new_args['etc'] = sub_args['etc']
    with open(os.path.join(sub_args['etc']['checkpoint_root'], main_args.task, main_args.mode, sub_args['task']['subtask'], sub_args['hyperparameter']['stamp'], 'model_desc.yml'), 'w') as f:
        yaml.dump(dict(new_args), f, default_flow_style=False)

def check_yaml(main_args):
    sub_args = yaml.safe_load(open(main_args.yaml))
    sub_args['task'] = sub_args['task'][main_args.task]
    sub_args['mode'] = sub_args['mode'][main_args.mode]
    
    # Sanity check for task
    if main_args.task == 'CDSS_Liver':
        assert sub_args['task']['subtask'], 'Subtask of maintask must be selected.'
        assert sub_args['task']['phase'], 'Phase of maintask must be selected.'
    
    # Sanity check for mode
    # Must set # of SE-block
    if 'se' in sub_args['mode']['bottom_up'] or 'se' in sub_args['mode']['top_down']:
        assert sub_args['mode']['num_se'] > 0, 'If selecting SE-block, choose # of SE-block over 0.'
        assert sub_args['mode']['conv_loop'] >= sub_args['mode']['num_se'], '# of SE-block cannot be more than conv loop.'

    # Cannot select unet and attention in skip connection
    if 'unet' in sub_args['mode']['skip'] and 'attention' in sub_args['mode']['skip']:
        raise ValueError('You must choose ONLY one skip connection.')

    # Sanity check for etc
    assert sub_args['etc']['checkpoint_root'], 'Root path of checkpoint must be selected.'
    assert sub_args['etc']['result_root'], 'Root path of result must be selected.'
    assert sub_args['etc']['data_root'], 'Root path of data must be selected.'

    return sub_args

def check_args(parsed_args):
    if not parsed_args.yaml:
        raise ValueError('You must enter the yaml file.')

    if not parsed_args.task:
        raise ValueError('You must enter the task to do')

    if not parsed_args.mode:
        raise ValueError('You must enter the mode.')

    return parsed_args
    

def get_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)

    return check_args(parser.parse_args(args))


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    main_args = get_arguments(args)
    sub_args = check_yaml(main_args)

    get_session()
    trainset, valset, testset = preprocessing.load(main_args, sub_args)

    print('TOTAL STEPS OF DATASET FOR TRAINING')
    if sub_args['task']['subtask'] == 'Vessel' and sub_args['hyperparameter']['patch']:
        if sub_args['hyperparameter']['patch'] == 'small':
            sub_args['hyperparameter']['input_shape'] = (32, 32, 32, 1)
            sub_args['hyperparameter']['stride'] = [8, 8, 8]
        elif sub_args['hyperparameter']['patch'] == 'medium':
            sub_args['hyperparameter']['input_shape'] = (32, 64, 64, 1)
            sub_args['hyperparameter']['stride'] = [8, 16, 16]
        elif sub_args['hyperparameter']['patch'] == 'large':
            sub_args['hyperparameter']['input_shape'] = (32, 128, 128, 1)
            sub_args['hyperparameter']['stride'] = [8, 32, 32]
        else:
            raise ValueError('If you choose \'Vessel task\', you must select a patch type.')

        print('========== trainset ==========')
        steps_per_epoch = sub_args['hyperparameter']['steps'] if sub_args['hyperparameter']['steps'] \
                          else calc_vessel_dataset(trainset, 
                                                   sub_args['task']['subtask'], 
                                                   sub_args['etc']['data_root'],
                                                   sub_args['etc']['result_root'],
                                                   sub_args['hyperparameter']['input_shape'], 
                                                   sub_args['hyperparameter']['stride'])
        print('    -->', steps_per_epoch)
        print('========== valset ==========')
        validation_steps = calc_vessel_dataset(valset, 
                                               sub_args['task']['subtask'], 
                                               sub_args['etc']['data_root'],
                                               sub_args['etc']['result_root'],
                                               sub_args['hyperparameter']['input_shape'], 
                                               sub_args['hyperparameter']['stride'])
        print('    -->', validation_steps)
    
    else:
        sub_args['hyperparameter']['input_shape'] = (None, None, None, 1)
        sub_args['hyperparameter']['stride'] = 1
        print('========== trainset ==========')
        steps_per_epoch = sub_args['hyperparameter']['steps'] if sub_args['hyperparameter']['steps'] else len(trainset)
        print('    -->', steps_per_epoch)
        print('========== valset ==========')
        validation_steps = len(valset)
        print('    -->', validation_steps)

    ##############################################
    # Set Model
    ##############################################
    if main_args.mode == 'segmentation':
        backbone = models.backbone(main_args.task, sub_args)
        model = backbone.unet()
    elif main_args.mode == 'classification':
        pass
    else:
        raise ValueError()

    if sub_args['etc']['summary']:
        from keras.utils import plot_model
        plot_model(model, to_file=os.path.join(sub_args['etc']['result_root'], 'model.png'), show_shapes=True)
        model.summary()
        return

    if sub_args['etc']['checkpoint']:
        model.load_weights(sub_args['etc']['checkpoint'])
        print("Load weights successfully at {}".format(sub_args['etc']['checkpoint']))
        sub_args['hyperparameter']['initial_epoch'] = int(sub_args['etc']['checkpoint'].split('/')[-1].split('_')[-2])
        sub_args['hyperparameter']['stamp'] = sub_args['etc']['checkpoint'].split('/')[5]
    else:
        sub_args['hyperparameter']['initial_epoch'] = 0
        sub_args['hyperparameter']['stamp'] = time.strftime("%c").replace(":", "_").replace(" ", "_")
    
    print()
    for k, v in sub_args.items():
        if k in ['task', 'mode']:
            print('{} :'.format(k), vars(main_args)[k])
        else:
            print(k)

        for kk, vv in v.items():
            print('    {} :'.format(kk), vv)
        print()

    optimizer, loss, metric = models.compile(sub_args)
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metric=metric)

    ##############################################
    # Set Callbacks
    ##############################################
    if sub_args['etc']['callback']:
        set_cbdir(main_args, sub_args)

        cp = callback_checkpoint(filepath=os.path.join('/mnas/CDSS_Liver/{}/{}/{}/checkpoint'.format(args.mode, args.task, stamp), '{epoch:04d}_{val_dice:.4f}.h5'),
                                monitor='val_dice',
                                verbose=1,
                                mode='max',
                                save_best_only=True,
                                save_weights_only=False)

        el = callback_epochlogger(filename=os.path.join('/mnas/CDSS_Liver/{}/{}/{}/history'.format(args.mode, args.task, stamp), 'epoch.csv'),
                                separator=',', append=True)

        bl = callback_batchlogger(filename=os.path.join('/mnas/CDSS_Liver/{}/{}/{}/history'.format(args.mode, args.task, stamp), 'batch.csv'),
                                separator=',', append=True)

        tb = callback_tensorboard(log_dir=os.path.join('/mnas/CDSS_Liver/{}/{}/{}/logs'.format(args.mode, args.task, stamp)), batch_size=1)
        
        ls = callback_learningrate(args.lr)

        callbacks = [cp, el, bl, tb, ls]
    
    else:
        callbacks = []

    ##############################################
    # Set Generator
    ##############################################
    train_generator = generator_dict[args.task](
        args=args,
        datalist=trainset, 
        mode='training',
        rotation_range=[10., 10., 10.]
    )

    val_generator = generator_dict[args.task](
        args=args,
        datalist=valset,
        mode='validation', 
        rotation_range=[0., 0., 0.], 
        shuffle=False
    )

    model.fit_generator(generator=train_generator,
                                steps_per_epoch=steps_per_epoch//args.batch_size,
                                verbose=1,
                                epochs=args.epochs,
                                validation_data=val_generator,
                                validation_steps=validation_steps//args.batch_size,
                                callbacks=callbacks,
                                shuffle=True,
                                initial_epoch=args.initial_epoch)


if __name__ == "__main__":
    main()