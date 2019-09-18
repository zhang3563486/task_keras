import os
import yaml
import argparse

import keras
import tensorflow as tf
from collections import OrderedDict

from .. import callbacks
from ..preprocessing import cdss_segmentation
from ..preprocessing import cdss_classification

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def check_yaml(main_args):
    sub_args = yaml.load(open(main_args.yaml))
    sub_args['task'] = sub_args['task'][main_args.task]
    sub_args['mode'] = sub_args['mode'][main_args.mode]
    
    # Sanity check for task
    if main_args.task == 'CDSS_Liver':
        assert sub_args['task']['subtask'], 'Subtask of maintask must be selected.'
        assert sub_args['task']['phase'], 'Phase of maintask must be selected.'
    
    # Sanity check for mode
    # Must set # of SE-block
    if main_args.mode == 'segmentation':
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


def create_callbacks(main_args, sub_args, generator, eval_steps):
    # set_cbdir(main_args, sub_args)
    # cp = callbacks.callback_checkpoint(filepath=os.path.join(sub_args['etc']['checkpoint_root'], 
    #                                                          main_args.task, main_args.mode, 
    #                                                          sub_args['task']['subtask'], 
    #                                                          sub_args['hyperparameter']['stamp'],
    #                                                          'checkpoint',
    #                                                          '{epoch:04d}_{val_dice:.4f}.h5' if main_args.mode == 'segmentation' \
    #                                                          else '{epoch:04d}_{val_loss:.4f}.h5'),
    #                                    monitor='val_dice' if main_args.mode == 'segmentation' else 'val_loss',
    #                                    verbose=1,
    #                                    mode='max' if main_args.mode == 'segmentation' else 'min',
    #                                    save_best_only=False,
    #                                    save_weights_only=False)

    # el = callbacks.callback_epochlogger(filename=os.path.join(sub_args['etc']['checkpoint_root'], 
    #                                                           main_args.task, main_args.mode, 
    #                                                           sub_args['task']['subtask'], 
    #                                                           sub_args['hyperparameter']['stamp'],
    #                                                           'history', 'epoch.csv'),
    #                                     separator=',', append=True)

    # bl = callbacks.callback_batchlogger(filename=os.path.join(sub_args['etc']['checkpoint_root'], 
    #                                                           main_args.task, main_args.mode, 
    #                                                           sub_args['task']['subtask'], 
    #                                                           sub_args['hyperparameter']['stamp'],
    #                                                           'history', 'batch.csv'),
    #                                     separator=',', append=True)

    # tb = callbacks.callback_tensorboard(log_dir=os.path.join(sub_args['etc']['checkpoint_root'], 
    #                                                          main_args.task, main_args.mode, 
    #                                                          sub_args['task']['subtask'], 
    #                                                          sub_args['hyperparameter']['stamp'],
    #                                                          'logs'), 
    #                                     batch_size=1)
    
    # ls = callbacks.callback_learningrate(initlr=sub_args['hyperparameter']['lr'],
    #                                      mode=sub_args['hyperparameter']['lr_mode'], 
    #                                      value=sub_args['hyperparameter']['lr_value'], 
    #                                      duration=sub_args['hyperparameter']['lr_duration'], 
    #                                      total_epoch=sub_args['hyperparameter']['epochs'])

    # callback_list = [cp, el, bl, tb, ls]
    callback_list = []
    if main_args.mode == 'classification':
        ev = callbacks.callback_evaluate(generator=generator, eval_steps=eval_steps)
        callback_list.append(ev)
    
    return callback_list


def create_generator(main_args, sub_args, datalist, mode, rotation_range):
    if main_args.task == 'CDSS_Liver':
        generator_dict = {
            # segmentation
            'multi_organ'     : cdss_segmentation.Generator_multiorgan,
            'Vessel'          : cdss_segmentation.Generator_Vessel,

            # classification
            'ascites'         : cdss_classification.Generator,
            'varix'           : cdss_classification.Generator,
            'Pvi_loca'        : cdss_classification.Generator,
            'distrutubion'    : cdss_classification.Generator,
            'RFA_feasiblity'  : cdss_classification.Generator
        }

        generator = generator_dict[sub_args['task']['subtask']](
            sub_args=sub_args,
            datalist=datalist,
            mode=mode,
            rotation_range=rotation_range,
            shuffle=True if mode == 'train' else False
        )

    return generator