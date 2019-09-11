# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import json
import argparse
import numpy as np

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

from .. import models
from .. import callbacks
from .. import preprocessing

def set_cbdir(args, stamp):
    if not os.path.isdir('/mnas/{}'.format(args.task)):
        os.mkdir('/mnas/{}'.format(args.task))

    if not os.path.isdir('/mnas/{}/{}'.format(args.task, args.mode)):
        os.mkdir('/mnas/{}/{}'.format(args.task, args.mode))

    if not os.path.isdir('/mnas/{}/{}/{}'.format(args.task, args.mode, args.subtask)):
        os.mkdir('/mnas/{}/{}/{}'.format(args.task, args.mode, args.subtask))

    if not os.path.isdir('/mnas/{}/{}/{}/{}'.format(args.task, args.mode, args.subtask, stamp)):
        os.mkdir('/mnas/{}/{}/{}/{}'.format(args.task, args.mode, args.subtask, stamp))
    
    for i in ['checkpoint', 'history', 'logs']:
        if not os.path.isdir('/mnas/{}/{}/{}/{}/{}'.format(args.task, args.mode, args.subtask, stamp, i)):
            os.mkdir('/mnas/{}/{}/{}/{}/{}'.format(args.task, args.mode, args.subtask, stamp, i))
    
    with open('/mnas/{}/{}/{}/{}/model_desc.json'.format(args.task, args.mode, args.subtask, stamp), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

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
    return

    keras.backend.tensorflow_backend.set_session(get_session())

    trainset, testset = load(args.task)
    valset = trainset[:len(testset)]
    trainset = trainset[len(testset):]
    print('mode :', args.mode)
    print('task :', args.task)
    print('  --> # of training data :', len(trainset), '/ # of validation data :', len(valset))

    print('TOTAL STEPS OF DATASET FOR TRAINING')
    if args.task == 'Vessel' and args.patch is not None:
        if args.patch == 'small1':
            args.input_shape = (32, 32, 32, 1)
            args.stride = [8, 8, 8]
        elif args.patch == 'medium':
            args.input_shape = (32, 64, 64, 1)
            args.stride = [8, 16, 16]
        elif args.patch == 'large':
            args.input_shape = (32, 128, 128, 1)
            args.stride = [8, 32, 32]
        else:
            raise ValueError('If you choose \'Vessel task\', you must select a patch type.')

        print('========== trainset ==========')
        steps_per_epoch = calc_vessel_dataset(trainset, args.task, args.input_shape, args.stride)
        print('    -->', steps_per_epoch)
        print('========== valset ==========')
        validation_steps = calc_vessel_dataset(valset, args.task, args.input_shape, args.stride)
        print('    -->', validation_steps)
    
    else:
        args.input_shape = (None, None, None, 1)
        args.stride = 1
        print('========== trainset ==========')
        steps_per_epoch = len(trainset)
        print('    -->', steps_per_epoch)
        print('========== valset ==========')
        validation_steps = len(valset)
        print('    -->', validation_steps)

    for k, v in vars(args).items():
        print('{} : {}'.format(k, v))

    generator_dict = {'multi_organ': Generator_multiorgan,
                      'Liver': Generator_Liver,
                      'HCC1': Generator_HCC,
                      'HCC2': Generator_HCC,
                      'HCC3': Generator_HCC,
                      'Vessel': Generator_Vessel}

    ##############################################
    # Set Model
    ##############################################
    model = MyModel(args)

    if args.summary:
        from keras.utils import plot_model
        plot_model(model.mymodel, to_file='./model.png', show_shapes=True)
        model.mymodel.summary()
        return

    if args.checkpoint:
        model.mymodel.load_weights(args.checkpoint)
        print("Load weights successfully at {}".format(args.checkpoint))
        args.initial_epoch = int(args.checkpoint.split('/')[-1].split('_')[-2])
        stamp = args.checkpoint.split('/')[5]
    else:
        args.initial_epoch = 0
        stamp = time.strftime("%c").replace(":", "_").replace(" ", "_")
    
    print("Initial epoch :", args.initial_epoch)
    print("Stamp :", stamp)

    model.compile(args.optimizer, args.lr)

    ##############################################
    # Set Callbacks
    ##############################################
    if args.callback:
        set_cbdir(args, stamp)

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

    model.mymodel.fit_generator(generator=train_generator,
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