# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np

import keras
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

SEED = 42
np.random.seed(SEED)

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import task_keras.bin  # noqa: F401
    __package__ = "task_keras.bin"

from . import get_session, get_arguments, check_yaml, set_cbdir, create_generator, create_callbacks
from .. import models
from .. import callbacks
from .. import preprocessing


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
                          else preprocessing.calc_vessel_dataset(trainset, 
                                                                 sub_args['task']['subtask'], 
                                                                 sub_args['etc']['data_root'],
                                                                 sub_args['etc']['result_root'],
                                                                 sub_args['hyperparameter']['input_shape'], 
                                                                 sub_args['hyperparameter']['stride'])
        print('    -->', steps_per_epoch)
        print('========== valset ==========')
        validation_steps = preprocessing.calc_vessel_dataset(valset, 
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
        backbone = models.backbone(main_args, sub_args)
        model = backbone.unet()
    elif main_args.mode == 'classification':
        model = models.classification.set_model(main_args=main_args,
                                                sub_args=sub_args, 
                                                base_filter=int(32//sub_args['hyperparameter']['divide']))
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

    optimizer, loss, metric = models.compile(main_args, sub_args)
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=metric)


    ##############################################
    # Set Generator
    ##############################################
    train_generator = create_generator(main_args=main_args, 
                                       sub_args=sub_args, 
                                       datalist=trainset, 
                                       mode='train', 
                                       rotation_range=[5., 5., 5.])

    val_generator = create_generator(main_args=main_args, 
                                     sub_args=sub_args, 
                                     datalist=valset, 
                                     mode='validation', 
                                     rotation_range=[0., 0., 0.])

    ##############################################
    # Set Callbacks
    ##############################################
    callbacks = create_callbacks(main_args, sub_args, val_generator, validation_steps) if sub_args['etc']['callback'] else []


    ##############################################
    # Train
    ##############################################
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch//sub_args['hyperparameter']['batch_size'],
                        verbose=1,
                        epochs=sub_args['hyperparameter']['epochs'],
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        shuffle=True,
                        initial_epoch=sub_args['hyperparameter']['initial_epoch'])


if __name__ == "__main__":
    main()