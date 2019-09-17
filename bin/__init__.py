import keras
import tensorflow as tf
from collections import OrderedDict

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

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