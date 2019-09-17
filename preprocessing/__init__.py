from .cdss_vessel_patch import calc_vessel_dataset

import os
from sklearn.model_selection import train_test_split

def load(main_args, sub_args, seed=42, test_size=.1):
    if not os.path.isfile(os.path.join(sub_args['etc']['result_root'], 
                                       '{}_trainset.txt'.format(sub_args['task']['subtask']))):

        if main_args.task == 'CDSS_Liver':
            if sub_args['task']['subtask'] == 'multi_organ':
                whole_cases = [c for c in os.listdir(os.path.join(sub_args['etc']['data_root'],
                                                                  sub_args['task']['subtask'],
                                                                  'image')) if 'hdr' in c]

            elif sub_args['task']['subtask'] == 'Liver':
                pass

            elif sub_args['task']['subtask'] == 'HCC':
                pass

            elif sub_args['task']['subtask'] == 'Vessel':
                whole_cases = [c[:-11] for c in os.listdir(os.path.join(sub_args['etc']['data_root'],
                                                                        sub_args['task']['subtask'],
                                                                        'imagesTr')) if 'hdr' in c]

            train_idx, test_idx = train_test_split(range(1, len(whole_cases)+1), test_size=test_size, random_state=seed)
            trainset = [whole_cases[x-1] for x in train_idx]
            testset = [whole_cases[x-1] for x in test_idx]
            print(testset)

        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_trainset.txt'.format(sub_args['task']['subtask'])), 'w') as f:
            for t in trainset:
                f.write(t+'\n')

        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_testset.txt'.format(sub_args['task']['subtask'])), 'w') as f:
            for t in testset:
                f.write(t+'\n')

    else:
        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_trainset.txt'.format(sub_args['task']['subtask'])), 'r') as f:
            trainset = [t.rstrip() for t in f.readlines()]
        
        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_testset.txt'.format(sub_args['task']['subtask'])), 'r') as f:
            testset = [t.rstrip() for t in f.readlines()]

    valset = trainset[:len(testset)]
    trainset = trainset[len(testset):]
    print('# of training data :', len(trainset), '/ # of validation data :', len(valset), '/ # of test data :', len(testset))
    return trainset, valset, testset


class Preprocessing(Object):
    def __init__(self, sub_args, rotation_range=[0., 0., 0.]):
        self.sub_args = sub_args
        self.rotation_range = rotation_range

        self.windowing_min = sub_args['hyperparameter']['wlevel'] - sub_args['hyperparameter']['wwidth']//2
        self.windowing_max = sub_args['hyperparameter']['wlevel'] + sub_args['hyperparameter']['wwidth']//2

    def _array2img(self, x, ismask=False):
        raise NotImplementedError('_array2img method is not implemented.')

    def _resize(self, x, ismask=False):
        raise NotImplementedError('_resize method is not implemented.')

    def _getvoi(self, x, voi, ismask=False):
        raise NotImplementedError('_getvoi method is not implemented.')

    def _windowing(self, x):
        raise NotImplementedError('_windowing method is not implemented.')

    def _standard(self, x):
        raise NotImplementedError('_standard method is not implemented.')

    def _expand(self, x, ismask=False):
        raise NotImplementedError('_expand method is not implemented.')

    def _onehot(self, x):
        raise NotImplementedError('_onehot method is not implemented.')

    def _flip(self, x, ismask=False):
        raise NotImplementedError('_flip method is not implemented.')

    def _rotation(self, x, theta=None, dep_index=0, row_index=1, col_index=2, ismask=False):
        raise NotImplementedError('_rotation method is not implemented.')

    def __transform_matrix_offset_center(self, matrix, x, y, z):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5
        offset_matrix = np.array([[1, 0, 0, o_x],
                                  [0, 1, 0, o_y], 
                                  [0, 0, 1, o_z], 
                                  [0, 0, 0, 1]])
        reset_matrix = np.array([[1, 0, 0, -o_x], 
                                 [0, 1, 0, -o_y], 
                                 [0, 0, 1, -o_z], 
                                 [0, 0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def __apply_transform(self, x, transform_matrix, fill_mode='nearest', cval=0.):
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]
        x = ndimage.interpolation.affine_transform(x, 
                                                   final_affine_matrix, 
                                                   final_offset, 
                                                   order=0, 
                                                   mode=fill_mode, 
                                                   cval=cval)
        return x


