import os
import json
import random
import threading
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from . import Preprocessing

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def Generator(
    sub_args,
    datalist,
    mode,
    rotation_range=[5., 5., 5.],
    seed=42,
    shuffle=True,
    **kwargs):

    def _preprocessing(img, mask, prep, voi=None):
        img, mask = prep._array2img(img), prep._array2img(mask)
        img, mask = prep._getvoi(img, voi), prep._getvoi(mask, voi)
        if sub_args['hyperparameter']['only_liver']:
            img = prep._only_liver(img, mask)
            
        if sub_args['hyperparameter']['voi_mode'] == 'whole':
            img = prep._resize(img)

        if mode == 'training':
            theta = [np.pi / 180 * np.random.uniform(-rr, rr) for rr in rotation_range]
            img = prep._rotation(img, theta)

        img = prep._windowing(img)
        img = prep._standard(img)
        img = prep._expand(img)

        return img

    random.seed(seed)
    prep = Prep_Classification(sub_args=sub_args,
                               rotation_range=rotation_range)

    df = pd.read_csv(os.path.join(sub_args['etc']['result_root'], 'datalist.csv'))
    voidict = {'artery': json.load(open(os.path.join(sub_args['etc']['result_root'], 'Liver_artery_voilist.json'), 'r')),
               'portal': json.load(open(os.path.join(sub_args['etc']['result_root'], 'Liver_portal_voilist.json'), 'r')),
               'pre': json.load(open(os.path.join(sub_args['etc']['result_root'], 'Liver_pre_voilist.json'), 'r')),
               'delay': json.load(open(os.path.join(sub_args['etc']['result_root'], 'Liver_delay_voilist.json'), 'r'))}
    
    data_class = {}
    if mode == 'train':
        if sub_args['task']['subtask'] in ['ascites', 'Pvi_loca']:
            data_class[0] = datalist[datalist[sub_args['task']['subtask']] == 0]
            data_class[1] = pd.concat((df[df[sub_args['task']['subtask']] == 1], df[df[sub_args['task']['subtask']] == 2]))
            major_class = len(data_class[1])
        elif sub_args['task']['subtask'] == 'varix':
            data_class[0] = datalist[datalist[sub_args['task']['subtask']] == 0]
            data_class[1] = pd.concat((df[df[sub_args['task']['subtask']] == 1], df[df[sub_args['task']['subtask']] == 3]))
            major_class = len(data_class[1])
        elif sub_args['task']['subtask'] == 'distritubion':
            data_class[0] = datalist[datalist[sub_args['task']['subtask']] == 1.]
            data_class[1] = datalist[datalist[sub_args['task']['subtask']] == 2.]
            data_class[2] = datalist[datalist[sub_args['task']['subtask']] == 3.]
            major_class = len(data_class[1])
        elif sub_args['task']['subtask'] == 'RFA_feasiblity':
            data_class[0] = datalist[datalist[sub_args['task']['subtask']] == 0]
            data_class[1] = datalist[datalist[sub_args['task']['subtask']] == 1]
            major_class = len(data_class[0])
    else:
        data_class[0] = datalist
        major_class = len(data_class[0])
    
    img_list = [d for d in os.listdir(os.path.join(sub_args['etc']['data_root'], 'Liver/CT',
                                                   sub_args['task']['phase'].upper())) if 'hdr' in d]
    label = np.zeros((sub_args['hyperparameter']['batch_size'], sub_args['hyperparameter']['classes']))
    batch = 0

    while True:
        if shuffle:
            data_order = np.random.permutation(major_class)
        else:
            data_order = np.arange(major_class)
        
        for d in data_order:
            for k, v in data_class.items():
                data = v.iloc[d]
                try:
                    img_path = [os.path.join(sub_args['etc']['data_root'], 'Liver/CT', 
                                             sub_args['task']['phase'].upper(), img) 
                                for img in img_list if str(data.id) in img][0]

                    img = sitk.ReadImage(img_path)
                    mask = sitk.ReadImage(img_path.replace('CT', 'Mask'))
                    if mode == 'test':
                        print(img_path)

                    img = _preprocessing(img=img, 
                                         mask=mask,
                                         prep=prep,
                                         voi=voidict[sub_args['task']['phase']][img_path.split('/')[-1][:-4]])

                    if mode == 'train':
                        label[batch, k] = 1
                    else:
                        # distribution 수정할 것!!!
                        label[batch, 0 if data[sub_args['task']['subtask']] == 0 else 1] = 1

                    batch += 1
                    if batch >= sub_args['hyperparameter']['batch_size']:
                        yield img, label
                        label = np.zeros((sub_args['hyperparameter']['batch_size'], sub_args['hyperparameter']['classes']))
                        batch = 0
                except IndexError:
                    # if not file
                    pass
                # except KeyboardInterrupt:
                #     raise


class Prep_Classification(Preprocessing):
    mean_std = []

    def _array2img(self, x):
        return sitk.GetArrayFromImage(x).astype('float32')

    def _resize(self, x):
        return ndimage.zoom(x, [1., 1./self.sub_args['hyperparameter']['resize'], 1./self.sub_args['hyperparameter']['resize']], 
                            order=0, mode='constant', cval=0.)

    def _getvoi(self, x, voi, is_mask=False):
        if self.sub_args['hyperparameter']['voi_mode'] == 'local':
            if self.sub_args['task']['subtask'] in ['ascites', 'varix', 'RFA_feasiblity', 'distritubion']:
                return x[voi[0]:voi[1],voi[2]:voi[3],voi[4]:voi[5]]
            elif self.sub_args['task']['subtask'] in ['Pvi_loca']:
                return x[voi[0]:voi[1]+3,voi[2]:voi[3],voi[4]:voi[5]]
        else:
            if self.sub_args['task']['subtask'] in ['ascites', 'varix', 'RFA_feasiblity', 'distritubion']:
                return x[voi[0]:voi[1],:,:]
            elif self.sub_args['task']['subtask'] in ['Pvi_loca']:
                return x[voi[0]:voi[1]+3,:,:]

    def _windowing(self, x):
        return np.clip(x, self.windowing_min, self.windowing_max)

    def _standard(self, x):
        if self.sub_args['hyperparameter']['standard'] == 'minmax':
            return (x - self.windowing_min) / (self.windowing_max - self.windowing_min)
        elif self.sub_args['hyperparameter']['standard'] == 'norm':
            # will be editted
            return (x - self.mean_std[0]) / self.mean_std[1]
        elif self.sub_args['hyperparameter']['standard'] == 'eachnorm':
            return (x - x.mean()) / x.std()
        else:
            return x

    def _only_liver(self, x, y):
        x[np.where(y == y.min())] = x.min()
        return x

    def _expand(self, x):
        return x[np.newaxis,...,np.newaxis]

    def _rotation(self, x, theta=None, dep_index=0, row_index=1, col_index=2, is_mask=False):
        theta1, theta2, theta3 = theta

        rotation_matrix_z = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0],
                                      [np.sin(theta1), np.cos(theta1), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        rotation_matrix_y = np.array([[np.cos(theta2), 0, -np.sin(theta2), 0],
                                      [0, 1, 0, 0],
                                      [np.sin(theta2), 0, np.cos(theta2), 0],
                                      [0, 0, 0, 1]])
        rotation_matrix_x = np.array([[1, 0, 0, 0],
                                      [0, np.cos(theta3), -np.sin(theta3), 0],
                                      [0, np.sin(theta3), np.cos(theta3), 0],
                                      [0, 0, 0, 1]])

        rotation_matrix = np.dot(np.dot(rotation_matrix_y, rotation_matrix_z), rotation_matrix_x)

        d, h, w = x.shape[dep_index], x.shape[row_index], x.shape[col_index]
        transform_matrix = self.__transform_matrix_offset_center(rotation_matrix, d, w, h)
        x = self.__apply_transform(x, transform_matrix, 'nearest', 0.)

        return x