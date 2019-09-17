import random
import threading
import numpy as np
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
def Generator_multiorgan(
    sub_args,
    datalist,
    mode,
    rotation_range=[10., 10., 10.],
    seed=42,
    shuffle=True,
    **kwargs):

    def _preprocessing(img, mask, prep):
        img, mask = prep._array2img(img), prep._array2img(mask, ismask=True)
        img, mask = prep._resize(img), prep._resize(mask, ismask=True)

        if mode == 'training':
            theta = [np.pi / 180 * np.random.uniform(-rr, rr) for rr in self.rotation_range]
            img, mask = prep._rotation(img, theta), prep._rotation(mask, theta, ismask=True)

        img = prep._windowing(img)
        img = prep._standard(img)
        if sub_args['hyperparameter']['classes'] > 1:
            mask = prep._onehot(mask)

        img, mask = prep._expand(img), prep._expand(mask, ismask=True)

        return img, mask

    random.seed(seed)
    prep = Prep_Segmentation(sub_args=sub_args,
                             rotation_range=rotation_range)

    while True:
        if shuffle:
            random.shuffle(datalist)

        for data in datalist:
            if mode == 'test':
                print(data)

            img = sitk.ReadImage(os.path.join(sub_args['etc']['data_root'], sub_args['task']['subtask'], 'image', data))
            mask = sitk.ReadImage(os.path.join(sub_args['etc']['data_root'], sub_args['task']['subtask'], 'mask', data))
            img, mask = _preprocessing(img, mask, prep)

            yield img, mask


@threadsafe_generator
def Generator_Vessel(
    sub_args,
    datalist,
    mode,
    rotation_range=[10., 10., 10.],
    seed=42,
    shuffle=True,
    **kwargs):

    voi = json.loads(open(os.path.join(sub_args['etc']['result_root'], 'Vessel_voilist.json')).read())

    def _preprocessing(img, mask, prep, voirange):
        img, mask = prep._array2img(img), prep._array2img(mask, ismask=True)

        if sub_args['hyperparameter']['classes'] == 1:
            mask[mask == 2.] = 0. # remove tumors

        img, mask = prep._getvoi(img, voirange), prep._getvoi(mask, voirange, ismask=True)

        if mode == 'training':
            prob = np.random.random()
            img, mask = prep._flip(img, prob), prep._flip(mask, prob, ismask=True)

            theta = [np.pi / 180 * np.random.uniform(-rr, rr) for rr in self.rotation_range]
            img, mask = prep._rotation(img, theta), prep._rotation(mask, theta, ismask=True)

        img = prep._windowing(img)
        img = prep._standard(img)
        if sub_args['hyperparameter']['classes'] > 1:
            mask = prep._onehot(mask)

        img, mask = prep._expand(img), prep._expand(mask, ismask=True)

        return img, mask

    random.seed(seed)
    prep = Prep_Segmentation(sub_args=sub_args,
                             rotation_range=rotation_range)

    if sub_args['hyperparameter']['patch']:
        if isinstance(sub_args['hyperparameter']['stride'], int):
            sub_args['hyperparameter']['stride'] = [sub_args['hyperparameter']['stride'] for i in range(3)]
        
        for i in range(3):
            if sub_args['hyperparameter']['stride'][i] > sub_args['hyperparameter']['input_shape'][i]:
                raise ValueError("You must set strides that are shorter than input shape.")

        img_input = np.zeros((sub_args['hyperparameter']['batch_size'],)+sub_args['hyperparameter']['input_shape'])
        mask_input = np.zeros((sub_args['hyperparameter']['batch_size'],)+sub_args['hyperparameter']['input_shape'][:3]+(sub_args['hyperparameter']['classes'],))
        batch = 0
        while True:
            if shuffle:
                random.shuffle(datalist)

            for data in datalist:
                if mode == 'test':
                    print(data)

                img = sitk.ReadImage(os.path.join(INPUT_FOLDER, args.task, 'imagesTr', data+'_resize.hdr'))
                mask = sitk.ReadImage(os.path.join(INPUT_FOLDER, args.task, 'labelsTr', data+'_resize.hdr'))
                img, mask = _preprocessing(img, mask, prep, voi[data+'_resize'])
                
                patch_range = [(s-args.input_shape[i])//args.stride[i]+1 if mode != 'test' else (s-args.input_shape[i])//args.stride[i]+1+1 for i, s in enumerate(img.shape[1:-1])]
                patch_shuffle = np.random.permutation(np.array(patch_range).prod()) if mode != 'test' else np.arange(np.array(patch_range).prod())
                for patch in patch_shuffle:
                    zloc = patch % patch_range[0]
                    yloc = (patch // patch_range[0]) % patch_range[1]
                    xloc = patch // (patch_range[0]*patch_range[1])

                    if mode == 'test':
                        if zloc+1 == patch_range[0]:
                            if yloc+1 == patch_range[1]:
                                if xloc+1 == patch_range[2]:
                                    img_input[batch] = img[0,-args.input_shape[0]:,-args.input_shape[1]:,-args.input_shape[2]:]
                                    mask_input[batch] = mask[0,-args.input_shape[0]:,-args.input_shape[1]:,-args.input_shape[2]:]
                                else:
                                    img_input[batch] = img[0,-args.input_shape[0]:,-args.input_shape[1]:,
                                                        xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                                    mask_input[batch] = mask[0,-args.input_shape[0]:,-args.input_shape[1]:,
                                                            xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                            else:
                                if xloc+1 == patch_range[2]:
                                    img_input[batch] = img[0,-args.input_shape[0]:,
                                                        yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],-args.input_shape[2]:]
                                    mask_input[batch] = mask[0,-args.input_shape[0]:,
                                                            yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],-args.input_shape[2]:]
                                else:
                                    img_input[batch] = img[0,-args.input_shape[0]:,
                                                        yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                                                        xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                                    mask_input[batch] = mask[0,-args.input_shape[0]:,
                                                            yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                                                            xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                        else:
                            if yloc+1 == patch_range[1]:
                                if xloc+1 == patch_range[2]:
                                    img_input[batch] = img[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                                                        -args.input_shape[1]:,-args.input_shape[2]:]
                                    mask_input[batch] = mask[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                                                            -args.input_shape[1]:,-args.input_shape[2]:]
                                else:
                                    img_input[batch] = img[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],-args.input_shape[1]:,
                                                        xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                                    mask_input[batch] = mask[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],-args.input_shape[1]:,
                                                            xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                            else:
                                if xloc+1 == patch_range[2]:
                                    img_input[batch] = img[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                                                        yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],-args.input_shape[2]:]
                                    mask_input[batch] = mask[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                                                            yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],-args.input_shape[2]:]
                                else:
                                    img_input[batch] = img[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                                                        yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                                                        xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                                    mask_input[batch] = mask[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                                                            yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                                                            xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]

                        batch += 1
                        if batch >= args.batch_size:
                            yield img_input, mask_input, data, patch_range, [zloc, yloc, xloc]
                            batch = 0
                            img_input = np.zeros((args.batch_size,)+args.input_shape)
                            mask_input = np.zeros((args.batch_size,)+args.input_shape[:3]+(args.classes,))

                    else:
                        # img_input[batch] = img[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                        #                     yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                        #                     xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                                    
                        # mask_input[batch] = mask[0,zloc*args.stride[0]:zloc*args.stride[0]+args.input_shape[0],
                        #                         yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                        #                         xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                        img_input[batch] = img[0,:,
                                            yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                                            xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]
                                    
                        mask_input[batch] = mask[0,:,
                                                yloc*args.stride[1]:yloc*args.stride[1]+args.input_shape[1],
                                                xloc*args.stride[2]:xloc*args.stride[2]+args.input_shape[2]]

                        batch += 1
                        if batch >= args.batch_size:
                            yield img_input, mask_input
                            batch = 0
                            img_input = np.zeros((args.batch_size,)+args.input_shape)
                            mask_input = np.zeros((args.batch_size,)+args.input_shape[:3]+(args.classes,))
    
    else:
        while True:
            if shuffle:
                random.shuffle(datalist)

            for data in datalist:
                if mode == 'test':
                    print(data)

                img = sitk.ReadImage(os.path.join(sub_args['etc']['data_root'], sub_args['task']['subtask'], 'imagesTr', data+'_resize.hdr'))
                mask = sitk.ReadImage(os.path.join(sub_args['etc']['data_root'], sub_args['task']['subtask'], 'labelsTr', data+'_resize.hdr'))
                img, mask = _preprocessing(img, mask, prep, voi[data+'_resize'])

                yield img, mask


class Prep_Segmentation(Preprocessing):
    mean_std = {'multi_organ': [29.311405133024834, 43.38181786843102],
                'Vessel': [29.311405133024834, 43.38181786843102]}

    windowing_range = {'multi_organ': [100., 180.], # Portal!!!
                       'Liver': [100., 180.],
                       'HCC1': [50., 300.],
                    #    'Vessel': [88., 150.]}
                    #    'Vessel': [100., 180.]}
                    #    'Vessel': [150., 300.]}
                       'Vessel': [50., 300.]}

    def _array2img(self, x, ismask=False):
        if ismask and isinstance(x, list):
            return [sitk.GetArrayFromImage(m).astype('float32') for m in x]
        else:
            return sitk.GetArrayFromImage(x).astype('float32')

    def _resize(self, x, ismask=False):
        if ismask and isinstance(x, list):
            return [ndimage.zoom(m, [1., 1./self.sub_args['hyperparameter']['resize_rate'], 1./self.sub_args['hyperparameter']['resize_rate']], 
                                 order=0, mode='constant', cval=0.) for m in x]
        else:
            return ndimage.zoom(x, [1., 1./self.sub_args['hyperparameter']['resize_rate'], 1./self.sub_args['hyperparameter']['resize_rate']], 
                                order=0, mode='constant', cval=0.)

    def _getvoi(self, x, voi, ismask=False):
        if ismask and isinstance(x, list):
            return [m[voi[0]:voi[1],voi[2]:voi[3],voi[4]:voi[5]] for m in x]
        else:
            return x[voi[0]:voi[1],voi[2]:voi[3],voi[4]:voi[5]]

    def _windowing(self, x):
        return np.clip(x, self.windowing_min, self.windowing_max)

    def _standard(self, x):
        if self.standard == 'minmax':
            return (x - self.windowing_min) / (self.windowing_max - self.windowing_min)
        elif self.standard == 'norm':
            return (x - self.mean_std[self.sub_args['task']['subtask']][0]) / self.mean_std[self.sub_args['task']['subtask']][1]
        elif self.standard == 'eachnorm':
            return (x - x.mean()) / x.std()
        else:
            return x

    def _expand(self, x, ismask=False):
        if ismask:
            if isinstance(x, list):
                bg = np.ones_like(x[0])
                for i in range(len(x)):
                    bg -= x[i]
                bg = np.clip(bg, 0., 1.)
                return np.concatenate([bg[np.newaxis,...,np.newaxis]]+[m[np.newaxis,...,np.newaxis] for m in x], axis=-1)
            else:
                return x[np.newaxis,...] if sub_args['hyperparameter']['classes'] > 1 else x[np.newaxis,...,np.newaxis]
        else:
            return x[np.newaxis,...,np.newaxis]
    
    def _onehot(self, x):
        result = np.zeros(x.shape+(self.sub_args['hyperparameter']['classes'],))
        for i in range(self.sub_args['hyperparameter']['classes']):
            result[...,i][np.where(x == i)] = 1.
        return result

    def _flip(self, x, prob, ismask=False):
        if prob < 0.25:
            return [m[:,::-1,::-1] for m in x] if ismask and isinstance(x, list) else x[:,::-1,::-1]
        elif 0.25 <= prob < 0.5:
            return [m[:,:,::-1] for m in x] if ismask and isinstance(x, list) else x[:,:,::-1]
        elif 0.5 <= prob < 0.75:
            return [m[:,::-1,:] for m in x] if ismask and isinstance(x, list) else x[:,::-1,:]
        else:
            return x

    def _rotation(self, x, theta, dep_index=0, row_index=1, col_index=2, ismask=False):
        theta1, theta2, theta3 = theta
        
        # theta1 = np.pi / 180 * np.random.uniform(-self.rotation_range[0], self.rotation_range[0])
        # theta2 = np.pi / 180 * np.random.uniform(-self.rotation_range[1], self.rotation_range[1])
        # theta3 = np.pi / 180 * np.random.uniform(-self.rotation_range[2], self.rotation_range[2])
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
        if ismask and isinstance(mask, list):
            return [self.__apply_transform(m, transform_matrix, fill_mode, cval) for m in x]
        else:
            return self.__apply_transform(x, transform_matrix, fill_mode, cval)