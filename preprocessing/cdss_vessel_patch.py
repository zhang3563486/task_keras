import os
import json
import numpy as np
import SimpleITK as sitk

def calc_vessel_dataset(datalist,
                        subtask,
                        data_root,
                        result_root,
                        input_shape=(8, 32, 32, 1),
                        strides=4,
                        mode='training'):
    
    if isinstance(strides, int):
        strides = [strides for i in range(3)]
    
    for i in range(3):
        if strides[i] > input_shape[i]:
            raise ValueError("You must set strides that are shorter than input shape.")

    voilist = json.load(open(os.path.join(result_root, 'Vessel_voilist.json'), 'r'))

    cnt = 0
    for i, data in enumerate(datalist):
        voi = voilist[data+'_resize']
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_root, subtask, 'imagesTr', data+'_resize.hdr'))).astype('float32')
        img = img[voi[0]:voi[1]+1,voi[2]:voi[3]+1,voi[4]:voi[5]+1]
        # patch_range = [(s-input_shape[i])//strides[i]+1 if mode != 'test' else (s-input_shape[i])//strides[i]+1+1 for i, s in enumerate(img.shape)]
        patch_range = [1] + [(s-input_shape[i])//strides[i]+1 if mode != 'test' else (s-input_shape[i])//strides[i]+1+1 for i, s in enumerate(img.shape) if i > 0]
        patch_shuffle = np.random.permutation(np.array(patch_range).prod())
        cnt += len(patch_shuffle)

        if i % 10 == 0:
            print('{}/{}'.format(i+1, len(datalist)), data, patch_range)

    return cnt