#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created load_dataset.py by zk in 2020-04-12.
"""

# coding: utf-8
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import os.path as osp
from skimage import io,filters,util,draw
import random
import tensorflow.keras as keras

# three types of distortions: gaussian blur, gaussian noise, ring artifact
sp = 224
blur_sigma = [1.2, 2.5, 6.5, 15.2, 33.2]
noise_variance = [0.001, 0.006, 0.022, 0.088, 1.00]
ring_artifact=[0.1,0.13,0.15,0.18,0.2] # coefficient for rings
ring_width=2
ring_number=30


class DataGenerator(keras.utils.Sequence):
    def __init__(self, param_str):
        # === Read input parameters ===
        # params is a python dictionary with layer parameters.
        self.param_str = param_str
        # Check the paramameters for validity.
        check_params(self.param_str)

        # store input as class variables
        self.batch_size = self.param_str['batch_size']
        self.root_dir = self.param_str['root_dir']
        self.data_root = self.param_str['data_root']
        self.im_shape = self.param_str['im_shape']

        # get list of image indexes.
        list_file = self.param_str['split'] + '.txt'
        filename = [line.rstrip('\n') for line in open(osp.join(self.root_dir,self.data_root, list_file)).readlines()]
        self.image_path = []
        for i in filename:
            image_name=i.split(",")[0]
            self.image_path.append(image_name)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # generate a batch of data
        X, y = self.__data_generation(self.root_dir+self.image_path[index])
        return X, y

    def __data_generation(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
        elif image.shape[0]>255:
            # resize image
            image = cv2.resize(image, (255, 255), interpolation=cv2.INTER_AREA)
            # randomly crop a part
            x = image.shape[0]
            y = image.shape[1]
            x_p = np.random.randint(x - sp, size=1)[0]
            y_p = np.random.randint(y - sp, size=1)[0]
            image = image[x_p:x_p + sp, y_p:y_p + sp, :]
        image = image / 255
        image_batch, label_batch=generate_distortion(image)
        return image_batch, label_batch


def generate_distortion(crop_img):
    distortion_level=len(blur_sigma)
    image_batch=[]
    image_batch.append(crop_img)
    for j in range(distortion_level):
        blur_crop=filters.gaussian(crop_img,sigma=blur_sigma[j],multichannel=True)
        image_batch.append(blur_crop)
        
    image_batch.append(crop_img)
   
    for j in range(distortion_level):
        noise_crop = util.random_noise(crop_img, var=noise_variance[j])
        image_batch.append(noise_crop)
        

    image_batch.append(crop_img)
    for j in range(distortion_level):
        ring_radius = 5
        ring_crop = crop_img.copy()
        temp = crop_img.copy()
        for i in range(ring_number):
            rr1, cc1 = draw.circle(sp // 2, sp // 2, radius=ring_radius, shape=(sp, sp))
            rr2, cc2 = draw.circle(sp // 2, sp // 2, radius=ring_radius + ring_width,
                                   shape=(sp, sp))
            ring_radius += random.randint(3, 10)
            if ring_radius > 100:
                break

            # add rings to image
            temp[rr1, cc1] = 0
            if i % 2 == 1:
                ring_crop[rr2, cc2] += ring_artifact[j] * temp[rr2, cc2]
            else:
                ring_crop[rr2, cc2] -= ring_artifact[j] * temp[rr2, cc2]
        ring_crop[ring_crop > 1] = 1
        ring_crop[ring_crop < 0] = 0
        image_batch.append(ring_crop)

    return np.array(image_batch), np.zeros((len(image_batch),1)) # for ranking process, no need for labels

def check_params(params):
    """
    A utility function to check the parameters for the TOMO layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'data_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


if __name__=="__main__":
    root_dir="/content/drive/My Drive/pyIQA/"
    param_str={'root_dir':root_dir,'data_root':'TOMO/','split':'train_info','im_shape':[224,224],'batch_size':18}
    rank_data = DataGenerator(param_str)

    for i in range(1):
        image_batch,label_batch = rank_data.__getitem__(2)
        # for blob_name, blob in blobs.items():
        #     print(blob_name, blob.shape)
        print(image_batch.shape)
        print(label_batch.shape)
    print(label_batch)

