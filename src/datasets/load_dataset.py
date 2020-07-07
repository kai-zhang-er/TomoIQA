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


# three types of distortions: gaussian blur, gaussian noise, ring artifact
sp = 224
blur_sigma = [1.2, 2.5, 6.5, 15.2, 33.2]
noise_variance = [0.001, 0.006, 0.022, 0.088, 1.00]
ring_artifact = [5, 10, 20, 40, 80]  # radius of rings

class Dataset():

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
        self.scores = {}
        # self.dir="/gdrive/My Drive/pyIQA/dataset/cropped/"
        for i in filename:
            image_name=i.split(",")[0]
            self.image_path.append(image_name)
            self.scores.setdefault(image_name,float(i.split(",")[1]))
        print(len(list(self.scores.keys())),self.scores.keys())
        # print(self.scores["l02_2_0_0.bmp"])
    def next_batch(self):
        """Get blobs and copy them into this layer's top blob vector."""
        base_image_index=random.randint(0,len(self.image_path)-1)
        crop_image=preprocess(self.image_path[base_image_index])

        # print("train image:"+self.image_path[base_image_index]+"\n")
        data_batch,score_batch=generate_distortion(crop_image)
        return data_batch,score_batch
    def next_batch_finetune(self):
        base_image_index=random.randint(0,len(self.image_path)-1)
        image_name=self.image_path[base_image_index]
        image=cv2.imread(self.root_dir+"/dataset/cropped/"+image_name[:3]+"_0_0_0.bmp")
        # print(self.root_dir+"/dataset/cropped/"+image_name[:3]+"_0_0_0.bmp")
        return generate_distortion_finetune(image,image_name,self.scores)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

# 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255;
# caffe: [B,C,W,H]
# tensorflow模型转caffe模型时，遇到了几个坑其中之一就是caffe的padding方式和tensorflow的padding方式有很大的区别，导致每一层的输出都无法对齐;
# 卷积层的通道顺序：在caffe里是[N,C,H,W]，而tensorflow是[H,W,C,N]
# fc层的通道顺序：在caffe 里是[c_in,c_out]，而tensorflow是[c_out,c_in]
def preprocess(data):
    # print(data)
    
    im = cv2.imread(data)
    image = cv2.resize(im, (255, 255),interpolation = cv2.INTER_AREA)
    x = image.shape[0]
    y = image.shape[1]
    x_p = np.random.randint(x - sp, size=1)[0]
    y_p = np.random.randint(y - sp, size=1)[0]
    image = image[x_p:x_p + sp, y_p:y_p + sp, :]

    image=image/255
    # print x_p,y_p
    #  # caffe: .transpose([2, 0, 1])
    # print images.shape
    return image

def generate_distortion(crop_img):
    distortion_level=len(blur_sigma)
    score_batch=[]
    image_batch=[]
    image_batch.append(crop_img)
    score_batch.append(1)
    for j in range(distortion_level):
        blur_crop=filters.gaussian(crop_img,sigma=blur_sigma[j])
        score_crop=ssim(crop_img,blur_crop,multichannel=True,dynamic_range=blur_crop.max() - blur_crop.min())
        image_batch.append(blur_crop)
        score_batch.append(score_crop)

    image_batch.append(crop_img)
    score_batch.append(1)
    for j in range(distortion_level):
        noise_crop = util.random_noise(crop_img, var=noise_variance[j])
        score_crop = ssim(crop_img, noise_crop, multichannel=True,dynamic_range=noise_crop.max() - noise_crop.min())
        image_batch.append(noise_crop)
        score_batch.append(score_crop)

    image_batch.append(crop_img)
    score_batch.append(1)
    for j in range(distortion_level):
        rr, cc = draw.circle(sp // 2, sp // 2, radius=ring_artifact[j])
        ring_crop = crop_img.copy()
        ring_crop[rr, cc] = 100
        score_crop = ssim(crop_img, ring_crop, multichannel=True,dynamic_range=ring_crop.max() - ring_crop.min())
        image_batch.append(ring_crop)
        score_batch.append(score_crop)

    return np.array(image_batch), 1-np.array(score_batch) # 0=>good, 1=>bad

def generate_distortion_finetune(crop_img,image_name,score_dict):
    distortion_level=len(blur_sigma)
    score_batch=[]
    image_batch=[]
    image_batch.append(crop_img)
    score_batch.append(score_dict[image_name])
    for j in range(distortion_level):
        blur_crop=filters.gaussian(crop_img,sigma=blur_sigma[j])
        temp="{}_{}_0_0.bmp".format(image_name[:3],j+1)
        image_batch.append(blur_crop)
        score_batch.append(score_dict[temp])

    image_batch.append(crop_img)
    score_batch.append(1)
    for j in range(distortion_level):
        noise_crop = util.random_noise(crop_img, var=noise_variance[j])
        temp="{}_0_{}_0.bmp".format(image_name[:3],j+1)
        image_batch.append(noise_crop)
        score_batch.append(score_dict[temp])

    image_batch.append(crop_img)
    score_batch.append(1)
    for j in range(distortion_level):
        rr, cc = draw.circle(sp // 2, sp // 2, radius=ring_artifact[j])
        ring_crop = crop_img.copy()
        ring_crop[rr, cc] = 100
        temp="{}_0_0_{}.bmp".format(image_name[:3],j+1)
        image_batch.append(ring_crop)
        score_batch.append(score_dict[temp])

    return np.array(image_batch), np.array(score_batch) # 0=>good, 1=>bad

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
    root_dir="/gdrive/My Drive/pyIQA"
    param_str={'root_dir':root_dir,'data_root':'TOMO','split':'train_finetune','im_shape':[224,224],'batch_size':18}
    rank_data = Dataset(param_str)

    for i in range(10):
        image_batch,label_batch = rank_data.next_batch_finetune()
        # for blob_name, blob in blobs.items():
        #     print(blob_name, blob.shape)
        print(image_batch.shape)
        print(label_batch.shape)
    print(label_batch)

