from skimage import io,filters,util,draw,img_as_ubyte
import numpy as np
import cv2
import os
import random

blur_sigma = [1.2, 2.5, 6.5, 15.2, 33.2]
noise_variance = [0.001, 0.006, 0.022, 0.088, 1.00]
coefficients=[0.1,0.13,0.15,0.18,0.2]

data_root_dir="dataset/uncertainty/"  # dataset root path
data_dir=data_root_dir  # directory of original image data
save_distorted_dir=data_root_dir+"distorted/"  # output distorted image save directory

distortion_level=5
resized_size=1024
crop_size=1024
ring_width=2
ring_number=50

def add_distortion(image_name, crop_img):
    for j in range(distortion_level):
        noise_crop = util.random_noise(crop_img, var=noise_variance[j])
        io.imsave(save_distorted_dir + "{}_{}_{}_{}.bmp".format(image_name, 0, j + 1, 0), (255*noise_crop).astype(np.uint8))
        blur_crop=filters.gaussian(crop_img,sigma=blur_sigma[j])
        io.imsave(save_distorted_dir + "{}_{}_{}_{}.bmp".format(image_name, j + 1, 0, 0), (255*blur_crop).astype(np.uint8))

        ori_img=crop_img.copy()
        ring_radius = 20
        for i in range(ring_number):
            rr1, cc1 = draw.circle(crop_size // 2, crop_size // 2, radius=ring_radius, shape=(crop_size,crop_size))
            rr2, cc2 = draw.circle(crop_size // 2, crop_size // 2, radius=ring_radius+ring_width, shape=(crop_size,crop_size))
            ring_radius+=random.randint(5,25)
            if ring_radius>400:
                break
            ring_crop = ori_img.copy()
            ring_crop[rr1,cc1]=0
            if i%2==1:
                ori_img[rr2,cc2]+=coefficients[j]*ring_crop[rr2,cc2]
            else:
                ori_img[rr2, cc2] -= coefficients[j] * ring_crop[rr2, cc2]
        ori_img[ori_img>1]=1
        ori_img[ori_img<0]=0
        result_name="{}_0_0_{}.bmp".format(image_name[:3],j+1)
        io.imsave(save_distorted_dir+result_name,img_as_ubyte(ori_img))


image_list=os.listdir(data_dir)
for item in image_list:
    if item[-3:]!="bmp":
        continue
    image_ind=item[:3]
    print(item)
    img=util.img_as_float(io.imread(data_dir+item))
    crop_image=img
    if img.shape[0]!=crop_size or img.shape[1]!=crop_size:
        img_resized=cv2.resize(img,(resized_size,resized_size),interpolation=cv2.INTER_AREA)
        x_p = np.random.randint(resized_size - crop_size, size=1)[0]
        y_p = np.random.randint(resized_size - crop_size, size=1)[0]
        crop_image=img_resized[x_p:x_p + crop_size, y_p:y_p + crop_size]
    add_distortion(image_ind,crop_image)
    io.imsave(save_distorted_dir + "{}_0_0_0.bmp".format(image_ind), (255*crop_image).astype(np.uint8))