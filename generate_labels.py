import random
from skimage import filters,util,draw,io
from skimage.measure import compare_ssim,compare_mse,compare_psnr
from metrics.mdsi import gray_mdsi
from metrics.gmsd import gmsd
from utils import ex
import json


@ex.capture
def calculate_score(crop_image):
    # assume crop_image: 224*224
    blur_score=[]
    noise_score=[]
    ring_score=[]

    for j in range(distortion_level):
        # calculate the score for image with blur
        blur_crop = filters.gaussian(crop_image, sigma=blur_sigma[j])
        blur_crop=blur_crop*255
        mse_blur = compare_mse(crop_image, blur_crop)
        mdsi_blur=gray_mdsi(crop_image,blur_crop,f=0,alpha=0.6)
        ssim_blur=compare_ssim(crop_image,blur_crop,dynamic_range=blur_crop.max() - blur_crop.min())
        gmsd_blur=gmsd(crop_image,blur_crop)
        pnsr_blur=compare_psnr(crop_image,blur_crop)
        info_blur = "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\r".format(mse_blur,mdsi_blur,ssim_blur,gmsd_blur,pnsr_blur)
#         print(info_blur)
        blur_score.append(info_blur)

        # calculate the score for image with noise
        noise_crop = util.random_noise(crop_image, var=noise_variance[j])
        noise_crop = noise_crop * 255
        mse_noise = compare_mse(crop_image, noise_crop)
        mdsi_noise = gray_mdsi(crop_image, noise_crop, f=0, alpha=0.6)
        ssim_noise = compare_ssim(crop_image, noise_crop,dynamic_range=noise_crop.max() - noise_crop.min())
        gmsd_noise = gmsd(crop_image, noise_crop)
        pnsr_noise = compare_psnr(crop_image, noise_crop)
        info_noise = "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\r".format(mse_noise, mdsi_noise, ssim_noise, gmsd_noise, pnsr_noise)
        # print(info_noise)
        noise_score.append(info_noise)

        # calculate the score for image with ring artifact
        ring_radius = 5
        for i in range(ring_number):
            rr1, cc1 = draw.circle(crop_size // 2, crop_size // 2, radius=ring_radius, shape=(crop_size,crop_size))
            rr2, cc2 = draw.circle(crop_size // 2, crop_size // 2, radius=ring_radius+ring_width, shape=(crop_size,crop_size))
            ring_radius+=random.randint(3,10)
            if ring_radius>100:
                break
            ring_crop = crop_image.copy()/255
            ring_result=crop_image.copy()/255
            ring_crop[rr1,cc1]=0
            temp=i
            if temp%2==1:
                ring_result[rr2,cc2] +=ring_artifact[j]*ring_crop[rr2,cc2]
            else:
                ring_result[rr2, cc2] -= ring_artifact[j] * ring_crop[rr2, cc2]
        ring_result[ring_result>1]=1
        ring_result[ring_result<0]=0
        ring_result=ring_result * 255
        mse_ring = compare_mse(crop_image, ring_result)
        mdsi_ring = gray_mdsi(crop_image, ring_result, f=0, alpha=0.6)
        ssim_ring = compare_ssim(crop_image, ring_result,dynamic_range=ring_crop.max() - ring_crop.min())
        gmsd_ring = gmsd(crop_image, ring_result)
        psnr_ring = compare_psnr(crop_image, ring_result)
        info_ring="{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\r".format(mse_ring, mdsi_ring, ssim_ring, gmsd_ring, psnr_ring)
#         print(info_ring)
        ring_score.append(info_ring)

    return blur_score, noise_score, ring_score


@ex.capture
def rescale_scores(score_list,ori_score):
    final_score_list=[]
    mse_scores=[]
    mdsi_scores=[]
    ssim_scores=[]
    gmsd_scores=[]
    psnr_scores=[]
    for score_line in score_list:
        items=score_line.strip().split(",")
        mse_scores.append(float(items[0]))
        mdsi_scores.append(float(items[1]))
        ssim_scores.append(float(items[2]))
        gmsd_scores.append(float(items[3]))
        psnr_scores.append(float(items[4]))
    mse_sec_max=mse_scores[4] # set the second max mse score
    mdsi_max=mdsi_scores[4]
    gmsd_max=gmsd_scores[4]
    psnr_max=psnr_scores[0]
    for i in range(distortion_level):
        mse_scores[i]=max(1,round((1-min(1,mse_scores[i]/mse_sec_max))*ori_score))
        mdsi_scores[i]=max(1,round(max(0,1-mdsi_scores[i]/mdsi_max)*ori_score))
        ssim_scores[i]=max(1,round(ssim_scores[i]*ori_score))
        gmsd_scores[i]=max(1,round(max(0,1-gmsd_scores[i]/gmsd_max)*ori_score))
        psnr_scores[i]=max(1,round(psnr_scores[i]/psnr_max*ori_score))
        final_score_list.append((mse_scores[i]+mdsi_scores[i]+ssim_scores[i]+gmsd_scores[i]+psnr_scores[i])/5)
        
    return final_score_list


@ex.main
def main():
    # load the annotated scores of origin images
    with open(config_dir+"annotations.json","r") as json_file:
        annotated_score=json.load(json_file)
    
    generated_score_dict={}

    with open(config_dir+"train_info.txt","r") as f:
        lines=f.readlines()

    # generate the scores for the distorted images
    for line in lines:
        line=line.strip("\n")
        image_name=line.split(",")[0]
        if not image_name.endswith("_0_0_0.bmp"):# only keep the original image
            continue    
        if not annotated_score.__contains__(image_name):
            continue
        image=cv2.imread(save_distorted_dir+image_name,0)
        # print(image_name)
        # make sure the image is at crop_size*crop_size
        assert(image.shape[0]==crop_size)
        # score projection
        blur_list, noise_list,ring_list=calculate_score(image)
        new_blur_list=rescale_scores(blur_list,annotated_score[image_name])
        new_noise_list=rescale_scores(noise_list,annotated_score[image_name])
        new_ring_list=rescale_scores(ring_list,annotated_score[image_name])
        # assure each image is assessed by 5 observers
        assert(len(new_ring_list)==5)
        
        temp_str="{}_0_0_0.bmp".format(image_name[:3])
        generated_score_dict.setdefault(temp_str,annotated_score[image_name])
        
        for ind,score in enumerate(new_blur_list):
            temp_str="{}_{}_0_0.bmp".format(image_name[:3],ind%5+1)
            generated_score_dict.setdefault(temp_str,score)
        for ind,score in enumerate(new_noise_list):
            temp_str="{}_0_{}_0.bmp".format(image_name[:3],ind%5+1)
            generated_score_dict.setdefault(temp_str,score)
        for ind,score in enumerate(new_ring_list):
            temp_str="{}_0_0_{}.bmp".format(image_name[:3],ind%5+1)
            generated_score_dict.setdefault(temp_str,score)

    # save the generated scores that are used in finetune 
    with open(config_dir+"train_finetune.txt","w") as f:
        for key in generated_score_dict.keys():
            f.write("{},{}\n".format(key, generated_score_dict[key]))
    print("done!")