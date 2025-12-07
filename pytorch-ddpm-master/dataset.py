from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import random
import os
from os.path import join, splitext, basename
from glob import glob
import pandas as pd
import math
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop

def rand_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and random crop to target size

    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = random.randint(0, width - target_width)
        starting_y = random.randint(0, height - target_height)
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = random.randint(0, new_width - target_width)
        starting_y = random.randint(0, new_height - target_height)
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

## 2023 11 14 홍진성 사용
def center_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and center crop to target size
    
    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = (width - target_width) / 2
        starting_y = (height - target_height) / 2
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = (new_width - target_width) / 2
        starting_y = (new_height - target_height) / 2
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

class dataset_norm(Dataset):
    def __init__(self, root='', transforms=None, imglist=[]):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        #self.img_list = []
        self.transforms = transforms

        self.img_list = imglist

        self.size = len(self.img_list)

    def __getitem__(self, index):
        index = index % self.size

        img = Image.open(self.img_list[index]).convert("RGB")

        img = self.transforms(img)

        return img

    def __len__(self):
        return self.size


## 2023 11 14 홍진성 사용
class dataset_test4(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1, imglist=[]):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        self.inputsize2 = inputsize + 64 * (pred_step - 1)

        self.img_list = imglist

        # for name in file_list:
        #     img = Image.open(name)
        #     #if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
        #     if (img.size[0] >= 0) and (img.size[1] >= 0):
        #         self.img_list += [name]

        self.size = len(self.img_list)
    ## 128 x 128 real 이고 나머지는 마스킹 영역
    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')

        i = (self.imgSize - self.inputsize) // 2  # 32
        # j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img

        ## 2023 11 14 홍진성 마스킹 수정 (너비 방향으로만 처리)
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
        #mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i:i + self.inputsize2, i:i+self.inputsize2] = img
        else:
            mask_img[:, :, i:i + self.inputsize2] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0], name.replace('\\', '/').split('/')[-2]
        # return torch.rand((3,192,192)), torch.rand((3,192,192)), torch.rand((3,192,192)), '', ''

    def __len__(self):
        return self.size
        # return 10000
