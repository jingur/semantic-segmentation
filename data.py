import os
import json
import torch
import scipy.misc
import string

import torch.nn as nn
import torchvision.transforms as transforms
import torch
import imageio
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from mask2label import read_masks

# defined by me
import torchvision.transforms.functional as TF
import random

# ImageNet mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]




class DATA(Dataset):


    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        # define the data
        self.data = []

        # read the data based on the mode
        if self.mode == 'train' or self.mode == 'validation':
            self.data_dir = args.input_dir


            train_img_path_list = sorted([file for file in os.listdir(self.data_dir) if file.endswith('_sat.jpg')])
            train_seg_path_list = sorted([file for file in os.listdir(self.data_dir) if file.endswith('_mask.png')])


            for i, train_img_path in enumerate(train_img_path_list):
                file_name = os.path.join(self.data_dir, train_img_path)
                self.data.append([file_name, os.path.join(self.data_dir, train_seg_path_list[i])])

        else:
            # testing
            train_img_path_list = sorted([file for file in os.listdir(args.input_dir) if file.endswith('_sat.jpg')])

            self.img_dir = args.input_dir

            for i, train_img_path in enumerate(train_img_path_list):
                file_name = os.path.join(self.img_dir, train_img_path)
                self.data.append([file_name, None])

        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])



    def my_segmentation_transforms(self, image, segmentation):
        if random.random() > 0.5:
            # flip the image
            image = TF.hflip(image)
            segmentation = TF.hflip(segmentation)
        if random.random() > 0.25:
            # rotation
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
        if random.random() > 0.5:
            # rotation
            angle = random.randint(-10, 10)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
            
        # more transforms ...
        return image, segmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path, cls = self.data[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        seg_img = None # case of test

        if cls != None: # training or validation set
            #Load labels into numpy
            classified_img = read_masks(cls)
            gray_scale_image = Image.fromarray(np.uint8(classified_img) , 'L')
            #label_name = read_masks(self.data_dir, train_seg_path_list)
            seg_img = gray_scale_image
        
        if self.mode == 'train':
            img, seg_img = self.my_segmentation_transforms(img, seg_img)
       
        return self.transform(img), torch.Tensor(np.array(seg_img)).long()
       
