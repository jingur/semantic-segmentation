"""
This model defined based on the something else
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import numpy as np


def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class VGG16_FCN32(nn.Module):
    
    def __init__(self, args):
        super(VGG16_FCN32, self).__init__()  
        num_classes = 7
        self.n_classes = num_classes
        pretrained = True
        self.conv1_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),        
            nn.ReLU(inplace=True),                   
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True), 
        )


        self.conv2_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv3_block = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv4_block = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv5_block = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        if pretrained:
            self.init_vgg16()

    def init_vgg16(self):
        vgg16 = models.vgg16(pretrained=True)           
        vgg16_features = list(vgg16.features.children())

        conv_blocks = [self.conv1_block, self.conv2_block, self.conv3_block, self.conv4_block, self.conv5_block]
        conv_ids_vgg = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 30]]  

        for conv_block_id, conv_block in enumerate(conv_blocks):
            conv_id_vgg = conv_ids_vgg[conv_block_id]
            for l1, l2 in zip(conv_block, vgg16_features[conv_id_vgg[0]:conv_id_vgg[1]]):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()

                    l1.weight.data = l2.weight.data
                    l1.bias.data = l2.bias.data

        vgg16_classifier = list(vgg16.classifier.children())
        for l1, l2 in zip(self.classifier, vgg16_classifier[0:3]):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
                l1.weight.data = l2.weight.data.view(l1.weight.size())
                l1.bias.data = l2.bias.data.view(l1.bias.size())

        l1 = self.classifier[6]
        l2 = vgg16_classifier[6]
        if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
            l1.weight.data = l2.weight.data[:self.n_classes, :].view(l1.weight.size())
            l1.bias.data = l2.bias.data[:self.n_classes].view(l1.bias.size())

    def forward(self, x):
        '''

        :param x: (1, 3, 360, 480)
        :return:
        '''
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        conv3 = self.conv3_block(conv2)
        conv4 = self.conv4_block(conv3)
        conv5 = self.conv5_block(conv4)
        score = self.classifier(conv5)

        out = F.interpolate(score, x.size()[2:], mode='bilinear', align_corners=True)
        return out
    
        
        
       
class Res34_FCN8(nn.Module):

    def __init__(self, args):
        super(Res34_FCN8, self).__init__()  

        '''Create the baseline model'''
        # define vgg16 with imagenet weights
        num_classes = 7
        # img -> (Nx3x352x448)
        self.resnet34 = models.resnet34(pretrained = True)
        self.stage1 = nn.Sequential(*list(self.resnet34.children())[:-4]) 
        self.stage2 = list(self.resnet34.children())[-4] 
        self.stage3 = list(self.resnet34.children())[-3] 
        
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) # 使用双线性 kernel
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel

        
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s2)
        return s
        
       





