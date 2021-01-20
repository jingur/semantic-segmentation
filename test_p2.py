import os
import parser
import models
from PIL import Image
import data
import data_test 
import viz_mask

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score
import skimage
from skimage import io

from mean_iou_evaluate import mean_iou_score
from mean_iou_evaluate import read_masks
# import torch library
import torch


def prediction_labeller(idx):
    remain = '0' * (4-len(idx))
    idx = remain + idx
    return str(idx)

# change the hard coded path $2 -> output dir
# output_dir = "./preds/"

#check it's a validation set or test set
def validation_check(input_dir):
    seg_path_list = sorted([file for file in os.listdir(input_dir) if file.endswith('_mask.png')])
    if len(seg_path_list)!=0:
        return True
    else:
        return False

def evaluate(model, data_loader, mode = "train", output_dir = "./preds/"):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        cnt = 0
        for _, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            rgb_mask = np.empty((512,512,3), dtype=int)

            if mode == "validation" or mode == "test":
                # save images only during test and validation
                for p in pred:
                    p = torch.argmax(p.squeeze(), dim=0).detach().cpu().numpy()
                    for j in range (0, 512):
                        for i in range (0, 512):
                            if p[i][j] == 0:
                                rgb_mask[i][j] = (0, 255, 255)
                            elif p[i][j] == 1:
                                rgb_mask[i][j] = (255, 255, 0)
                            elif p[i][j] == 2:
                                rgb_mask[i][j] = (255, 0, 255)
                            elif p[i][j] == 3:
                                rgb_mask[i][j] = (0, 255, 0)
                            elif p[i][j] == 4:
                                rgb_mask[i][j] = (0, 0, 255)
                            elif p[i][j] == 5:
                                rgb_mask[i][j] = (255, 255, 255)
                            elif p[i][j] == 6:
                                rgb_mask[i][j] = (0, 0, 0)

                    p = Image.fromarray(np.uint8(rgb_mask) , 'RGB')
        
                    p.save(output_dir+"/"+prediction_labeller(str(cnt))+"_mask.png", "png")
                    #skimage.io.imsave(os.path.join(output_dir, prediction_labeller(str(cnt)) + "_mask.png"), p)
                    cnt += 1
                pass
            else:
                # no need to save during training
                pass

            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            preds.append(pred)
            gt = gt.numpy().squeeze()    
            gts.append(gt)
        gts = np.concatenate(gts)
        preds = np.concatenate(preds)
        return mean_iou_score(gts, preds)

def evaluate_test(model, data_loader, output_dir = "./preds/"):
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    

    with torch.no_grad(): # do not need to caculate information for gradient during eval
        cnt = 0
        for _, imgs in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            rgb_mask = np.empty((512,512,3), dtype=int)
            # save images only during test and validation
            for p in pred:
                q = torch.argmax(p.squeeze(), dim=0).detach().cpu().numpy()
                for j in range (0, 512):
                    for i in range (0, 512):
                        if q[i][j] == 0:
                            rgb_mask[i][j] = (0, 255, 255)
                        elif q[i][j] == 1:
                            rgb_mask[i][j] = (255, 255, 0)
                        elif q[i][j] == 2:
                            rgb_mask[i][j] = (255, 0, 255)
                        elif q[i][j] == 3:
                            rgb_mask[i][j] = (0, 255, 0)
                        elif q[i][j] == 4:
                            rgb_mask[i][j] = (0, 0, 255)
                        elif q[i][j] == 5:
                            rgb_mask[i][j] = (255, 255, 255)
                        elif q[i][j] == 6:
                            rgb_mask[i][j] = (0, 0, 0)

                    test_img = Image.fromarray(np.uint8(rgb_mask) , 'RGB')
        
                    test_img.save(output_dir+"/"+prediction_labeller(str(cnt))+"_mask.png", "png")
                #skimage.io.imsave(os.path.join(output_dir, prediction_labeller(str(cnt)) + "_mask.png"), p)               
                #skimage.io.imsave(os.path.join(output_dir, prediction_labeller(str(cnt)) + ".png"), p)
                cnt += 1
            pass

            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            
            preds.append(pred)
         
    preds = np.concatenate(preds)
    return 0



if __name__ == '__main__':

    args = parser.arg_parse()

    # get input and output directory
    input_dir = args.input_dir
    output_dir = args.output_dir

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    if validation_check(input_dir):
        print("Validation...")
        test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='validation'),
                                                  batch_size=args.test_batch,
                                                  num_workers=args.workers,
                                                  shuffle=False)
    else:
        print("Testing...")
        test_loader = torch.utils.data.DataLoader(data_test.DATA_TEST(args, mode='test'),
                                                  batch_size=args.test_batch,
                                                  num_workers=args.workers,
                                                  shuffle=False)
    ''' prepare mode '''
    #define model by "--model Baseline model" or "--model Improved model" in shell file

    model = models.VGG16_FCN32(args).cuda()
#    model = models.Res34_FCN8(args).cuda()

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    if validation_check(input_dir):
        acc = evaluate(model, test_loader, mode="validation", output_dir = args.output_dir)
        print('Testing Accuracy: {}'.format(acc))
    else:
        _ = evaluate_test(model, test_loader, output_dir = args.output_dir)
