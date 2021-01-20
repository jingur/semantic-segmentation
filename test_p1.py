import os
import numpy as np
from PIL import Image
from scipy import misc
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import parser

def read_image(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 64, 64, 3), dtype=np.uint8)
    y = np.zeros(len(image_dir), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        image = Image.open(os.path.join(path,file))
        image = image.resize((64, 64) , Image.ANTIALIAS)
        # image = misc.imread(os.path.join(path, file))
        x[i] = image
        if label:
            y[i] = int(file.split('_')[0])

    if label:
        return x, y
    else:
        return x

def read_file_name(path):
    image_dir = sorted(os.listdir(path))
    return image_dir

def validation_check(input_dir):
    seg_path_list = sorted([file for file in os.listdir(input_dir) if str(file)[1]=='_' or str(file)[2]=="_"])
    if len(seg_path_list)!=0:
        return True
    else:
        return False


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transforms=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transforms = transforms

    def __len__(self):
        return(len(self.x))

    def __getitem__(self, index):
        X = self.x[index]
        if self.transforms is not None:
            X = self.transforms(X)
        else:
            X = torch.Tensor(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X



if __name__ == '__main__':
    
    batch_size = 1
    args = parser.arg_parse()
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  
    ])
    # get input and output directory
    input_dir = args.input_dir
    output_dir = args. output_dir

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    if validation_check(input_dir):
    #load validation dataset
        print("Validation...")
        val_x, val_y = read_image(input_dir, label=True)
        val_set = ImgDataset(val_x, val_y, transforms=val_transform)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


    else:
        print("Testing...")
    #load test dataset    
        test_x = read_image(input_dir, label=False)
        test_set = ImgDataset(test_x, None, transforms=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



    ''' resume save model '''
    model = models.vgg16_bn(pretrained=True)
    model.classifier._modules['6'] = nn.Linear(4096, 50)
    model = model.cuda()
    #use resume to load models from dropbox thriugh bash .sh 
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    if validation_check(input_dir):
        val_loss = 0
        val_acc = 0
        file = []
        pred_label = []
        model.eval()
        for i, data in enumerate(val_loader):
            pred = model(data[0].cuda())

            val_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1) == data[1].numpy())
            file.append(read_file_name(input_dir)[i])
            pred_label.append(np.argmax(pred.cpu().data.numpy()))
        
        #pred_label = pred_label.flatten()
        csv_file_path = os.path.join(output_dir, 'test_pred.csv')
        dataframe = pd.DataFrame({'image_id':file,'label':pred_label})
        dataframe.to_csv(csv_file_path,index=False,sep=',') 
        print('Val Acc: %3.6f ' %(val_acc/val_set.__len__()))
    else:
        test_loss = 0
        test_acc = 0
        file = []
        pred_label = []
        model.eval()
        for i, data in enumerate(test_loader):
            pred = model(data.cuda())
            
            file.append(read_file_name(input_dir)[i])
            pred_label.append(np.argmax(pred.cpu().data.numpy()))
        #pred_label = pred_label.flatten()
        csv_file_path = os.path.join(output_dir, 'test_pred.csv')
        dataframe = pd.DataFrame({'image_id':file,'label':pred_label})
        dataframe.to_csv(csv_file_path,index=False,sep=',')            

