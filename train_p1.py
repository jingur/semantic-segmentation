import os
import numpy as np
from PIL import Image
from scipy import misc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import time

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


train_data_path = './hw2_data/p1_data/train_50'
validation_data_path = './hw2_data/p1_data/val_50'
print("Read data")
train_x, train_y = read_image(train_data_path, label=True)
print("Train data size: {}".format(len(train_y)))
val_x, val_y = read_image(validation_data_path, label=True)
print("Validation data size: {}".format(len(val_y)))

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(10),  
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ToTensor(), 
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ToTensor(),  
])


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


batch_size = 128
train_set = ImgDataset(train_x, train_y, transforms=train_transform)
val_set = ImgDataset(val_x, val_y, transforms=val_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# create the model

model = models.vgg16_bn(pretrained=True)
model.classifier._modules['6'] = nn.Linear(4096, 50)
model = model.cuda()
print(model)
torch.cuda.manual_seed_all(1234)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
print("Start Train")
num_epoch = 20
best_val_acc = 0
for epoch in range(num_epoch):
    model.train()
    epoch_start_time = time.time()
    train_acc = 0
    train_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        # print(data[0].shape)
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(),
                                      axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    val_loss = 0
    val_acc = 0
    model.eval()
    for i, data in enumerate(val_loader):
        pred = model(data[0].cuda())
        batch_loss = loss(pred, data[1].cuda())

        val_acc += np.sum(np.argmax(pred.cpu().data.numpy(),
                                    axis=1) == data[1].numpy())
        val_loss += batch_loss.item()

    epoch_end_time = time.time()
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f train_loss: %3.6f | Val Acc: %3.6f val_loss: %3.6f' %
          (epoch + 1, num_epoch, epoch_end_time-epoch_start_time,
           train_acc/train_set.__len__(), train_loss,
           val_acc/val_set.__len__(), val_loss))

    if best_val_acc < val_acc / val_set.__len__():
        print('Save model!')
        best_val_acc = val_acc / val_set.__len__()
        torch.save(model.state_dict(), 'model.pth')

