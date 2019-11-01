import pdb
import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.sampler import *
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from timeit import default_timer as timer



from efficientnet import *
import warnings
warnings.filterwarnings('ignore')


## classification path set
sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"
CHECKPOINT = '../input/resnet34_train_model/00190500_model.pth'


## segmentation path set
DATA_DIR    = '../input/severstal-steel-defect-detection'
CHECKPOINT_FILE = '../input/efficientnet_b5_train_model/00037000_model.pth'

###############################################################################
## segmentation
#################################################################################
class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group,out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral

def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x


class Net(nn.Module):

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = EfficientNetB5(drop_connect_rate)
        self.stem   = e.stem
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        self.block5 = e.block5
        self.block6 = e.block6
        self.block7 = e.block7
        self.last   = e.last
        e = None  #dropped

        #---
        self.lateral0 = nn.Conv2d(2048, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d( 176, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(  64, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d(  40, 64,  kernel_size=1, padding=0, stride=1)

        self.top1 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d( 64, 64),
        )
        self.top4 = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit_mask = nn.Conv2d(64,num_class+1,kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        x = self.stem(x)            #; print('stem  ',x.shape)
        x = self.block1(x)    ;x0=x #; print('block1',x.shape)
        x = self.block2(x)    ;x1=x #; print('block2',x.shape)
        x = self.block3(x)    ;x2=x #; print('block3',x.shape)
        x = self.block4(x)          #; print('block4',x.shape)
        x = self.block5(x)    ;x3=x #; print('block5',x.shape)
        x = self.block6(x)          #; print('block6',x.shape)
        x = self.block7(x)          #; print('block7',x.shape)
        x = self.last(x)      ;x4=x #; print('last  ',x.shape)

        # segment
        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #128x128
        t2 = self.top2(t2) #128x128
        t3 = self.top3(t3) #128x128

        t = torch.cat([t1,t2,t3],1)
        t = self.top4(t)
        logit_mask = self.logit_mask(t)
        logit_mask = F.interpolate(logit_mask, scale_factor=2.0, mode='bilinear', align_corners=False)

        return logit_mask


def probability_mask_to_probability_label(probability):
    batch_size,num_class,H,W = probability.shape
    probability = probability.permute(0, 2, 3, 1).contiguous().view(batch_size,-1, 5)
    value, index = probability.max(1)
    probability = value
    return probability

def image_to_input(image):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    return input

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

class KaggleTestDataset(Dataset):
    def __init__(self):

        df =  pd.read_csv(DATA_DIR + '/sample_submission.csv')
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.uid = df['ImageId'].unique().tolist()

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        return string

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        image_id = self.uid[index]
        image = cv2.imread(DATA_DIR + '/test_images/%s'%(image_id), cv2.IMREAD_COLOR)
        return image, image_id


def null_collate(batch):
    batch_size = len(batch)
    input = []
    image_id = []
    for b in range(batch_size):
        input.append(batch[b][0])
        image_id.append(batch[b][1])
    input = np.stack(input)
    input = torch.from_numpy(image_to_input(input))
    return input, image_id

def run_length_encode(mask):
    m = mask.T.flatten()
    if m.sum()==0:
        rle=''
    else:
        m   = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
    return rle

threshold_label      = [ 0.75, 0.85, 0.2, 0.75,]
threshold_mask_pixel = [ 0.40, 0.40, 0.40, 0.40,]
threshold_mask_size  = [    1,    1,    1,    1,]


net = Net().cuda()
net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage))


dataset = KaggleTestDataset()

loader  = DataLoader(
    dataset,
    sampler     = SequentialSampler(dataset),
    batch_size  = 4,
    drop_last   = False,
    num_workers = 0,
    pin_memory  = True,
    collate_fn  = null_collate
)

image_id_class_id = []
encoded_pixel     = []

net.eval()

start_timer = timer()
for t,(input, image_id) in enumerate(loader):
    if t%200==0:
        print('\r loader: t = %4d / %4d  %s  %s : %s'%(
            t, len(loader)-1, str(input.shape), image_id[0], time_to_str((timer() - start_timer),'sec'),
        ),end='', flush=True)

    with torch.no_grad():
        input = input.cuda()

        logit = net(input) #data_parallel(net,input)  #net(input)
        probability = torch.softmax(logit,1)

        probability_mask  = probability[:,1:]#just drop background
        probability_label = probability_mask_to_probability_label(probability)[:,1:]


    probability_mask  = probability_mask.data.cpu().numpy()
    probability_label = probability_label.data.cpu().numpy()

    batch_size = len(image_id)
    for b in range(batch_size):
        for c in range(4):
            rle=''

            predict_label = probability_label[b,c]>threshold_label[c]
            if predict_label:
                try:
                    predict_mask = probability_mask[b,c] > threshold_mask_pixel[c]
                    rle = run_length_encode(predict_mask)

                except:
                    print('An exception occurred : %s'%(image_id[b]+'_%d'%(c+1)))


            image_id_class_id.append(image_id[b]+'_%d'%(c+1))
            encoded_pixel.append(rle)

df_mask = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])

###############################################################################
# classification
###############################################################################
BatchNorm2d = nn.BatchNorm2d

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

###############################################################################
CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'conv1.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'bn1.weight',	(64,),
 'block0.1.bias',	(64,),	 'bn1.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'bn1.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'bn1.running_var',	(64,),
 'block1.1.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv1.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn1.bn.weight',	(64,),	 'layer1.0.bn1.weight',	(64,),
 'block1.1.conv_bn1.bn.bias',	(64,),	 'layer1.0.bn1.bias',	(64,),
 'block1.1.conv_bn1.bn.running_mean',	(64,),	 'layer1.0.bn1.running_mean',	(64,),
 'block1.1.conv_bn1.bn.running_var',	(64,),	 'layer1.0.bn1.running_var',	(64,),
 'block1.1.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv2.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn2.bn.weight',	(64,),	 'layer1.0.bn2.weight',	(64,),
 'block1.1.conv_bn2.bn.bias',	(64,),	 'layer1.0.bn2.bias',	(64,),
 'block1.1.conv_bn2.bn.running_mean',	(64,),	 'layer1.0.bn2.running_mean',	(64,),
 'block1.1.conv_bn2.bn.running_var',	(64,),	 'layer1.0.bn2.running_var',	(64,),
 'block1.2.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv1.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn1.bn.weight',	(64,),	 'layer1.1.bn1.weight',	(64,),
 'block1.2.conv_bn1.bn.bias',	(64,),	 'layer1.1.bn1.bias',	(64,),
 'block1.2.conv_bn1.bn.running_mean',	(64,),	 'layer1.1.bn1.running_mean',	(64,),
 'block1.2.conv_bn1.bn.running_var',	(64,),	 'layer1.1.bn1.running_var',	(64,),
 'block1.2.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv2.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn2.bn.weight',	(64,),	 'layer1.1.bn2.weight',	(64,),
 'block1.2.conv_bn2.bn.bias',	(64,),	 'layer1.1.bn2.bias',	(64,),
 'block1.2.conv_bn2.bn.running_mean',	(64,),	 'layer1.1.bn2.running_mean',	(64,),
 'block1.2.conv_bn2.bn.running_var',	(64,),	 'layer1.1.bn2.running_var',	(64,),
 'block1.3.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv1.weight',	(64, 64, 3, 3),
 'block1.3.conv_bn1.bn.weight',	(64,),	 'layer1.2.bn1.weight',	(64,),
 'block1.3.conv_bn1.bn.bias',	(64,),	 'layer1.2.bn1.bias',	(64,),
 'block1.3.conv_bn1.bn.running_mean',	(64,),	 'layer1.2.bn1.running_mean',	(64,),
 'block1.3.conv_bn1.bn.running_var',	(64,),	 'layer1.2.bn1.running_var',	(64,),
 'block1.3.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv2.weight',	(64, 64, 3, 3),
 'block1.3.conv_bn2.bn.weight',	(64,),	 'layer1.2.bn2.weight',	(64,),
 'block1.3.conv_bn2.bn.bias',	(64,),	 'layer1.2.bn2.bias',	(64,),
 'block1.3.conv_bn2.bn.running_mean',	(64,),	 'layer1.2.bn2.running_mean',	(64,),
 'block1.3.conv_bn2.bn.running_var',	(64,),	 'layer1.2.bn2.running_var',	(64,),
 'block2.0.conv_bn1.conv.weight',	(128, 64, 3, 3),	 'layer2.0.conv1.weight',	(128, 64, 3, 3),
 'block2.0.conv_bn1.bn.weight',	(128,),	 'layer2.0.bn1.weight',	(128,),
 'block2.0.conv_bn1.bn.bias',	(128,),	 'layer2.0.bn1.bias',	(128,),
 'block2.0.conv_bn1.bn.running_mean',	(128,),	 'layer2.0.bn1.running_mean',	(128,),
 'block2.0.conv_bn1.bn.running_var',	(128,),	 'layer2.0.bn1.running_var',	(128,),
 'block2.0.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.0.conv2.weight',	(128, 128, 3, 3),
 'block2.0.conv_bn2.bn.weight',	(128,),	 'layer2.0.bn2.weight',	(128,),
 'block2.0.conv_bn2.bn.bias',	(128,),	 'layer2.0.bn2.bias',	(128,),
 'block2.0.conv_bn2.bn.running_mean',	(128,),	 'layer2.0.bn2.running_mean',	(128,),
 'block2.0.conv_bn2.bn.running_var',	(128,),	 'layer2.0.bn2.running_var',	(128,),
 'block2.0.shortcut.conv.weight',	(128, 64, 1, 1),	 'layer2.0.downsample.0.weight',	(128, 64, 1, 1),
 'block2.0.shortcut.bn.weight',	(128,),	 'layer2.0.downsample.1.weight',	(128,),
 'block2.0.shortcut.bn.bias',	(128,),	 'layer2.0.downsample.1.bias',	(128,),
 'block2.0.shortcut.bn.running_mean',	(128,),	 'layer2.0.downsample.1.running_mean',	(128,),
 'block2.0.shortcut.bn.running_var',	(128,),	 'layer2.0.downsample.1.running_var',	(128,),
 'block2.1.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv1.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn1.bn.weight',	(128,),	 'layer2.1.bn1.weight',	(128,),
 'block2.1.conv_bn1.bn.bias',	(128,),	 'layer2.1.bn1.bias',	(128,),
 'block2.1.conv_bn1.bn.running_mean',	(128,),	 'layer2.1.bn1.running_mean',	(128,),
 'block2.1.conv_bn1.bn.running_var',	(128,),	 'layer2.1.bn1.running_var',	(128,),
 'block2.1.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv2.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn2.bn.weight',	(128,),	 'layer2.1.bn2.weight',	(128,),
 'block2.1.conv_bn2.bn.bias',	(128,),	 'layer2.1.bn2.bias',	(128,),
 'block2.1.conv_bn2.bn.running_mean',	(128,),	 'layer2.1.bn2.running_mean',	(128,),
 'block2.1.conv_bn2.bn.running_var',	(128,),	 'layer2.1.bn2.running_var',	(128,),
 'block2.2.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv1.weight',	(128, 128, 3, 3),
 'block2.2.conv_bn1.bn.weight',	(128,),	 'layer2.2.bn1.weight',	(128,),
 'block2.2.conv_bn1.bn.bias',	(128,),	 'layer2.2.bn1.bias',	(128,),
 'block2.2.conv_bn1.bn.running_mean',	(128,),	 'layer2.2.bn1.running_mean',	(128,),
 'block2.2.conv_bn1.bn.running_var',	(128,),	 'layer2.2.bn1.running_var',	(128,),
 'block2.2.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv2.weight',	(128, 128, 3, 3),
 'block2.2.conv_bn2.bn.weight',	(128,),	 'layer2.2.bn2.weight',	(128,),
 'block2.2.conv_bn2.bn.bias',	(128,),	 'layer2.2.bn2.bias',	(128,),
 'block2.2.conv_bn2.bn.running_mean',	(128,),	 'layer2.2.bn2.running_mean',	(128,),
 'block2.2.conv_bn2.bn.running_var',	(128,),	 'layer2.2.bn2.running_var',	(128,),
 'block2.3.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv1.weight',	(128, 128, 3, 3),
 'block2.3.conv_bn1.bn.weight',	(128,),	 'layer2.3.bn1.weight',	(128,),
 'block2.3.conv_bn1.bn.bias',	(128,),	 'layer2.3.bn1.bias',	(128,),
 'block2.3.conv_bn1.bn.running_mean',	(128,),	 'layer2.3.bn1.running_mean',	(128,),
 'block2.3.conv_bn1.bn.running_var',	(128,),	 'layer2.3.bn1.running_var',	(128,),
 'block2.3.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv2.weight',	(128, 128, 3, 3),
 'block2.3.conv_bn2.bn.weight',	(128,),	 'layer2.3.bn2.weight',	(128,),
 'block2.3.conv_bn2.bn.bias',	(128,),	 'layer2.3.bn2.bias',	(128,),
 'block2.3.conv_bn2.bn.running_mean',	(128,),	 'layer2.3.bn2.running_mean',	(128,),
 'block2.3.conv_bn2.bn.running_var',	(128,),	 'layer2.3.bn2.running_var',	(128,),
 'block3.0.conv_bn1.conv.weight',	(256, 128, 3, 3),	 'layer3.0.conv1.weight',	(256, 128, 3, 3),
 'block3.0.conv_bn1.bn.weight',	(256,),	 'layer3.0.bn1.weight',	(256,),
 'block3.0.conv_bn1.bn.bias',	(256,),	 'layer3.0.bn1.bias',	(256,),
 'block3.0.conv_bn1.bn.running_mean',	(256,),	 'layer3.0.bn1.running_mean',	(256,),
 'block3.0.conv_bn1.bn.running_var',	(256,),	 'layer3.0.bn1.running_var',	(256,),
 'block3.0.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.0.conv2.weight',	(256, 256, 3, 3),
 'block3.0.conv_bn2.bn.weight',	(256,),	 'layer3.0.bn2.weight',	(256,),
 'block3.0.conv_bn2.bn.bias',	(256,),	 'layer3.0.bn2.bias',	(256,),
 'block3.0.conv_bn2.bn.running_mean',	(256,),	 'layer3.0.bn2.running_mean',	(256,),
 'block3.0.conv_bn2.bn.running_var',	(256,),	 'layer3.0.bn2.running_var',	(256,),
 'block3.0.shortcut.conv.weight',	(256, 128, 1, 1),	 'layer3.0.downsample.0.weight',	(256, 128, 1, 1),
 'block3.0.shortcut.bn.weight',	(256,),	 'layer3.0.downsample.1.weight',	(256,),
 'block3.0.shortcut.bn.bias',	(256,),	 'layer3.0.downsample.1.bias',	(256,),
 'block3.0.shortcut.bn.running_mean',	(256,),	 'layer3.0.downsample.1.running_mean',	(256,),
 'block3.0.shortcut.bn.running_var',	(256,),	 'layer3.0.downsample.1.running_var',	(256,),
 'block3.1.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv1.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn1.bn.weight',	(256,),	 'layer3.1.bn1.weight',	(256,),
 'block3.1.conv_bn1.bn.bias',	(256,),	 'layer3.1.bn1.bias',	(256,),
 'block3.1.conv_bn1.bn.running_mean',	(256,),	 'layer3.1.bn1.running_mean',	(256,),
 'block3.1.conv_bn1.bn.running_var',	(256,),	 'layer3.1.bn1.running_var',	(256,),
 'block3.1.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv2.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn2.bn.weight',	(256,),	 'layer3.1.bn2.weight',	(256,),
 'block3.1.conv_bn2.bn.bias',	(256,),	 'layer3.1.bn2.bias',	(256,),
 'block3.1.conv_bn2.bn.running_mean',	(256,),	 'layer3.1.bn2.running_mean',	(256,),
 'block3.1.conv_bn2.bn.running_var',	(256,),	 'layer3.1.bn2.running_var',	(256,),
 'block3.2.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv1.weight',	(256, 256, 3, 3),
 'block3.2.conv_bn1.bn.weight',	(256,),	 'layer3.2.bn1.weight',	(256,),
 'block3.2.conv_bn1.bn.bias',	(256,),	 'layer3.2.bn1.bias',	(256,),
 'block3.2.conv_bn1.bn.running_mean',	(256,),	 'layer3.2.bn1.running_mean',	(256,),
 'block3.2.conv_bn1.bn.running_var',	(256,),	 'layer3.2.bn1.running_var',	(256,),
 'block3.2.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv2.weight',	(256, 256, 3, 3),
 'block3.2.conv_bn2.bn.weight',	(256,),	 'layer3.2.bn2.weight',	(256,),
 'block3.2.conv_bn2.bn.bias',	(256,),	 'layer3.2.bn2.bias',	(256,),
 'block3.2.conv_bn2.bn.running_mean',	(256,),	 'layer3.2.bn2.running_mean',	(256,),
 'block3.2.conv_bn2.bn.running_var',	(256,),	 'layer3.2.bn2.running_var',	(256,),
 'block3.3.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv1.weight',	(256, 256, 3, 3),
 'block3.3.conv_bn1.bn.weight',	(256,),	 'layer3.3.bn1.weight',	(256,),
 'block3.3.conv_bn1.bn.bias',	(256,),	 'layer3.3.bn1.bias',	(256,),
 'block3.3.conv_bn1.bn.running_mean',	(256,),	 'layer3.3.bn1.running_mean',	(256,),
 'block3.3.conv_bn1.bn.running_var',	(256,),	 'layer3.3.bn1.running_var',	(256,),
 'block3.3.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv2.weight',	(256, 256, 3, 3),
 'block3.3.conv_bn2.bn.weight',	(256,),	 'layer3.3.bn2.weight',	(256,),
 'block3.3.conv_bn2.bn.bias',	(256,),	 'layer3.3.bn2.bias',	(256,),
 'block3.3.conv_bn2.bn.running_mean',	(256,),	 'layer3.3.bn2.running_mean',	(256,),
 'block3.3.conv_bn2.bn.running_var',	(256,),	 'layer3.3.bn2.running_var',	(256,),
 'block3.4.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv1.weight',	(256, 256, 3, 3),
 'block3.4.conv_bn1.bn.weight',	(256,),	 'layer3.4.bn1.weight',	(256,),
 'block3.4.conv_bn1.bn.bias',	(256,),	 'layer3.4.bn1.bias',	(256,),
 'block3.4.conv_bn1.bn.running_mean',	(256,),	 'layer3.4.bn1.running_mean',	(256,),
 'block3.4.conv_bn1.bn.running_var',	(256,),	 'layer3.4.bn1.running_var',	(256,),
 'block3.4.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv2.weight',	(256, 256, 3, 3),
 'block3.4.conv_bn2.bn.weight',	(256,),	 'layer3.4.bn2.weight',	(256,),
 'block3.4.conv_bn2.bn.bias',	(256,),	 'layer3.4.bn2.bias',	(256,),
 'block3.4.conv_bn2.bn.running_mean',	(256,),	 'layer3.4.bn2.running_mean',	(256,),
 'block3.4.conv_bn2.bn.running_var',	(256,),	 'layer3.4.bn2.running_var',	(256,),
 'block3.5.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv1.weight',	(256, 256, 3, 3),
 'block3.5.conv_bn1.bn.weight',	(256,),	 'layer3.5.bn1.weight',	(256,),
 'block3.5.conv_bn1.bn.bias',	(256,),	 'layer3.5.bn1.bias',	(256,),
 'block3.5.conv_bn1.bn.running_mean',	(256,),	 'layer3.5.bn1.running_mean',	(256,),
 'block3.5.conv_bn1.bn.running_var',	(256,),	 'layer3.5.bn1.running_var',	(256,),
 'block3.5.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv2.weight',	(256, 256, 3, 3),
 'block3.5.conv_bn2.bn.weight',	(256,),	 'layer3.5.bn2.weight',	(256,),
 'block3.5.conv_bn2.bn.bias',	(256,),	 'layer3.5.bn2.bias',	(256,),
 'block3.5.conv_bn2.bn.running_mean',	(256,),	 'layer3.5.bn2.running_mean',	(256,),
 'block3.5.conv_bn2.bn.running_var',	(256,),	 'layer3.5.bn2.running_var',	(256,),
 'block4.0.conv_bn1.conv.weight',	(512, 256, 3, 3),	 'layer4.0.conv1.weight',	(512, 256, 3, 3),
 'block4.0.conv_bn1.bn.weight',	(512,),	 'layer4.0.bn1.weight',	(512,),
 'block4.0.conv_bn1.bn.bias',	(512,),	 'layer4.0.bn1.bias',	(512,),
 'block4.0.conv_bn1.bn.running_mean',	(512,),	 'layer4.0.bn1.running_mean',	(512,),
 'block4.0.conv_bn1.bn.running_var',	(512,),	 'layer4.0.bn1.running_var',	(512,),
 'block4.0.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.0.conv2.weight',	(512, 512, 3, 3),
 'block4.0.conv_bn2.bn.weight',	(512,),	 'layer4.0.bn2.weight',	(512,),
 'block4.0.conv_bn2.bn.bias',	(512,),	 'layer4.0.bn2.bias',	(512,),
 'block4.0.conv_bn2.bn.running_mean',	(512,),	 'layer4.0.bn2.running_mean',	(512,),
 'block4.0.conv_bn2.bn.running_var',	(512,),	 'layer4.0.bn2.running_var',	(512,),
 'block4.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer4.0.downsample.0.weight',	(512, 256, 1, 1),
 'block4.0.shortcut.bn.weight',	(512,),	 'layer4.0.downsample.1.weight',	(512,),
 'block4.0.shortcut.bn.bias',	(512,),	 'layer4.0.downsample.1.bias',	(512,),
 'block4.0.shortcut.bn.running_mean',	(512,),	 'layer4.0.downsample.1.running_mean',	(512,),
 'block4.0.shortcut.bn.running_var',	(512,),	 'layer4.0.downsample.1.running_var',	(512,),
 'block4.1.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv1.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn1.bn.weight',	(512,),	 'layer4.1.bn1.weight',	(512,),
 'block4.1.conv_bn1.bn.bias',	(512,),	 'layer4.1.bn1.bias',	(512,),
 'block4.1.conv_bn1.bn.running_mean',	(512,),	 'layer4.1.bn1.running_mean',	(512,),
 'block4.1.conv_bn1.bn.running_var',	(512,),	 'layer4.1.bn1.running_var',	(512,),
 'block4.1.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv2.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn2.bn.weight',	(512,),	 'layer4.1.bn2.weight',	(512,),
 'block4.1.conv_bn2.bn.bias',	(512,),	 'layer4.1.bn2.bias',	(512,),
 'block4.1.conv_bn2.bn.running_mean',	(512,),	 'layer4.1.bn2.running_mean',	(512,),
 'block4.1.conv_bn2.bn.running_var',	(512,),	 'layer4.1.bn2.running_var',	(512,),
 'block4.2.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv1.weight',	(512, 512, 3, 3),
 'block4.2.conv_bn1.bn.weight',	(512,),	 'layer4.2.bn1.weight',	(512,),
 'block4.2.conv_bn1.bn.bias',	(512,),	 'layer4.2.bn1.bias',	(512,),
 'block4.2.conv_bn1.bn.running_mean',	(512,),	 'layer4.2.bn1.running_mean',	(512,),
 'block4.2.conv_bn1.bn.running_var',	(512,),	 'layer4.2.bn1.running_var',	(512,),
 'block4.2.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv2.weight',	(512, 512, 3, 3),
 'block4.2.conv_bn2.bn.weight',	(512,),	 'layer4.2.bn2.weight',	(512,),
 'block4.2.conv_bn2.bn.bias',	(512,),	 'layer4.2.bn2.bias',	(512,),
 'block4.2.conv_bn2.bn.running_mean',	(512,),	 'layer4.2.bn2.running_mean',	(512,),
 'block4.2.conv_bn2.bn.running_var',	(512,),	 'layer4.2.bn2.running_var',	(512,),
 'logit.weight',	(1000, 512),	 'fc.weight',	(1000, 512),
 'logit.bias',	(1000,),	 'fc.bias',	(1000,),

]

###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=3, padding=1, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z

class ResNet34(nn.Module):

    def __init__(self, num_class=1000 ):
        super(ResNet34, self).__init__()


        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1  = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),
          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.block2  = nn.Sequential(
             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),
          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,4)],
        )
        self.block3  = nn.Sequential(
             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),
          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,6)],
        )
        self.block4 = nn.Sequential(
             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),
          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.logit = nn.Linear(512,num_class)



    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit


class Resnet34_classification(nn.Module):
    def __init__(self,num_class=4):
        super(Resnet34_classification, self).__init__()
        e = ResNet34()
        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4,
        ])
        e = None  #dropped
        self.feature = nn.Conv2d(512,32, kernel_size=1) #dummy conv for dim reduction
        self.logit = nn.Conv2d(32,num_class, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

        x = F.dropout(x,0.5,training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
        return logit

model_classification = Resnet34_classification()
model_classification.load_state_dict(torch.load(CHECKPOINT, map_location=lambda storage, loc: storage), strict=True)

class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


# hyperparameters
batch_size = 1

# mean and std
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)

# dataloader
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

threshold_label = [0.50,0.50,0.50,0.50,]
augment = ['null']

def sharpen(p,t=0.5):
        if t!=0:
            return p**t
        else:
            return p

def get_classification_preds(net,test_loader):
    test_probability_label = []
    test_id   = []

    net = net.cuda()
    for t, (fnames, images) in enumerate(tqdm(test_loader)):
        batch_size,C,H,W = images.shape
        images = images.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                logit =  net(images)
                probability = torch.sigmoid(logit)

                probability_label = sharpen(probability,0)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = net(torch.flip(images,dims=[3]))
                probability  = torch.sigmoid(logit)

                probability_label += sharpen(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = net(torch.flip(images,dims=[2]))
                probability = torch.sigmoid(logit)

                probability_label += sharpen(probability)
                num_augment+=1

            probability_label = probability_label/num_augment

        probability_label = probability_label.data.cpu().numpy()

        test_probability_label.append(probability_label)
        test_id.extend([i for i in fnames])

    test_probability_label = np.concatenate(test_probability_label)
    return test_probability_label, test_id


# Get prediction for classification model
probability_label, image_id = get_classification_preds(model_classification, testset)
predict_label = probability_label>np.array(threshold_label).reshape(1,4,1,1)

image_id_class_id = []
encoded_pixel = []
for b in range(len(image_id)):
    for c in range(4):
        image_id_class_id.append(image_id[b]+'_%d'%(c+1))
        if predict_label[b,c]==0:
            rle=''
        else:
            rle ='1 1'
        encoded_pixel.append(rle)

df_classification = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])

df_label = df_classification.copy()

################################################################
# submit
################################################################

assert(np.all(df_mask['ImageId_ClassId'].values == df_label['ImageId_ClassId'].values))
df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels']=''

df_mask.to_csv("submission.csv", index=False)


df = df_mask.copy()
text = ''
df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
num_image = len(df)//4
num = len(df)

pos = (df['Label']==1).sum()
neg = num-pos


pos1 = ((df['Class']==1) & (df['Label']==1)).sum()
pos2 = ((df['Class']==2) & (df['Label']==1)).sum()
pos3 = ((df['Class']==3) & (df['Label']==1)).sum()
pos4 = ((df['Class']==4) & (df['Label']==1)).sum()

neg1 = num_image-pos1
neg2 = num_image-pos2
neg3 = num_image-pos3
neg4 = num_image-pos4



text += 'compare with LB probing ... \n'
text += '\t\tnum_image = %5d(1801) \n'%num_image
text += '\t\tnum  = %5d(7204) \n'%num
text += '\n'

text += '\t\tpos1 = %5d( 128)  %0.3f\n'%(pos1,pos1/128)
text += '\t\tpos2 = %5d(  43)  %0.3f\n'%(pos2,pos2/43)
text += '\t\tpos3 = %5d( 741)  %0.3f\n'%(pos3,pos3/741)
text += '\t\tpos4 = %5d( 120)  %0.3f\n'%(pos4,pos4/120)
text += '\n'

text += '\t\tneg1 = %5d(1673)  %0.3f  %3d\n'%(neg1,neg1/1673, neg1-1673)
text += '\t\tneg2 = %5d(1758)  %0.3f  %3d\n'%(neg2,neg2/1758, neg2-1758)
text += '\t\tneg3 = %5d(1060)  %0.3f  %3d\n'%(neg3,neg3/1060, neg3-1060)
text += '\t\tneg4 = %5d(1681)  %0.3f  %3d\n'%(neg4,neg4/1681, neg4-1681)
text += '--------------------------------------------------\n'
text += '\t\tneg  = %5d(6172)  %0.3f  %3d \n'%(neg,neg/6172, neg-6172)
text += '\n'
print(text)