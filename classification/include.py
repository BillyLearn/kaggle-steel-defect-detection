import pandas as pd
import cv2
import numpy as np
import copy
import os
import glob
import random
from timeit import default_timer as timer

import torch

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

from torch.nn.utils.rnn import *

PI = np.pi

## 数据集目录
DATA_DIR = '../input/severstal-steel-defect-detection'

## .npy 目录
TRAIN_NPY = '../input/split/'

## 加载的resnet34 预训练模型
PRETRAIN_FILE = \
    '../input/pretrain_model/resnet34-333f7ec4.pth'

## 保存的训练文件模型
RESNET34_MODEL_PATH = '../input/resnet34_train_model'

## 文件目录
#INITIAL_CHECKPOINT = None
INITIAL_CHECKPOINT = '../input/resnet34_train_model/00106500_model.pth'
