import pandas as pd
import cv2
import numpy as np
import copy
import os
import glob
import random
from timeit import default_timer as timer

import torch

from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

from torch.nn.utils.rnn import *

## 数据集目录
DATA_DIR = '../input/severstal-steel-defect-detection'

## .npy 目录
TRAIN_NPY = '../input/split/'

## 加载的efficientnetB5 预训练模型
PRETRAIN_FILE = '../input/pretrain_model/efficientnet-b5-b6417697.pth'

# ## 保存的训练文件模型
EFFICIENTNET_B5_MODEL_PATH = '../input/efficientnet_b5_train_model'


## 文件目录
# INITIAL_CHECKPOINT = None
INITIAL_CHECKPOINT = '../input/efficientnet_b5_train_model/00006000_model.pth'

PI = np.pi