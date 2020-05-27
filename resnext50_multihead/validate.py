from efficientnet_pytorch import EfficientNet
import numpy as np
import os
import torch
import torch.nn as nn
import time
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from DataAug import *
from Functions import *
from Dataset import *
import pandas as pd
from Network import *
import matplotlib.pyplot as plt
from sklearn import metrics


#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')

#hyperparameters
batch_size=32
checkpoints_folder='checkpoints'
num_classes=[6,4,4,]
fold=0
epoch=9
#dataset
tensor_path="/mnt/sda1/Kaggle/PANDA/data/pickled_datas256n24"
csv_path='/mnt/sda1/Kaggle/PANDA/data/train_4fold.csv'



checkpoints_folder='checkpoints_fold{}'.format(fold)
csv_file='log_fold{}.csv'.format(fold)

model = Network(num_classes,'efficientnet-b1',dropout_p=0.4).to(device)
model=nn.DataParallel(model)

# history=pd.read_csv(csv_file,header=None)
# scores=np.asarray(history)[:,1]
# top_epochs=scores.argsort()[-top:][::-1]

top_checkponints=['best_weights/fold{}top1.ckpt'.format(i)
for i in range(fold)]


checkpoint='/mnt/sda1/Kaggle/PANDA/layer1/test0_N24/checkpoints_fold{}/epoch{}.ckpt'.format(fold,epoch)
model.load_state_dict(torch.load(checkpoint))
dataset=PANDADataset(tensor_path,csv_path,fold=fold,batch_size=batch_size)
validate(model,'s',device,dataset,epoch=10,batch_size=32)
