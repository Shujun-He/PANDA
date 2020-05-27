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


def get_best_weights_from_fold(fold,top=3):
    csv_file='log_fold{}.csv'.format(fold)

    history=pd.read_csv(csv_file,header=None)
    #print(history[0])
    scores=np.asarray(history)[:,-2]
    top_epochs=scores.argsort()[-3:][::-1]
    print(scores[top_epochs])
    os.system('mkdir best_weights')

    logits=[]
    for i in range(top):
        weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,history[0][top_epochs[i]])
        print(weights_path)
        os.system('cp {} best_weights/fold{}top{}.ckpt'.format(weights_path,fold,i+1))


for i in range(2,3):
    get_best_weights_from_fold(i)


# fold=0
# top=5
#
#
# csv_file='log_fold{}.csv'.format(fold)
#
# history=pd.read_csv(csv_file,header=None)
# scores=np.asarray(history)[:,1]
# top_epochs=scores.argsort()[-top:][::-1]
#
# os.system('mkdir best_weights')
#
# logits=[]
# for i in range(top):
#     weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,top_epochs[i]+1)
#     os.system('cp {} best_weights/fold{}top{}.ckpt'.format(weights_path,fold,i+1))
