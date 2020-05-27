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
fold=4
#top=3

#dataset
tensor_path="/mnt/sda1/Kaggle/PANDA/data/pickled_datas256n36"
csv_path='/mnt/sda1/Kaggle/PANDA/data/train_4fold_clean.csv'



checkpoints_folder='checkpoints_fold{}'.format(fold)
csv_file='log_fold{}.csv'.format(fold)
opt_level = 'O1'
model = Network(num_classes,'efficientnet-b1',dropout_p=0.4).to(device)
optimizer = torch.optim.Adam(model.parameters())
#model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model=nn.DataParallel(model)

# history=pd.read_csv(csv_file,header=None)
# scores=np.asarray(history)[:,1]
# top_epochs=scores.argsort()[-top:][::-1]

top_checkponints=['best_weights/fold{}top1.ckpt'.format(i)
for i in range(fold)]

predictions=[]
ground_truths=[]
raw=[]
for i in range(4):
    model.load_state_dict(torch.load(top_checkponints[i]))
    model.eval()
    dataset=PANDADataset(tensor_path,csv_path,fold=i,batch_size=batch_size)
    #validate(model,'s',device,dataset,epoch,batch_size=64)
    raw_outputs,y_pred,y=predict(model,device,dataset,mean=0,std=0,batch_size=24)
    raw.append(raw_outputs)
    predictions.append(y_pred)
    ground_truths.append(y)

raw=np.concatenate(raw).reshape(-1)
predictions=np.concatenate(predictions).reshape(-1)
ground_truths=np.concatenate(ground_truths).reshape(-1)

score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')


with open('cv_scores.txt','w+') as f:
    f.write('cv qwk score is: {}'.format(score))


raw_outputs=[]
for number in raw:
    raw_outputs.append(number.cpu().numpy())
raw_outputs=np.asarray(raw_outputs)

pickle.dump([raw_outputs,ground_truths],open('pred_truth.p','wb+'))
