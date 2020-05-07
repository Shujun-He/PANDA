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

#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')

#hyperparameters
epochs=70
lr=1e-4
lr_decay=0.3
lr_decay_period=30
weight_decay=5e-5
save_freq=1
batch_size=32
checkpoints_folder='checkpoints'
reload_tensor_epoch=-1
reload_tensor_path="/mnt/sda1/Kaggle/PANDA/data/processed_data/512x512.p"
reload_batch_size=16
cutout_ratio=0.5
num_classes=[6,4,4,]
label_weights=[1,0,0]
label_smoothing=0.05

fold=0
top=1

#dataset
tensor_path="/mnt/sda1/Kaggle/PANDA/data/processed_data/concat_n16.p"
csv_path='/mnt/sda1/Kaggle/PANDA/data/train_4fold.csv'



dataset=PANDADataset(tensor_path,csv_path,fold=fold,batch_size=batch_size)
checkpoints_folder='checkpoints_fold{}'.format(fold)
csv_file='log_fold{}.csv'.format(fold)

model = Network(num_classes,'efficientnet-b1',dropout_p=0.4).to(device)
model=nn.DataParallel(model)

history=pd.read_csv(csv_file,header=None)
scores=np.asarray(history)[:,1]
top_epochs=scores.argsort()[-top:][::-1]

top_checkponints=['best_weights/fold{}top{}.ckpt'.format(fold,i+1)
for i in range(top)]

mean=torch.Tensor(dataset.data_mean).to(device).reshape(1,1,3,1,1)
std=torch.Tensor(dataset.data_std).to(device).reshape(1,1,3,1,1)

logits=[]
for checkpoint in top_checkponints:
    model.load_state_dict(torch.load(checkpoint))
    logits.append(predict(model,device,dataset,mean,std,batch_size=128))
    break

# model.load_state_dict(torch.load(top_checkponints[2]))
# logits.append(predict(model,device,dataset,mean,std,batch_size=128))

if len(logits)==1:
    predictions=np.argmax(np.asarray(logits).squeeze(),1)
else:
    predictions=np.argmax(np.mean(np.asarray(logits),0),1)

ground_truths=dataset.isup_grades[dataset.val_indices]

from sklearn import metrics

kappa_score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')

top_scores=scores[top_epochs]
print('Checkpoint avg score: {} Top epoch scores: {}'.format(kappa_score,top_scores))

to_plot=[kappa_score]
ticks=['Checkpoint avg']
x = np.arange(len(top_scores)+1)
count=0
for score in top_scores:
    count+=1
    to_plot.append(score)
    ticks.append('Top{}'.format(count))

plt.bar(x,to_plot)
plt.xticks(x, ticks)
plt.savefig('fold{}.png'.format(fold))
