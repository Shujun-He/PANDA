#from efficientnet_pytorch import EfficientNet
import numpy as np
import os
import torch
import torch.nn as nn
import time
try:
    #from apex.parallel import DistributedDataParallel as DDP
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
tensor_path="../pickled_datas256n36"
csv_path='../train_4fold.csv'



checkpoints_folder='checkpoints_fold{}'.format(fold)
csv_file='log_fold{}.csv'.format(fold)
opt_level = 'O1'


# history=pd.read_csv(csv_file,header=None)
# scores=np.asarray(history)[:,1]
# top_epochs=scores.argsort()[-top:][::-1]

top_checkponints=['best_weights/fold{}top1.ckpt'.format(i)
for i in range(fold)]

i=0
batch_size=6
random_augment=None
lr=5e-6
confidence=0.2
epochs=5
predictions=[]
ground_truths=[]
raw=[]
fold=0
model = Network(num_classes,'efficientnet-b1',dropout_p=0.4).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model=nn.DataParallel(model)
model.load_state_dict(torch.load(top_checkponints[i]))
dataset=PANDADataset(tensor_path,csv_path,fold=i,batch_size=batch_size)
#validate(model,'s',device,dataset,epoch,batch_size=64)
batches=int(len(dataset.val_indices)/batch_size)+1

total=0
ground_truths=dataset.isup_grades[dataset.val_indices]
predictions=[]
raw_regression_outputs=[]
losses=0
criterion1=nn.SmoothL1Loss()
criterion2=nn.CrossEntropyLoss()
mae_criterion=nn.L1Loss(reduction='none')
#validate(model,'ttpl.csv',device,dataset,0,batch_size=batch_size*2)
for epoch in range(epochs):
    dataset.switch_mode(training=False)
    dataset.update_batchsize(batch_size)
    model.train(False)
    for i in tqdm(range(len(dataset))):
        data=dataset[i]
        batch_X=torch.Tensor(data['tensors']).to(device,)
        batch_Y=torch.Tensor(data['labels']).to(device,dtype=torch.int64)
        batch_X=standardize_batch(batch_X)
        outputs = model(batch_X)
        outputs_cpu=outputs[0].detach().cpu().numpy()
        pl=np.around(outputs_cpu)
        mae_loss=np.abs(outputs_cpu-pl)
        indices=mae_loss<confidence
        del outputs
        if np.sum(indices)>0:
            batch_X=batch_X[indices]
            pl=torch.Tensor(pl[indices]).to(device)
            outputs = model(batch_X)
            optimizer.zero_grad()
            loss=criterion1(outputs[0],pl)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                loss.backward()
            optimizer.step()
    validate(model,'ttpl.csv',device,dataset,epoch,batch_size=batch_size)
        
    save_weights(model,optimizer,epoch,'ttpl')
        