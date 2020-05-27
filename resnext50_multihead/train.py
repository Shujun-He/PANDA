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

#dataset
from Dataset import *
tensor_path="/mnt/sda1/Kaggle/PANDA/data/processed_data/concat_n12.p"
csv_path='/mnt/sda1/Kaggle/PANDA/data/train.csv'

from fold import *

dataset=PANDADataset(tensor_path,csv_path,fold=fold,batch_size=batch_size)

#exit()
#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')


#network
import torchvision.models as models
from Network import Network
from LabelSmoothingLoss import LabelSmoothingLoss
model = Network(num_classes,'efficientnet-b3',dropout_p=0.4).to(device)
#model=models.resnext50_32x4d(num_classes=6).to(device)
#model=models.densenet121(num_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
#criterion = nn.CrossEntropyLoss(reduction='mean')
#criterion = nn.NLLLoss(reduction='mean')
criterion=LabelSmoothingLoss(label_smoothing,num_classes[0])
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model = nn.DataParallel(model)


#training loop
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Total number of paramters: {}'.format(pytorch_total_params))

from LrSchedulers import *
lr_scheduler=lr_AIAYN(optimizer,320,warmup_steps=1000,factor=0.1)

from Functions import *
for epoch in range(epochs):
    if epoch==reload_tensor_epoch:
        dataset.reload_tensor(reload_tensor_path,reload_batch_size)
    mean=torch.Tensor(dataset.data_mean).to(device).reshape(1,1,3,1,1)
    std=torch.Tensor(dataset.data_std).to(device).reshape(1,1,3,1,1)
    model.train(True)
    t=time.time()
    total_loss=0
    #dataset.train_batches=1
    for step in range(dataset.train_batches):
    #for step in range(10):
        #lr_scheduler.step()
        x,y=dataset.get_batch(step)
        x=torch.Tensor(x)

        x=x.to(device)
        x=rotate_and_flip(x,device,p=0.6)

        #y=[torch.Tensor(y[i]).to(device,dtype=torch.int64) for i in range(len(num_classes))]
        y=torch.Tensor(y[0]).to(device,dtype=torch.int64)
        #y=int2onehot(y,device)
        x=standardize_batch(x,mean,std)
        optimizer.zero_grad()
        x=cutout(x,device,cutout_ratio,std)
        outputs=model(x)
        #loss=multi_label_crossentropy(outputs,y,label_weights)
        loss=smoothcrossentropyloss(outputs,y,smoothing=label_smoothing)
        #loss=criterion(outputs,y)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
           scaled_loss.backward()
        #loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss+=loss.item()

        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Time: {}"
               .format(epoch+1, epochs, step+1, dataset.train_batches, total_loss/(step+1) ,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
    print('')
    if (epoch+1)%lr_decay_period==0:
        lr*=lr_decay
        update_lr(optimizer,lr)

    #if (epoch+1)%save_freq==0:
    save_weights(model,optimizer,epoch,checkpoints_folder)

    validate(model,device,dataset,mean,std,epoch)
