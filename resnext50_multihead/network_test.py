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

#hyperparameters
epochs=200
lr=0.0256
lr_decay=0.98
lr_decay_period=2
weight_decay=1e-5
save_freq=20
batch_size=32
checkpoints_folder='checkpoints'

#dataset
from Dataset import *
tensor_path="/mnt/sda1/Kaggle/PANDA/data/processed_data/512x512.p"
csv_path='/mnt/sda1/Kaggle/PANDA/data/train.csv'

dataset=PANDADataset(tensor_path,csv_path,batch_size=batch_size)


#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#network
model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=6).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss(reduction='mean')
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model = nn.DataParallel(model)


#training loop
from Functions import *
for epoch in range(epochs):
    model.train(True)
    t=time.time()
    total_loss=0
    #for step in range(dataset.train_batches):
    for step in range(1):
        x,y=dataset.get_batch(step)
        x=torch.Tensor(x).to(device)
        y=torch.Tensor(y).to(device,dtype=torch.int64)
        x=normalize_by_max(x)

        output=model(x)
        loss=criterion(output,y)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        optimizer.step()
        total_loss+=loss.item()

        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Time: {}"
               .format(epoch+1, epochs, step+1, dataset.train_batches, total_loss/(step+1) ,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
    print('')
    if (epoch+1)%lr_decay_period==0:
        lr*=lr_decay
        update_lr(optimizer,lr)

    if (epoch+1)%save_freq==0:
        save_weights(model,optimizer,epoch,checkpoints_folder)

    validate(model,device,dataset)
