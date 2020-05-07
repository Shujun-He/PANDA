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



def train_fold(fold):

    #hyperparameters
    epochs=65
    lr=2e-4
    lr_decay=0.3
    lr_decay_period=15
    weight_decay=1e-4
    save_freq=1
    batch_size=48
    checkpoints_folder='checkpoints'
    reload_tensor_epoch=-1
    reload_tensor_path="/mnt/sda1/Kaggle/PANDA/data/processed_data/512x512.p"
    reload_batch_size=16
    cutout_ratio=0.6
    num_classes=[6,4,4,]
    label_weights=[1,0,0]
    label_smoothing=0.05
    n_drop_tile=4

    #dataset
    tensor_path="/mnt/sda1/Kaggle/PANDA/data/processed_data/concat_n12.p"
    csv_path='/mnt/sda1/Kaggle/PANDA/data/train.csv'


    dataset=PANDADataset(tensor_path,csv_path,fold=fold,batch_size=batch_size)
    checkpoints_folder='checkpoints_fold{}'.format(fold)
    csv_file='log_fold{}.csv'.format(fold)
    #exit()

    #gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')


    #network
    import torchvision.models as models
    from Network import Network
    from LabelSmoothingLoss import LabelSmoothingLoss
    model = Network(num_classes,'efficientnet-b1',dropout_p=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)


    #training loop
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))


    for epoch in range(epochs):
        if epoch==reload_tensor_epoch:
            dataset.reload_tensor(reload_tensor_path,reload_batch_size)
        mean=torch.Tensor(dataset.data_mean).to(device).reshape(1,1,3,1,1)
        std=torch.Tensor(dataset.data_std).to(device).reshape(1,1,3,1,1)
        model.train(True)
        t=time.time()
        total_loss=0
        for step in range(dataset.train_batches):
        #for step in range(10):
            x,y=dataset.get_batch(step)
            x=drop_tile(x,n_drop_tile)
            #print(x.shape)
            #exit()
            x=torch.Tensor(x)
            x=x.to(device)
            x=rotate_and_flip(x,device,p=0.5)
            #y=[torch.Tensor(y[i]).to(device,dtype=torch.int64) for i in range(len(num_classes))]
            y=torch.Tensor(y[0]).to(device,dtype=torch.int64)
            #y=int2onehot(y,device)
            x=standardize_batch(x,mean,std)
            optimizer.zero_grad()
            #x=cutout(x,device,cutout_ratio,std)
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

        if (epoch+1)%save_freq==0:
            save_weights(model,optimizer,epoch,checkpoints_folder)

        validate(model,csv_file,device,dataset,mean,std,epoch)
    del model

for fold in range(5):
    train_fold(fold)
