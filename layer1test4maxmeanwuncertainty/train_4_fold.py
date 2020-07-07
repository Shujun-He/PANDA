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
    epochs=45
    lr=1e-4
    lr_decay=0.1
    lr_decay_period=25
    weight_decay=1e-5
    save_freq=1
    batch_size=16
    reload_tensor_epoch=0
    reload_batch_size=4
    cutout_ratio=0.6
    num_classes=[6,4,4,]
    label_weights=[1,0.3,0.3]
    label_smoothing=0.05
    n_drop_tile=4
    gradient_update_steps=3

    #dataset
    tensor_path= "/mnt/sda1/Kaggle/PANDA/data/pickled_datas128n36"
    reload_tensor_path="/mnt/sda1/Kaggle/PANDA/data/pickled_datas256n36"
    csv_path='/mnt/sda1/Kaggle/PANDA/data/train_4fold_clean.csv'
    #weights_path='/mnt/sda1/Kaggle/PANDA/layer1/test1_gleason/best_weights/fold{}top1.ckpt'.format(fold)

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
    model = Network(num_classes,'efficientnet-b3',dropout_p=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion=nn.SmoothL1Loss(reduction='none')
    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)
    #model.load_state_dict(torch.load(weights_path))

    #training loop
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))


    for epoch in range(epochs):
        if epoch==reload_tensor_epoch:
            dataset.tensor_path=reload_tensor_path
            batch_size=reload_batch_size
            gradient_update_steps*=4

        # mean=torch.Tensor(dataset.data_mean).to(device).reshape(1,1,3,1,1)
        # std=torch.Tensor(dataset.data_std).to(device).reshape(1,1,3,1,1)
        model.train(True)
        t=time.time()
        total_loss=0
        dataset.switch_mode(training=True)
        dataset.update_batchsize(batch_size)
        optimizer.zero_grad()
        for step in range(len(dataset)):
        #for step in range(1):
            #x,y=dataset.get_batch(step)
            data=dataset[step]
            x=data['tensors']
            y=data['labels']
            y2=data['gleason1']
            y3=data['gleason2']
            uncertainty_penalty=data['uncertainty_penalty']
            # print(x.shape)
            # print(y.shape)
            # print(x.shape)
            # print(y.shape)
            #x=drop_tile(x,n_drop_tile)
            #print(x.shape)
            #exit()
            uncertainty_penalty=torch.Tensor(uncertainty_penalty).to(device)
            x=torch.Tensor(x)
            x=x.to(device)
            x=rotate_and_flip(x,device,p=0.5)
            #x=cutout(x,cutout_ratio)
            #y=[torch.Tensor(y[i]).to(device,dtype=torch.int64) for i in range(len(num_classes))]
            #y=torch.Tensor(y[0]).to(device,dtype=torch.int64)
            y_regression=torch.Tensor(y).to(device)
            y2=torch.Tensor(y2).to(device)
            y3=torch.Tensor(y3).to(device)
            y_classification=torch.Tensor(y).to(device,dtype=torch.int64)
            #y=int2onehot(y,device)
            random_number=np.random.uniform()


            x=standardize_batch(x,)
            random_number=np.random.uniform()
            if random_number<0.5:
                x=cutout(x,cutout_ratio)
            else:
                x=random_color(x)

            outputs=model(x)
            del x
            loss=criterion(outputs[0],y_regression)+\
                 0.5*criterion(outputs[2],y2)+\
                 0.5*criterion(outputs[3],y3)+\
                 smoothcrossentropyloss(outputs[1],y_classification,smoothing=label_smoothing)
            loss=torch.mean(loss*uncertainty_penalty)
            loss=loss/gradient_update_steps
            #loss=smoothcrossentropyloss(outputs,y,smoothing=label_smoothing)
            #loss=criterion(outputs,y)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
               scaled_loss.backward()
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if step%gradient_update_steps==0 or step==len(dataset)-1:
                optimizer.step()
                optimizer.zero_grad()
            total_loss+=loss.item()*gradient_update_steps

            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Time: {}"
                   .format(epoch+1, epochs, step+1, dataset.train_batches, total_loss/(step+1) ,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
        print('')
        if (epoch+1)%lr_decay_period==0
            lr*=lr_decay
            update_lr(optimizer,lr)

    	if (epoch+1)==lr_decay_period:
    	       lr_decay_period=lr_decay_period//2

        if (epoch+1)%save_freq==0:
            save_weights(model,optimizer,epoch,checkpoints_folder)

        validate(model,csv_file,device,dataset,epoch,batch_size=batch_size*2)
    del model
    del loss
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

for fold in range(1):
    train_fold(fold)
