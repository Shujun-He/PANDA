import torch
import os
from sklearn import metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from DataAug import *

def smoothcrossentropyloss(pred,gold,n_class=6,smoothing=0.1):
    gold = gold.contiguous().view(-1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    #log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * pred)
    loss=loss.sum(1).mean()
    return loss

def multi_label_crossentropy(preds,targets,num_classes,weights,smoothing):
    loss=0
    for i in range(len(num_classes)):
        #print(preds[i].shape)
        #print(targets[i].shape)
        loss+=weights[i]*smoothcrossentropyloss(preds[i],targets[i],num_classes[i],smoothing)
    return loss

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_weights(model,optimizer,epoch,folder):
    if os.path.isdir(folder)==False:
        os.makedirs(folder,exist_ok=True)
    torch.save(model.state_dict(), folder+'/epoch{}.ckpt'.format(epoch+1))
    #print('###Weights saved###')
    #torch.save(optimizer.state_dict(), folder+'/opt{}.ckpt'.format(epoch+1))

def standardize_batch(batch,mean,std):
    batch=batch/255.0
    # for i in range(batch.shape[0]):
    #     for j in range(batch.shape[1]):
    #         for k in range(3):
    #             batch[i,j,k]=(batch[i,j,k]-mean[k])/std[k]

    # batch=(batch-mean)/std
    # batch=1-batch
    return batch

def validate(model,csv_file,device,dataset,mean,std,epoch,batch_size=64):
    batches=int(len(dataset.val_indices)/batch_size)+1
    model.train(False)
    total=0
    ground_truths=dataset.isup_grades[dataset.val_indices]
    predictions=[]
    losses=0
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(batches):
            #print(i)
            batch_indices=dataset.val_indices[i*batch_size:(i+1)*batch_size]
            batch_X=torch.Tensor(dataset.data[batch_indices]).to(device)
            batch_X=standardize_batch(batch_X,mean,std)
            batch_Y=torch.Tensor(dataset.isup_grades[batch_indices]).to(device,dtype=torch.int64)
            outputs = model(batch_X)[0]
            losses+=criterion(outputs,batch_Y)
            predicted = torch.argmax(outputs,dim=1)
            for pred in predicted:
                predictions.append(pred.cpu())
    losses=(losses/batches).cpu()
    predictions=np.asarray(predictions)
    #print(ground_truths.shape)
    #print(predictions.shape)
    score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')
    acc=np.sum(predictions==ground_truths)/len(ground_truths)
    print('Validation QWK: {}, Accuracy: {} Val Loss: {}'.format(score,acc,losses))
    with open(csv_file,'a') as f:
        f.write('{},{},{}'.format(epoch+1,score,losses))
        f.write('\n')

def predict(model,device,dataset,mean,std,batch_size=64,random_augment=None):
    batches=int(len(dataset.val_indices)/batch_size)+1
    model.train(False)
    total=0
    ground_truths=dataset.isup_grades[dataset.val_indices]
    logits=[]
    #losses=0
    criterion=nn.CrossEntropyLoss()
    Gleason2ISUP={'3+3':1,
                  '3+4':2,
                  '4+3':3,
                  '4+4':4,
                  '3+5':4,
                  '5+3':4,
                  '4+5':5,
                  '5+4':5,
                  '5+5':5}
    with torch.no_grad():
        for i in tqdm(range(batches)):
            #print(i)
            batch_indices=dataset.val_indices[i*batch_size:(i+1)*batch_size]
            batch_X=torch.Tensor(dataset.data[batch_indices]).to(device)
            batch_X=standardize_batch(batch_X,mean,std)
            batch_Y=torch.Tensor(dataset.isup_grades[batch_indices]).to(device,dtype=torch.int64)

            # outputs = F.log_softmax(model(batch_X)[0],dim=1)+\
            #           F.log_softmax(model(hflip(batch_X))[0],dim=1)+\
            #           F.log_softmax(model(vflip(batch_X))[0],dim=1)+\
            #           F.log_softmax(model(hflip(vflip(batch_X)))[0],dim=1)
            # #outputs=outputs[0]
            # outputs = model(batch_X)+\
            #           model(hflip(batch_X))+\
            #           model(vflip(batch_X))
            #outputs = F.log_softmax(model(batch_X),dim=1)
            outputs = [model(batch_X),]
                      # model(hflip(batch_X)),
                      # model(vflip(batch_X)),
                      # model(hflip(vflip(batch_X)))]
            for j in range(len(outputs)):
                if j==0:
                    ISUP_explicit=outputs[j][0]
                else:
                    ISUP_explicit=ISUP_explicit+outputs[j][0]
            # ISUP_implicit_master=torch.zeros_like(ISUP_explicit)
            # for j in range(len(outputs)):
            #     for k in range(outputs[j][0].shape[0]):
            #         ISUP_implicit=torch.zeros_like(ISUP_explicit)
            #         square=torch.matmul(outputs[j][1][k].reshape(4,1),outputs[j][2][k].reshape(1,4))
            #         ISUP_implicit[k,0]=ISUP_implicit[k,0]+square[0,0]
            #         ISUP_implicit[k,1]=ISUP_implicit[k,1]+square[1,1]
            #         ISUP_implicit[k,2]=ISUP_implicit[k,2]+square[1,2]
            #         ISUP_implicit[k,3]=ISUP_implicit[k,3]+square[2,1]
            #         ISUP_implicit[k,4]=ISUP_implicit[k,4]+square[2,2]
            #         ISUP_implicit[k,4]=ISUP_implicit[k,4]+square[3,1]
            #         ISUP_implicit[k,4]=ISUP_implicit[k,4]+square[1,3]
            #         ISUP_implicit[k,5]=ISUP_implicit[k,5]+square[2,3]
            #         ISUP_implicit[k,5]=ISUP_implicit[k,5]+square[3,2]
            #         ISUP_implicit[k,5]=ISUP_implicit[k,5]+square[3,3]
            #         ISUP_implicit[k]=ISUP_implicit[k]/torch.sum(ISUP_implicit[k])
            #         ISUP_implicit_master=ISUP_implicit_master+ISUP_implicit
            #predictions=ISUP_explicit+ISUP_implicit_master
            predictions=ISUP_explicit
            #outputs = F.log_softmax(model(batch_X),dim=1)
            if random_augment:
                for i in range(random_augment):
                    outputs+=model(rotate_and_flip(batch_X,device))
            #losses+=criterion(outputs,batch_Y)
            #predicted = torch.argmax(outputs,dim=1)
            for pred in predictions:
                logits.append(pred.cpu().numpy())
                #print(pred.cpu().numpy())
    # losses=(losses/batches).cpu()
    logits=np.asarray(logits)
    # #print(ground_truths.shape)
    # #print(predictions.shape)
    # score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')
    # acc=np.sum(predictions==ground_truths)/len(ground_truths)
    # print('Validation QWK: {}, Accuracy: {} Val Loss: {}'.format(score,acc,losses))
    # with open(csv_file,'a') as f:
    #     f.write('{},{},{}'.format(epoch,score,losses))
    #     f.write('\n')
    return logits
