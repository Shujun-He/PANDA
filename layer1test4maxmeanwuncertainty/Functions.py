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
    loss=loss.sum(1)
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
    print('Learning rate is updated to {}'.format(lr))

def save_weights(model,optimizer,epoch,folder):
    if os.path.isdir(folder)==False:
        os.makedirs(folder,exist_ok=True)
    torch.save(model.state_dict(), folder+'/epoch{}.ckpt'.format(epoch+1))
    #print('###Weights saved###')
    #torch.save(optimizer.state_dict(), folder+'/opt{}.ckpt'.format(epoch+1))

def standardize_batch(batch,):
    batch=batch/255.0
    # for i in range(batch.shape[0]):
    #     for j in range(batch.shape[1]):
    #         for k in range(3):
    #             batch[i,j,k]=(batch[i,j,k]-mean[k])/std[k]

    # batch=(batch-mean)/std
    # batch=1-batch
    return batch

def classification_threshold(tensor):
    for i in range(len(tensor)):
        if tensor[i] > 4.5:
            tensor[i]=5
        elif tensor[i] > 3.5:
            tensor[i]=4
        elif tensor[i] > 2.5:
            tensor[i]=3
        elif tensor[i] > 1.5:
            tensor[i]=2
        elif tensor[i] > 0.7:
            tensor[i]=1
        elif tensor[i] < 0.7:
            tensor[i]=0
    return tensor.astype('int')

def int2prob(tensor,nclass=6):
    out_prob=np.zeros(shape=(tensor.shape[0],nclass))
    for i in range(len(tensor)):
        for j in range(nclass):
            out_prob[i,j]=np.exp(-np.abs(tensor[i]-j))
            #out_prob[i,j]=1/np.abs(tensor[i]-j)
        #print(out_prob[i])
        out_prob[i]=out_prob[i]/np.sum(out_prob[i])
    return out_prob

def validate(model,csv_file,device,dataset,epoch,batch_size=64):
    batches=int(len(dataset.val_indices)/batch_size)+1
    model.train(False)
    total=0
    ground_truths=dataset.isup_grades[dataset.val_indices]
    predictions=[]
    losses=0
    criterion1=nn.SmoothL1Loss()
    criterion2=nn.NLLLoss()
    dataset.switch_mode(training=False)
    dataset.update_batchsize(batch_size)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            #print(i)
            #batch_indices=dataset.val_indices[i*batch_size:(i+1)*batch_size]
            data=dataset[i]
            batch_X=torch.Tensor(data['tensors']).to(device,)
            batch_Y=torch.Tensor(data['labels']).to(device,dtype=torch.int64)
            y2=data['gleason1']
            y3=data['gleason2']
            y2=torch.Tensor(y2).to(device)
            y3=torch.Tensor(y3).to(device)
            #batch_X=torch.Tensor(dataset.data[batch_indices]).to(device)
            batch_X=standardize_batch(batch_X)

            #batch_Y=torch.Tensor(dataset.isup_grades[batch_indices]).to(device,dtype=torch.int64)
            outputs = model(batch_X)
            del batch_X
            losses+=criterion1(outputs[0],batch_Y.squeeze().float())+\
                    0.5*criterion1(outputs[2],y2,)+\
                    0.5*criterion1(outputs[3],y3,)+\
                    criterion2(outputs[1],batch_Y)
            classification_predictions = torch.argmax(outputs[1],dim=1).squeeze()
            #predicted=(classification_predictions+outputs[0])/2
            predicted=outputs[0]
            predicted=classification_threshold(predicted.cpu().numpy())
            for pred in predicted:
                predictions.append(pred)
            del outputs
    torch.cuda.empty_cache()
    losses=(losses/batches).cpu()
    predictions=np.asarray(predictions)
    #print(ground_truths.shape)
    #print(predictions.shape)
    score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')
    acc=np.sum(predictions==ground_truths)/len(ground_truths)
    print('Validation QWK: {}, Accuracy: {} Val Loss: {}'.format(score,acc,losses))
    with open(csv_file,'a') as f:
        f.write('{},{},{},{}'.format(epoch+1,acc,score,losses))
        f.write('\n')

def predict(model,device,dataset,mean,std,batch_size=64,random_augment=None):
    batches=int(len(dataset.val_indices)/batch_size)+1
    model.train(False)
    total=0
    ground_truths=dataset.isup_grades[dataset.val_indices]
    predictions=[]
    raw_regression_outputs=[]
    uncertainty_penalty=[]
    losses=0
    criterion1=nn.SmoothL1Loss()
    criterion2=nn.CrossEntropyLoss()
    dataset.switch_mode(training=False)
    dataset.update_batchsize(batch_size)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            #print(i)
            #batch_indices=dataset.val_indices[i*batch_size:(i+1)*batch_size]
            data=dataset[i]
            batch_X=torch.Tensor(data['tensors']).to(device,)
            batch_Y=torch.Tensor(data['labels']).to(device,dtype=torch.int64)
            #batch_X=torch.Tensor(dataset.data[batch_indices]).to(device)
            batch_X=standardize_batch(batch_X)
            #batch_Y=torch.Tensor(dataset.isup_grades[batch_indices]).to(device,dtype=torch.int64)
            outputs = model(batch_X)
            del batch_X
            losses+=criterion1(outputs[0],batch_Y.squeeze().float())+criterion2(outputs[1],batch_Y)
            classification_predictions = torch.argmax(outputs[1],dim=1).squeeze()
            #predicted=(classification_predictions+outputs[0])/2
            predicted_raw=outputs[0]
            predicted=classification_threshold(predicted_raw.cpu().numpy())
            for pred in predicted:
                predictions.append(pred)
            for pred in predicted_raw:
                raw_regression_outputs.append(pred)
            for penalty in data['uncertainty_penalty']:
                uncertainty_penalty.append(penalty)

    losses=(losses/batches)
    predictions=np.asarray(predictions)
    raw_regression_outputs=np.asarray(raw_regression_outputs)
    ground_truths=dataset.isup_grades[dataset.val_indices]
    return raw_regression_outputs,predictions,ground_truths,uncertainty_penalty
