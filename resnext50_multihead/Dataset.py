import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

class PANDADataset(torch.utils.data.Dataset):
    def __init__(self,tensor_path,csv_path,fold,training=True,shuffle=True,batch_size=256,seed=1000,split=0.2):
        #self.data=pickle.load(open(tensor_path,'rb')).transpose((0,1,4,2,3))
        self.tensor_path=tensor_path
        self.csv_path=csv_path
        self.df=pd.read_csv(self.csv_path)
        self.image_ids=np.asarray(self.df.image_id)
        self.size=len(self.df)
        self.training=True
        self.shuffle=shuffle
        self.fold=fold
        self._get_labels()
        self.batch_size=batch_size
        self.split=split
        self.seed=seed
        #np.random.seed(seed)
        self._shuffle_and_split()
        self._update_len()
        np.random.seed()
        #self._normalize_data()

    def switch_mode(self,training):
        if training:
            self.training=True
        else:
            self.training=False
        return self

    def update_batchsize(self,batch_size):
        self.batch_size=batch_size
        self._update_len()
        return self

    def _update_len(self):
        if len(self.train_indices)%self.batch_size==0:
            self.train_batches=int(len(self.train_indices)/self.batch_size)
        else:
            self.train_batches=int(len(self.train_indices)/self.batch_size)+1
        if len(self.val_indices)%self.batch_size==0:
            self.val_batches=int(len(self.val_indices)/self.batch_size)
        else:
            self.val_batches=int(len(self.val_indices)/self.batch_size)+1
        return self

    def __len__(self):
        if self.training:
            return self.train_batches
        else:
            return self.val_batches

    def __getitem__(self,idx):
        if idx==0 and self.training and self.shuffle:
            np.random.shuffle(self.train_indices)
            print("###train indices shuffled###")
        if self.training:
            indices=self.train_indices[idx*self.batch_size:(idx+1)*self.batch_size]
        else:
            indices=self.val_indices[idx*self.batch_size:(idx+1)*self.batch_size]

        #idx=self.train_indices[idx]
        image_ids=self.image_ids[indices]
        #tensor=np.load(os.path.join(self.tensor_path,image_ids+'.np.npy'))
        #tensor=pickle.load(open(os.path.join(self.tensor_path,image_ids+'.p'),'rb'))
        #tensor=0
        tensors=self._get_tensors_from_paths(image_ids).transpose((0,1,4,2,3))
        #labels=[self.isup_grades[idx],self.score1[idx],self.score2[idx]]
        labels=self.isup_grades[indices]
        gleason1=self.score1[indices]
        gleason2=self.score2[indices]
        return {'tensors':tensors,'labels':labels,'gleason1':gleason1,'gleason2':gleason2}
        #return tensor

    def _get_tensors_from_paths(self,image_ids):
        tensors=[pickle.load(open(os.path.join(self.tensor_path,id+'.p'),'rb'))
                 for id in image_ids]
        tensors=np.asarray(tensors,dtype='uint8')
        #print(tensors.shape)
        return tensors

    def _normalize_data(self):
        stats_file='dataset_stats_{}x{}.p'.format(self.data.shape[3],self.data.shape[4])
        if os.path.exists(stats_file):
            with open(stats_file,'rb') as f:
                self.data_mean=pickle.load(f)
                self.data_std=pickle.load(f)
        else:
            self._compute_mean_std()
        #self.data_mean=9
        #self.data_std=12
    def _compute_mean_std(self):
        print('###Standardizing data###')
        #self.data=255-self.data
        batch_size=32
        batches=int(len(self.train_indices)/batch_size)+1
        mean=[]
        std=[]
        for i in tqdm(range(batches)):
            batch_indices=self.train_indices[i*batch_size:(i+1)*batch_size]
            mean.append(np.mean(self.data[batch_indices]/255.0,axis=(0,1,3,4)))
            std.append(np.std(self.data[batch_indices]/255.0,axis=(0,1,3,4)))

        mean=np.asarray(mean)
        std=np.asarray(std)

        self.data_mean=np.mean(mean,axis=0)
        self.data_std=np.mean(std,axis=0)
        stats_file='dataset_stats_{}x{}.p'.format(self.data.shape[3],self.data.shape[4])
        with open(stats_file,'wb+') as f:
            pickle.dump(self.data_mean,f)
            pickle.dump(self.data_std,f)

    def _get_labels(self):
        gleason_score_patterns=['0','3','4','5']

        self.isup_grades=np.asarray(self.df.isup_grade.to_list(),dtype='int')
        gleason_scores=self.df.gleason_score.to_list()
        self.score1=[]
        self.score2=[]
        for i in range(self.size):
            if gleason_scores[i] == 'negative':
                self.score1.append(0)
                self.score2.append(0)
            else:
                split=gleason_scores[i].split('+')
                self.score1.append(int(gleason_score_patterns.index(split[0])))
                self.score2.append(int(gleason_score_patterns.index(split[1])))
        self.score1=np.asarray(self.score1,dtype='int')
        self.score2=np.asarray(self.score2,dtype='int')
        #self.

    def _shuffle_and_split(self):
        self.indices=np.arange(self.size)
        #np.random.shuffle(self.indices)
        fold_samples=int(self.split*self.size)
        split_vector=np.asarray(self.df.split.to_list(),dtype='int')
        #mask=np.ones(self.size,dtype='bool')
        #mask[self.fold*fold_samples:(self.fold+1)*fold_samples]=False
        self.train_indices=self.indices[split_vector!=self.fold]
        self.val_indices=self.indices[split_vector==self.fold]
        if self.train_indices.shape[0]%self.batch_size==0:
            self.train_batches=int(self.train_indices.shape[0]/self.batch_size)
        else:
            self.train_batches=int(self.train_indices.shape[0]/self.batch_size)+1
        if self.val_indices.shape[0]%self.batch_size==0:
            self.val_batches=int(self.val_indices.shape[0]/self.batch_size)
        else:
            self.val_batches=int(self.val_indices.shape[0]/self.batch_size)+1
        return self
