import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class PANDADataset():
    def __init__(self,tensor_path,csv_path,fold,batch_size=256,seed=1000,split=0.2):
        self.data=pickle.load(open(tensor_path,'rb')).transpose((0,1,4,2,3))
        self.size=self.data.shape[0]
        self.csv_path=csv_path
        self.fold=fold
        self._get_labels()
        self.batch_size=batch_size
        self.split=split
        self.seed=seed
        ##np.random.seed(seed)
        self._shuffle_and_split()
        np.random.seed()
        self._normalize_data()

    def reload_tensor(self,tensor_path,updated_batch_size=None):
        print("###Reloading tensor###")
        del self.data
        self.data=pickle.load(open(tensor_path,'rb'))
        stats_file='dataset_stats_{}x{}.p'.format(self.data.shape[3],self.data.shape[4])
        if os.path.exists(stats_file):
            with open(stats_file,'rb') as f:
                self.data_mean=pickle.load(f)
                self.data_std=pickle.load(f)
        else:
            self._compute_mean_std()
        if updated_batch_size:
            self.batch_size=updated_batch_size
            self.train_batches=int(len(self.train_indices)/self.batch_size)+1
            self.val_batches=int(len(self.val_indices)/self.batch_size)+1
        return self

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
        self.df=pd.read_csv(self.csv_path)
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
        self.train_batches=int(self.train_indices.shape[0]/self.batch_size)+1
        self.val_batches=int(self.val_indices.shape[0]/self.batch_size)+1
        return self

    def get_batch(self,idx,training=True):
        if training:
            indices=self.train_indices[idx*self.batch_size:(idx+1)*self.batch_size]
        else:
            indices=self.val_indices[idx*self.batch_size:(idx+1)*self.batch_size]

        x=self.data[indices]
        y=[self.isup_grades[indices],self.score1[indices],self.score2[indices]]
        # padding_masks=self.masks[indices]

        if idx==self.train_batches-1 and training:
            #print('###Shuffling dataset###')
            np.random.shuffle(self.train_indices)
        return x,y
