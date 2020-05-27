import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from Dataset import *
import torch

epochs=50
lr=0.5e-4
lr_decay=0.3
lr_decay_period=25
weight_decay=1e-4
save_freq=1
batch_size=24
checkpoints_folder='checkpoints'
cutout_ratio=0.6
num_classes=[6,4,4,]
label_weights=[1,0.3,0.3]
label_smoothing=0.05
n_drop_tile=4
fold=0

tensor_path="/mnt/sda1/Kaggle/PANDA/data/pickled_data"
csv_path='/mnt/sda1/Kaggle/PANDA/data/train_4fold.csv'


dataset=PANDADataset(tensor_path,csv_path,fold=fold,batch_size=batch_size)

dataloader=torch.utils.data.DataLoader(dataset, batch_size=24,
                        shuffle=True, num_workers=8)

# import time
# t=time.time()
# for i,data in tqdm(enumerate(dataloader)):
#     pass
# t=time.time()-t
# print(t)


import time
t=time.time()
for i in tqdm(range(len(dataset))):
    data=dataset[i]
t=time.time()-t
print(t)
