import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import openslide
#import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import skimage.io
from skimage.transform import resize, rescale
from PIL import Image, ImageSequence
#import openslide
import pickle
#import torch
#import torch.nn.functional as F
import matplotlib.pyplot as plt

#os.nice(-20)

sz=256
N=36
#resize=320
train_labels = pd.read_csv('/mnt/sda1/Kaggle/PANDA/data/train_4fold_clean.csv')
data_dir = '/mnt/sda1/Kaggle/PANDA/data/train_images/'
mask_dir = '/mnt/sda1/Kaggle/PANDA/data/train_label_masks/'
mask_files = os.listdir(mask_dir)
save_dir = "/mnt/sda1/Kaggle/PANDA/data/pickled_datas{}n{}/".format(sz,N)
os.makedirs(save_dir, exist_ok=True)


def tile(img, sz=256,N=36):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    # mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
    #             constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        #mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img




for img_id in tqdm(train_labels.image_id):
    load_path = data_dir + img_id + '.tiff'
    save_path = save_dir + img_id + '.p'
    img = skimage.io.MultiImage(load_path,conserve_memory=True,dtype='tiff')[1]
    img=tile(img,sz=sz,N=N)
    for j in range(img.shape[0]):
        img[j] = cv2.cvtColor(img[j],cv2.COLOR_RGB2BGR)
    img=np.asarray(img,dtype='uint8')


    with open(save_path,'wb+') as f:
        pickle.dump(img,f)
