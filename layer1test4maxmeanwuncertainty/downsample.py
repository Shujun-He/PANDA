import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import openslide
import matplotlib.pyplot as plt
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
from tqdm import tqdm
load_dir = "/mnt/sda1/Kaggle/PANDA/data/pickled_datas256n64/"
save_dir = "/mnt/sda1/Kaggle/PANDA/data/pickled_datas128n64/"
os.makedirs(save_dir, exist_ok=True)

resize=128
for file in tqdm(os.listdir(load_dir)):
    image=pickle.load(open(load_dir+file,'rb'))
    new_image=[]
    for i in range(image.shape[0]):
        downsample=cv2.resize(image[i],(resize,resize))
        new_image.append(downsample)
    new_image=np.asarray(new_image,dtype='uint8')
    pickle.dump(new_image,open(save_dir+file,'wb+'))
