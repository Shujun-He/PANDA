import numpy as np
import kornia
import torch
import os


def drop_tile(tensor,n=3):
    sample_shape=tensor.shape
    mask=np.ones(sample_shape,dtype='bool')
    for i in range(tensor.shape[0]):
        to_drop=np.random.choice(tensor.shape[1],size=n,replace=False)
        mask[i,to_drop]=False
    tensor=tensor[mask].reshape(sample_shape[0],sample_shape[1]-n,sample_shape[2],sample_shape[3],sample_shape[4])
    return tensor

def rotate_and_flip(tensor,device,p=0.5):
    for i in range(tensor.shape[0]):
        random_number=np.random.uniform()
        if random_number < p:
            center = torch.ones(tensor.shape[1], 2).to(device)
            center[:, 0] = tensor.shape[3] / 2  # x
            center[:, 1] = tensor.shape[2] / 2  # y
            #scale: torch.tensor = torch.ones(1)#*np.random.uniform(0.8,1.2)
            angle = torch.tensor([np.random.randint(-90,90,)*np.ones(tensor.shape[1])]).squeeze().to(device).float()
            #print(angle.shape)
            #print(tensor[i].shape)
            #M = kornia.get_rotation_matrix2d(center, angle, scale)#.to(device)
            #Mt = torch.ones((tensor.shape[0],2,3))
            #Mt[:] = M
            #Mt=Mt.to(device)
            #tensor[:,j]=kornia.warp_affine(tensor[:,j], Mt, dsize=(tensor.shape[3], tensor.shape[4]))
            #print(tensor.dtype)
            tensor[i]=kornia.rotate(tensor[i],angle,center)
        random_number=np.random.uniform()
        if random_number < p:
            tensor[i,:]=kornia.hflip(tensor[i,:])

        random_number=np.random.uniform()
        if random_number < p:
            tensor[i,:]=kornia.vflip(tensor[i,:])

        random_number=np.random.uniform()
        if random_number < 0.5:
            tensor[i,:]=tensor[i,:].transpose(-1,-2)

    return tensor


def hflip(tensor,p=1):
    for i in range(tensor.shape[0]):
        random_number=np.random.uniform()
        if random_number < p:
            tensor[i,:]=kornia.vflip(tensor[i,:])

    return tensor


def vflip(tensor,p=1):
    for i in range(tensor.shape[0]):
        random_number=np.random.uniform()
        if random_number < p:
            tensor[i,:]=kornia.vflip(tensor[i,:])
    return tensor



def cutout(data,device,ratio,std):
    # bounds=[int(image_size*0.08),int(image_size*0.92)]
    # signs=[-1,1]
    # signs2=np.random.randint(2,size=2)
    size=int(data.shape[3]*ratio)
    #print(size)
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         for k in range(3):
    top_left=np.random.randint(data.shape[3],size=2)
            #data[i,0,top_left[0]-size:top_left[0]+size,top_left[1]-size:top_left[1]+size]=0.056976318359375
            #for j in range(3):
                #data[i,:,top_left[0]:top_left[0]+size,top_left[1]:top_left[1]+size]=(np.random.rand()-0.5)*std[j]*(12)**0.5
                #data[i,j,:,top_left[0]:top_left[0]+size,top_left[1]:top_left[1]+size]=(np.random.rand()-0.5)*std[k]*(12)**0.5

    #data[:,:,:,top_left[0]:top_left[0]+size,top_left[1]:top_left[1]+size]=(torch.randn((data.shape[0],data.shape[1],3,size,size),device=device)-0.5)*std*(12)**0.5
    #data[:,:,:,top_left[0]:top_left[0]+size,top_left[1]:top_left[1]+size]=0
    data[:,:,:,top_left[0]:top_left[0]+size,top_left[1]:top_left[1]+size]=(torch.randn((data.shape[0],data.shape[1],3,size,size),device=device)-0.5)*std*(12)**0.5

    return data
