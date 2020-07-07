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
        if random_number < p:
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

def random_color(tensor,p=.5):
    '''
    brightness_factor (Union[float, torch.Tensor]) – Brightness adjust factor per element in the batch.
    0 does not modify the input image while any other number modify the brightness.
    contrast_factor (Union[float, torch.Tensor]) – Contrast adjust factor per element in the batch.
    0 generates a compleatly black image,
    1 does not modify the input image while any other non-negative number modify the brightness by this factor.
    hue_factor (float):0 means no shift. Therefore,
    both -PI and PI will give an image with complementary colors while 0 gives the original image.
    saturation_factor (float) – How much to adjust the saturation. 0 will give a black
    order=0 is brightness, 1 is contrast, 2 is saturation, 4 is hue.
    '''
    gamma=torch.ones(tensor.shape[1])
    gamma=np.random.uniform(0.75,1.25)
    random_number=np.random.uniform()
    if random_number<p:
        for i in range(tensor.shape[0]):
            tensor[i]=kornia.adjust_gamma(tensor[i],gamma)

    # hue=torch.ones(tensor.shape[1])
    # hue=np.random.uniform(-0.4,0.4)
    # random_number=np.random.uniform()
    # if random_number<p:
    #     for i in range(tensor.shape[0]):
    #         tensor[i]=kornia.adjust_hue(tensor[i],hue)

    # contrast=torch.ones(tensor.shape[1])
    # contrast=np.random.uniform(0.6,1.4)
    # random_number=np.random.uniform()
    # if random_number<p:
    #     for i in range(tensor.shape[0]):
    #         tensor[i]=kornia.adjust_contrast(tensor[i],contrast)
    #
    # brightness=torch.ones(tensor.shape[1])
    # brightness=np.random.uniform(-0.4,0.4)
    # random_number=np.random.uniform()
    # if random_number<p:
    #     for i in range(tensor.shape[0]):
    #         tensor[i]=kornia.adjust_brightness(tensor[i],brightness)

    return tensor

def cutout(data,ratio):
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
    #data[:,:,:,top_left[0]:top_left[0]+size,top_left[1]:top_left[1]+size]=(torch.randn((data.shape[0],data.shape[1],3,size,size),device=device)-0.5)*std*(12)**0.5
    data[:,:,:,top_left[0]:top_left[0]+size,top_left[1]:top_left[1]+size]=1
    return data
