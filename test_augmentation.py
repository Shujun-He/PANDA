import kornia
import numpy as np
import pickle
import torch
import os
import matplotlib.pyplot as plt
import cv2
import kornia.augmentation as K

def random_color(tensor,p=1):
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
    order=np.array([0,1,2,4])
    np.random.shuffle(order)
    params={'brightness_factor': torch.Tensor(np.random.uniform(-0.2,0.2,size=tensor.shape[0])),
            'contrast_factor': torch.Tensor(np.random.uniform(0.8,1.2,size=tensor.shape[0])),
            'hue_factor': torch.Tensor(np.random.uniform(-0.2,0.2,size=tensor.shape[0])),
            'saturation_factor': torch.Tensor(np.random.uniform(0.8,1.2,size=tensor.shape[0])),
            'order': torch.Tensor(order)}
    brightness_factor=torch.Tensor(np.array([-0.4,0.4]))
    contrast_factor=torch.Tensor(np.array([0.6,1.4]))
    hue_factor=torch.Tensor(np.array([-0.4,0.4]))
    saturation_factor=torch.Tensor(np.array([0.6,1.4]))
    random_number=np.random.uniform()
    
    if random_number<p:
        tensor=kornia.color_jitter(tensor,
                                   brightness_factor,
                                   contrast_factor,
                                   saturation_factor,
                                   hue_factor,)
    return tensor

tensor_path='pickled_datas256n36'
files=[]
for file in os.listdir(tensor_path):
    files.append(os.path.join(tensor_path,file))
    
choice=np.random.choice(len(files))
file2visualize=files[choice]

tensor=pickle.load(open(file2visualize,'rb'))
shape=tensor.shape
gap=100
blank_image=np.zeros((2*(shape[1]+gap)+gap,shape[0]*(shape[2]+gap)+gap,3),dtype='uint8')
#blank_image=255

color_adjusted=random_color(torch.Tensor(tensor.copy()/255).permute(0,3,1,2)).permute(0,2,3,1).numpy()*255

#color_adjusted=tensor.copy()/2

for i in range(shape[0]):
    top_left=[gap,i*(shape[1]+gap)+gap]
    top_left_under=[2*gap+shape[1],i*(shape[1]+gap)+gap]
    blank_image[top_left[0]:top_left[0]+shape[1],top_left[1]:top_left[1]+shape[2]]=tensor[i]
    
    blank_image[top_left_under[0]:top_left_under[0]+shape[1],top_left_under[1]:top_left_under[1]+shape[2]]=color_adjusted[i]


# plt.imshow(blank_image)
# plt.show()

cv2.imwrite('image.png', blank_image)

# beta=15
# image_size=(shape[2]*beta,shape[2]*beta,3)
# blank_image=np.zeros(image_size,dtype='uint8')
# center=np.array([image_size[0]//2,image_size[1]//2])
# r=shape[1]*(beta//2-2)
# angle=2*3.1415/shape[0]
# post_size=100
# post_radius_scale=0.8
# for i in range(shape[0]):
    # x=int(r*np.cos(i*angle))
    # y=int(r*np.sin(i*angle))
    # top_left=np.array([x,y])+center
    # blank_image[top_left[0]-shape[1]//2:top_left[0]+shape[1]//2,top_left[1]-shape[2]//2:top_left[1]+shape[2]//2]=tensor[i]

    # post=np.array([int(r*np.cos(i*angle)*post_radius_scale),int(r*np.sin(i*angle)*post_radius_scale)])+center
    # #blank_image[post[0]-post_size//2:post[0]+post_size//2,post[1]-post_size//2:post[1]+post_size//2,:]=255


# cv2.imwrite('image.png', blank_image)
