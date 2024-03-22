
# %%
# read first images form outputs_2d/data/left/train/under_18 and outputs_2d/data/right/train/under_18
import os
import cv2
from torchvision.transforms import v2
import torch

data_folder = 'outputs_2d/data'
data_augmentation = True
class_names = ['under_18', 'over_18']

    
# %%

# define data augmentation 
## Gray scale images
data_transforms = {
    'train': v2.Compose([
        v2.RandomRotation(30) if data_augmentation else v2.RandomRotation(0),
        v2.RandomHorizontalFlip() if data_augmentation else v2.RandomHorizontalFlip(0),
        v2.Resize(224),
        v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': v2.Compose([
            v2.Resize(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }


# %%
# apply data transformation to the images
from torchvision.datasets import ImageFolder
left_images_datasets = {x: ImageFolder(os.path.join(data_folder, 'left', x), data_transforms[x]) for x in ['train', 'val']}
right_images_datasets = {x: ImageFolder(os.path.join(data_folder, 'right', x), data_transforms[x]) for x in ['train', 'val']}

print(left_images_datasets['train'][0][0].shape)


# %%
# stack the images from two sets on (M,N) to (M,N,2), use torch.stack, separate train and val and classes
import torch
import numpy as np

def stack_images(left_images_datasets, right_images_datasets):
    # stack the images from two sets on (M,N) to (M,N,2)
    left_images = np.stack([np.array(left_images_datasets['train'][i][0]) for i in range(len(left_images_datasets['train']))])
    right_images = np.stack([np.array(right_images_datasets['train'][i][0]) for i in range(len(right_images_datasets['train']))])
    stacked_images = np.stack([left_images, right_images], axis=3)
    return stacked_images

    stacked_images = stack_images(left_images_datasets, right_images_datasets)


#%%
# create pytorch image dataloaders with stacked images
from torch.utils.data import DataLoader, TensorDataset

batch_size = 4

dataloader = {x: DataLoader(

# %%
# show shape of first element in dataloade
inputs, classes = next(iter(dataloader['train']))
print(inputs.shape)
print(inputs[0].shape)
print(classes.shape)
print(inputs[0])

# %%
