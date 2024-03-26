#%%
import pandas as pd
import numpy as np
import cv2
import os
from torchvision.transforms import v2
from torchvision import models
import torch
from torchvision import datasets


SEED = 42

DATASET_FOLDER = "outputs_3d_images/data"

data_augmentation = True

#%%


model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model
# %%

# transform weight of conv1 layer to 2 channel
conv_weight = model.conv1.weight.data
print(conv_weight)
model.conv1.weight = torch.nn.Parameter(conv_weight[:, :2, :, :])
print(model.conv1.weight)

# %%
print(model.conv1.weight.shape)
# %%
print(model(torch.randn(1, 2, 224, 224)).shape)
# %%
