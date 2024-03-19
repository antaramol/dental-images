#%%
import torch
from torchvision import models
# %%
# model = models.densenet121(weights='IMAGENET1K_V1')
model = models.alexnet(weights='IMAGENET1K_V1')

model

# %%

# check if classifier is a list or a single layer
if isinstance(model.classifier, torch.nn.Sequential):
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)
else:
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 2)
model


# %%
