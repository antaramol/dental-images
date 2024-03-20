#%%
import torch
from torchvision import models
# %%
# model = models.densenet121(weights='IMAGENET1K_V1')
model = models.alexnet(weights='IMAGENET1K_V1')

# model = models.vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')
# model = models.swin_v2_b(weights='IMAGENET1K_V1')
# model = models.squeezenet1_0(weights='IMAGENET1K_V1')
# model = models.inception_v3(weights='IMAGENET1K_V1')
model
# %%

# check if classifier is a list or a single layer

try:
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

except AttributeError: # some models have classifier instead of fc
    # check if classifier is a list or a single layer
    try:
        if isinstance(model.classifier, torch.nn.Sequential):
            # sequential is a list of layers, take the last layer
            last_layer = model.classifier[-1]
            print(last_layer)
            num_ftrs = last_layer.in_features
            # set the last layer to a new layer with 2 output features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)

        else:
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, 2)
    except:
        # check if head is a list or a single layer
        try:
            num_ftrs = model.head.in_features
            model.head = torch.nn.Linear(num_ftrs, 2)
        except:
            num_ftrs = model.heads[-1].in_features
            model.heads[-1] = torch.nn.Linear(num_ftrs, 2)


model


# %%
