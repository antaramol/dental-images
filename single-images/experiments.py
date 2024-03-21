#%%
import torch
from torchvision import models
# model = models.densenet121(weights='IMAGENET1K_V1')
# model = models.alexnet(weights='IMAGENET1K_V1')

# model = models.vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')
# model = models.swin_v2_b(weights='IMAGENET1K_V1')
# model = models.squeezenet1_0(weights='IMAGENET1K_V1')
# model = models.inception_v3(weights='IMAGENET1K_V1')
# model = models.inception_v3(weights='IMAGENET1K_V1')
# model = models.squeezenet1_0(weights='IMAGENET1K_V1')


# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# model = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1)

# model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
# model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)

model


# %%
try:
    conv_weight = model.conv1.weight
    model.conv1.in_channels = 1

    model.conv1.weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
except:
    try:
        conv_weight = model.features[0].weight
        model.features[0].in_channels = 1
        model.features[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
    except: # regnet
        try:
            conv_weight = model.stem[0].weight
            model.stem[0].in_channels = 1
            model.stem[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
        except: # mobilenet
            conv_weight = model.features[0][0].weight
            model.features[0][0].in_channels = 1
            model.features[0][0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))


model(torch.rand(1, 1, 224, 224)).shape

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
            try:
                num_ftrs = model.heads[-1].in_features
                model.heads[-1] = torch.nn.Linear(num_ftrs, 2)
            except: # special case squeezenet
                num_ftrs = model.classifier[1].in_channels
                model.classifier[1] = torch.nn.Conv2d(num_ftrs, 2, kernel_size=(1, 1))

model


# %%
