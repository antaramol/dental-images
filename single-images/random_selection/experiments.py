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
# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# model = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1)

# model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
# model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)

model



# %%

# change input channels to 1

try:
    conv_weight = model.conv1.weight
    model.conv1.in_channels = 1

    model.conv1.weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
except:
    try:
        conv_weight = model.conv1[0].weight
        model.conv1[0].in_channels = 1

        model.conv1[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
    except:
        pass


model(torch.rand(1, 1, 224, 224)).shape


# %%
