# this code runs pipeline.py with different parameters

import os

# for architecture in ["resnet34", "resnet18", "shufflenet_v2_x2_0", "shufflenet_v2_x1_5", "shufflenet_v2_x1_0", "shufflenet_v2_x0_5"]:
#     # from scratch, from pretrained, from pretrained and fixed feature extractor
#     for from_pretrained in ["--from-pretrained", "--from-pretrained --fixed-feature-extractor", ""]:
#         for data_augmentation in ["--data-augmentation", ""]:
#             os.system(f"python pipeline.py --architecture {architecture} {from_pretrained} {data_augmentation} --k-fold 5")


import torchvision.models
pretrained_models = [name for name in torchvision.models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(torchvision.models.__dict__[name])]

already_tested = ['alexnet', 'convnext_tiny', 'convnext_small', 'convnext_base',
       'convnext_large', 'efficientnet_b0', 'efficientnet_b1',
       'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
       'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
       'efficientnet_v2_s', 'densenet121', 'densenet161', 'densenet169',
       'densenet201', 'efficientnet_v2_m', 'efficientnet_v2_l',
       'googlenet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
       'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large',
       'mobilenet_v3_small', 'regnet_y_400mf', 'regnet_y_800mf',
       'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf',
       'regnet_y_16gf', 'regnet_y_32gf', 'regnet_y_128gf',
       'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf',
       'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf',
       'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
       'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
       'wide_resnet50_2', 'wide_resnet101_2', 'shufflenet_v2_x0_5',
       'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
       'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
       'vgg19', 'vgg19_bn', 'maxvit_t']

pretrained_models = [model for model in pretrained_models if model not in already_tested]


for architecture in pretrained_models:
# for architecture in ["alexnet", "resnet18"]:
    os.system(f"python pipeline.py --architecture {architecture} --from-pretrained --weights all --data-augmentation")
