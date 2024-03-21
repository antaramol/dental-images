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
       'convnext_large', 'densenet121', 'densenet161', 'densenet169',
       'densenet201', 'efficientnet_b0', 'efficientnet_b1',
       'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
       'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

# pretrained_models = [model for model in pretrained_models if model not in already_tested]
pretrained_models = ['vit_h_14', 'inception_v3', 'squeezenet1_0', 'squeezenet1_1']

for architecture in pretrained_models:
# for architecture in ["alexnet", "resnet18"]:
    os.system(f"python pipeline.py --architecture {architecture} --from-pretrained --weights all --data-augmentation")
