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

for architecture in pretrained_models:
# for architecture in ["alexnet", "resnet18"]:
    os.system(f"python pipeline.py --architecture {architecture} --from-pretrained --weights all --data-augmentation")
