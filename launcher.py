# this code runs pipeline.py with different parameters

import os

for architecture in ["resnet18", "shufflenet_v2_x1_0"]:
    for from_pretrained in ["--from-pretrained", ""]:
        for data_augmentation in ["--data-augmentation", ""]:
            os.system(f"python pipeline.py --architecture {architecture} {from_pretrained} {data_augmentation}")