# this code runs pipeline.py with different parameters

import os

# for architecture in ["resnet34", "resnet18", "shufflenet_v2_x2_0", "shufflenet_v2_x1_5", "shufflenet_v2_x1_0", "shufflenet_v2_x0_5"]:
#     # from scratch, from pretrained, from pretrained and fixed feature extractor
#     for from_pretrained in ["--from-pretrained", "--from-pretrained --fixed-feature-extractor", ""]:
#         for data_augmentation in ["--data-augmentation", ""]:
#             os.system(f"python pipeline.py --architecture {architecture} {from_pretrained} {data_augmentation} --k-fold 5")


# import torchvision.models
# pretrained_models = [name for name in torchvision.models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(torchvision.models.__dict__[name])]

# already_tested = ['alexnet', 'convnext_tiny', 'convnext_small', 'convnext_base',
#        'convnext_large', 'densenet121', 'densenet161', 'densenet169',
#        'densenet201', 'efficientnet_b0', 'efficientnet_b1',
#        'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
#        'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

# pretrained_models = [model for model in pretrained_models if model not in already_tested]
# pretrained_models = ['resnet50', 'mobilenet_v2', 'wide_resnet50_2', 'regnet_x_16gf', 'resnext50_32x4d']



#### random selection
# for architecture in ['shufflenet_v2_x2_0', 'resnet18', 'resnet34', 'resnet50']:
#     for learning_rate in [0.0001, 0.00001, 0.001]:
#         for batch_size in [4, 8, 16, 32, 64, 128, 256]:
#             os.system(f"python pipeline.py --architecture {architecture} --from-pretrained --weights IMAGENET1K_V1 --data-augmentation --learning-rate {learning_rate} --epochs 60 --batch-size {batch_size} --input-data-folder UP_DOWN_stadiazione_CH_gimp")


#### random selection k-fold
# os.system(f"python pipeline.py --architecture resnet18 --from-pretrained --weights IMAGENET1K_V1 --data-augmentation --learning-rate 0.0001 --epochs 60 --batch-size 64 --input-data-folder UP_DOWN_stadiazione_CH_gimp --k-fold 5")
for architecture in ['resnet18', 'resnet34']:
    for learning_rate in [0.001, 0.0001]:
        for batch_size in [32, 64]:
            outputs_folder = f"outputs_10_fold_{architecture}_{learning_rate}_{batch_size}"
            # create .env file with the outputs folder as env variable
            with open(".env", "w") as f:
                f.write(f"OUTPUTS_FOLDER={outputs_folder}")

            os.system(f"python pipeline.py --architecture {architecture} --from-pretrained --weights IMAGENET1K_V1 --data-augmentation --learning-rate {learning_rate} --epochs 60 --batch-size {batch_size} --input-data-folder UP_DOWN_stadiazione_CH_gimp --k-fold 10")

