
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import time
import os
from tempfile import TemporaryDirectory

import logging


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn
import pandas as pd

MODELS_FOLDER = "outputs/models"

def load_dataset(train_folder, val_folder):
    data_transforms = {
    'train': v2.Compose([
        v2.RandomRotation(30),
        v2.RandomHorizontalFlip(),

        # v2.Resize(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ]),
    'val': v2.Compose([
        # v2.Resize(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': v2.Compose([
        # v2.Resize(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(train_folder if x == "train" else val_folder), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info(f"Classes: {class_names}")
    logging.info(f"Dataset sizes: {dataset_sizes}")
    logging.info(f"Device: {device}")

    return dataloaders, dataset_sizes, class_names, device


def train(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    since = time.time()

    history = {'train': {'loss': [], 'acc': []}, 'val': {'loss': [], 'acc': []}}

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[phase]['loss'].append(epoch_loss)
            # acc is a tensor, so we need to convert it to a float
            history[phase]['acc'].append(epoch_acc.item())

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # new line
        logging.info("")

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def train_model(train_folder, val_folder, from_pretrained=None):
    # load the dataset
    dataloaders, dataset_sizes, class_names, device = load_dataset(train_folder, val_folder)

    # load the model
    if from_pretrained:
        pretrained_weights = "IMAGENET1K_V1"
        if from_pretrained == "resnet18":
            model = models.resnet18(weights=pretrained_weights)

        elif from_pretrained == "shufflenet_v2_x1_0":
            model = models.shufflenet_v2_x1_0(weights=pretrained_weights)

        else:
            raise ValueError("Invalid model name")

    else:
        model = models.resnet18(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # train the model
    model, history = train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)
    
    # save the model into "outputs/pretrained_time/model.pth"
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = os.path.join(MODELS_FOLDER, f"{from_pretrained}_{now}", "model.pth")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)

    logging.info(f"Model saved into {model_path}")   


    return model_path










def evaluate_model(model, val_folder):
    # evaluate the model
    pass