
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

from .data_processing import MODELS_FOLDER, OUTPUTS_FOLDER

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn
import pandas as pd


def load_dataset(data_folder, data_augmentation, batch_size):

    ## Gray scale images
    data_transforms = {
        'train': v2.Compose([
            v2.RandomRotation(30) if data_augmentation else v2.RandomRotation(0),
            v2.RandomHorizontalFlip() if data_augmentation else v2.RandomHorizontalFlip(0),
            v2.Resize(224),
            v2.Grayscale(num_output_channels=1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485], std=[0.229])
        ]),
        'val': v2.Compose([
            v2.Resize(224),
            v2.Grayscale(num_output_channels=1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485], std=[0.229])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_folder, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    

    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # logging.info(image_datasets)
    # # show first image shape
    # inputs, classes = next(iter(dataloaders['train']))
    # logging.info(f"First image shape: {inputs[0].shape}")
    
    # logging.info(inputs[0])

    return dataloaders, dataset_sizes, class_names, device


def train(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    since = time.time()

    history = {'train': {'loss': [], 'acc': []}, 'val': {'loss': [], 'acc': []}}

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch+1, num_epochs))
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


def update_results_csv(architecture, from_pretrained, weight, fixed_feature_extractor, data_augmentation, epochs, learning_rate, batch_size,
                       best_acc, final_acc, model_path):

    results = pd.DataFrame({"architecture": [architecture], "from_pretrained": [from_pretrained], "weight": [weight],
                            "fixed_feature_extractor": [fixed_feature_extractor],
                            "data_augmentation": [data_augmentation], "epochs": [epochs], "learning_rate": [learning_rate], "batch_size": [batch_size],
                            "best_acc": [best_acc], "final_acc": [final_acc], "model_path": [model_path]})

    # read the csv file if it exists, else create a new one
    if os.path.exists(os.path.join(OUTPUTS_FOLDER, "results.csv")):
        old_results = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "results.csv"))
        results = pd.concat([old_results, results])

    results.to_csv(os.path.join(OUTPUTS_FOLDER, "results.csv"), index=False)



def train_model(dataloaders, dataset_sizes, class_names, device,
                 architecture='resnet18', weights='IMAGENET1K_V1',
                 from_pretrained=False, epochs=25, learning_rate=0.001, 
                 fixed_feature_extractor=False):
    
    # load the dataset


    # dataloaders, dataset_sizes, class_names, device = load_dataset(data_folder, data_augmentation, batch_size)
        
    

    # load the model    
    if weights == "all":
        logging.info(f"Loading all weights for {architecture}")
        weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", architecture)
    elif weights is not None:
        logging.info(f"Loading {weights} weights")
        weight_enum = [weights]
    else:
        logging.info("No weights specified, using default")
        weight_enum = ["IMAGENET1K_V1"]
        
    for weight in weight_enum:
        logging.info(f"Loading {weight} weights")
        model = models.__dict__[architecture](weights=weight) if from_pretrained else models.__dict__[architecture]()
        
        # logging.info(f"Model: {model}")
        if from_pretrained:
            logging.info(f"Model {architecture} loaded with {weight} weights")
        else:
            logging.info(f"Model {architecture} loaded without pretrained weights")


        if from_pretrained and fixed_feature_extractor:
            for param in model.parameters():
                param.requires_grad = False


        

        
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
                raise Exception("Could not change input channels to 1")

        logging.info(model(torch.randn(1, 1, 224, 224)).shape)


        try:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))

        except AttributeError: # some models have classifier instead of fc
            # check if classifier is a list or a single layer
            try:
                if isinstance(model.classifier, torch.nn.Sequential):
                    num_ftrs = model.classifier[-1].in_features
                    model.classifier[-1] = torch.nn.Linear(num_ftrs, len(class_names))
                else:
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, len(class_names))
            except:
                # check if head is a list or a single layer
                try:
                    num_ftrs = model.head.in_features
                    model.head = torch.nn.Linear(num_ftrs, len(class_names))
                except:
                    try:
                        num_ftrs = model.heads[-1].in_features
                        model.heads[-1] = torch.nn.Linear(num_ftrs, len(class_names))
                    except: # special case squeezenet
                        num_ftrs = model.classifier[1].in_channels
                        model.classifier[1] = torch.nn.Conv2d(num_ftrs, len(class_names), kernel_size=(1, 1))


        





    


        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        logging.info("Batch size: " + str(dataloaders['train'].batch_size))
        logging.info("Learning rate: " + str(learning_rate))

        # Observe that all parameters are being optimized
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Decay LR by a factor of 0.1 every N epochs
        n = 7
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=n, gamma=0.1)

        # train the model
        model, history = train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)
        
        # save the model into "outputs/arch_time/model.pth"
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(MODELS_FOLDER, f"{architecture}_{now}", "model.pth")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model, model_path)

        # plot history and save it into "outputs/pretrained_time/history.png"
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(history['train']['loss'], label="train")
        ax[0].plot(history['val']['loss'], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        
        ax[1].plot(history['train']['acc'], label="train")
        ax[1].plot(history['val']['acc'], label="val")
        ax[1].set_title("Acc")
        ax[1].legend()

        plt.savefig(os.path.join(os.path.dirname(model_path), "history.png"))


        # save a png with the confusion matrix
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)

        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

        plt.figure(figsize=(10, 7))
        seaborn.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(os.path.dirname(model_path), "confusion_matrix.png"))


        

        logging.info(f"Model saved into {model_path}")   

    return model_path, history




def evaluate_model(model_path, dataloaders, device):
    model = torch.load(model_path)
    model.eval()

    # get acc on the validation set
    predictions = []
    real_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            real_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    real_labels = np.array(real_labels)

    accuracy = np.mean(predictions == real_labels)
    logging.info(f"Accuracy: {accuracy}")
    return accuracy, predictions, real_labels

