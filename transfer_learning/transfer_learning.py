"""
Part 2 of the Small Data Project Solution: Transfer Learning
Using the VGG16 model for transfer learning
"""

import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder

from TrainModel import train_model
from TestModel import test_model
from vacation_dataset import VacationDataset

# Directory names for training, validation, and test data
data_groups = {"training": "train", "validation": "val", "testing": "test"}

# Class names
class_names = Path("./imagedata-50/train").glob("*/")
class_names = [Path(n).name for n in class_names]
class_names.sort()

# Output directory
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

# According to the PyTorch VGG16 documenation (https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16), the following transforms were used for training: resize to 256, center crop to 224, scale pixel values to be between 0 and 1, then normalize pixel values with the following mean and standard deviation:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transforms = {}
image_transforms["training"] = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)
image_transforms["validation"] = transforms.Compose(
    [
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)
image_transforms["testing"] = image_transforms["validation"]

##################################################################################################### Data sets and loaders

BATCH_SIZE = 10
NUM_WORKERS = 4

data = {}
dataloader = {}

# Loop over data groups
for key, group in data_groups.items():

    # Data sets
    directory = Path(f"./imagedata-50/{group}")
    data[key] = ImageFolder(directory, transform=image_transforms[key])

    # Data loaders
    dataloader[key] = DataLoader(
        data[key], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
####################################################################################################
# Modifications to model for transfer learning

# 1. Get VGG16 trained model weights
model = models.vgg16(weights="DEFAULT")

# 2. Freeze the model layers so they won't all be trained again with our data
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the top layer classifier with a classifer for our 3 categories
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
####################################################################################################

# Hyperparameters for model training
# 1. num_epochs
# 2. criterion
# 3. optimizer
# 4. train_lr_scheduler
NUM_EPOCHS = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
# Same as ExponentialLR with gamma=0.5

def main():
    """
    Training and validation of model + testing of model
    """
    # Training, validation, and testing
    trained_model = train_model(
        model,
        criterion,
        optimizer,
        train_lr_scheduler,
        dataloader["training"],
        dataloader["validation"],
        dataloader["testing"],
        output_dir=output_dir,
        num_epochs=NUM_EPOCHS
    )
    
    # Plot training progress
    progress_data = pd.read_csv(output_dir / "training_progress.csv")
    progress_data.plot(x="epoch", y=["training_accuracy", "validation_accuracy"])
    plt.savefig(output_dir / "training_progress.png", bbox_inches="tight")
    
    # Display test images and labels
    #test_model(dataloader["testing"], trained_model, class_names)

if __name__ == "__main__":
    main()
    print("done")
