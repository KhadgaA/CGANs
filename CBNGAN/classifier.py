import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# matplotlib inline
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import glob
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.models import mobilenet_v2

from torchvision import models, transforms

from datasets import *
from models.segmenatation_model import *
from models.Generator import Generator
from models.Discriminator import Discriminator 
ngpu = torch.cuda.device_count()
print('num gpus available: ', ngpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# wandb.init(project = 'DL_Assignment_4', entity='m23csa017')

image_dir = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train_data"
sketch_dir = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train/Contours"
labels_df = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train/Train_labels.csv"

image_dir_test = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test" 
sketch_dir_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test_contours" 
labels_df_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test_Labels.csv" 

lambda_seg = 2.0 # controls segmentation loss weight
num_classes = 7
image_size = 128
batch_size = 32
stats_image = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
stats_sketch = (0,5), (0.5)


def add_gaussian_noise(image, mean=0, stddev=1):

    noise = torch.randn_like(image)

    noisy_image = image + noise

    return noisy_image


# Transformations
transform_image = T.Compose(
    [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats_image),
    ]
)

transform_sketch = T.Compose(
    [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        # T.ToTensor(),
        # T.Normalize(*stats_sketch)
    ]
)
train_ds = ImageSketchDataset(
    image_dir,
    sketch_dir,
    labels_df,
    transform_image=transform_image,
    transform_sketch=transform_sketch,
)

val_ds = ImageSketchDataset(
    image_dir_test,
    sketch_dir_val,
    labels_df_val,
    transform_image=transform_image,
    transform_sketch=transform_sketch,
)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
)

val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)
def display_digit(image, label):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image.numpy().squeeze(), cmap='gray')
    ax.axis('off')
    ax.set_title(f"Label: {label}", fontsize=10)
    plt.show()



class ImageClassificationBase(nn.Module):

  def training_step(self, batch):
    images, _, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    labels = torch.argmax(labels, dim=1).type(torch.long)
    out = self(images)
    loss = F.cross_entropy(out, labels)
    acc = self.accuracy(out, labels)
    return  loss,acc.item()

  def training_epoch_end(self, outputs):
    batch_losses = [x['loss'] for x in outputs]
    epoch_losses = torch.stack(batch_losses).mean()
    batch_train_acc = [x['train_acc'] for x in outputs]
    epoch_train_acc = torch.stack(batch_train_acc).mean()
    return {'train_loss':epoch_losses.item(), 'train_acc':epoch_train_acc.item()}

  def validation_step(self, batch):
    images,_,  labels = batch
    images = images.to(device)
    labels = labels.to(device)
    labels = torch.argmax(labels, dim=1).type(torch.long)
    out = self(images)
    loss = F.cross_entropy(out, labels)
    acc = self.accuracy(out, labels)
    return {'val_loss' : loss, 'val_acc' : acc}

  def validation_epoch_end(self, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss' : epoch_loss.item(), 'val_acc' : epoch_acc.item()}

  def accuracy(self, outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  def epoch_end(self, epoch, result):
    print("Epoch [{}], train_loss:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}, train_acc:{:.4f}".format(
          epoch, result['train_loss'], result['val_loss'], result['val_acc'], result['train_acc']))

class MNISTCNNModel(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.network = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Linear(15376, num_classes),
            nn.Softmax(dim=1)
          )

    def forward(self, x):
        return self.network(x)

def evaluate(model, val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.Adam):
  history = []
  optimizer = opt_func(model.parameters(), lr)
  scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
  for epoch in range(epochs):
    model.train()
    train_losses = []
    train_acc = []
    for idx, batch in enumerate(tqdm(train_loader)):
      loss, acc = model.training_step(batch)
      train_losses.append(loss.item())
      train_acc.append(acc)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    scheduler.step()
    result = evaluate(model, val_loader)
    result['train_loss'] = (torch.tensor(train_losses)).mean().item()
    result['train_acc'] = (torch.tensor(train_acc)).mean().item()
    model.epoch_end(epoch, result)
    history.append(result)
    torch.save(model.state_dict(), f"classifier_{epoch}.pth")
  return history


num_epochs = 10
lr = 0.001
opt_func = torch.optim.Adam

model_10 = MNISTCNNModel(num_classes=7)
model_10.to(device)

history_10 = fit(num_epochs, lr, model_10, train_dl, val_dl, opt_func)

