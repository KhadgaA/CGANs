
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
sketch_dir_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test_contours " 
labels_df_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test_Labels.csv" 

lambda_seg = 2.0 # controls segmentation loss weight
num_classes = 7
image_size = 128
batch_size = 8
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

def Generate_Fakes(sketches):
    # noisy_sketchs = add_gaussian_noise(sketches)
    noisy_sketchs = sketches
    noisy_sketchs_ = []
    fake_labels = torch.randint(0, num_classes, (sketches.size(0), ), device=sketches.device)
    for noisy_sketch, fake_label in zip(noisy_sketchs, fake_labels):
        channels = torch.zeros(
            size=(num_classes, *noisy_sketch.shape), device=noisy_sketch.device
        )
        channels[fake_label] = 1.0
        noisy_sketch = torch.cat((noisy_sketch.unsqueeze(0), channels), dim=0)
        noisy_sketchs_.append(noisy_sketch)

    noisy_sketchs = torch.stack(noisy_sketchs_)

    # convert fake_labels to one-hot encoding
    fake_labels = F.one_hot(fake_labels, num_classes=7).squeeze(1).float().to(device)

    return noisy_sketchs, fake_labels

generator = Generator(ngpu=ngpu, num_classes=7).to(device)
Tensor = torch.cuda.FloatTensor if (device.type == 'cuda') else torch.FloatTensor
# calculate the inception score for the model


import torch
from torchmetrics.image.inception import InceptionScore


from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64)

# calculate the FID score for the model
def calculate_fid_is_score(generator, num_classes, n_samples=2000, eps=1e-6):
    # Generate fake images
    fake_images=[]
    real_images = []
    for idx, (real_image, sketches, real_labels_onehot) in tqdm(enumerate(train_dl), 
                                                              desc= "Training", dynamic_ncols=True,total=len(train_dl)):  # Ensure that real_labels are provided

        # real_images  = Variable(real_images.type(Tensor).to(device), requires_grad=True)
        sketches = sketches.to(device)
        real_labels_onehot = real_labels_onehot.to(device)

        # generate fake input
        latent_input, gen_labels_onehot = Generate_Fakes(sketches=sketches)
        
        latent_input =  Variable(latent_input.to(device))

        # convert one-hot to labels
        aux_real_labels = torch.argmax(real_labels_onehot, dim=1)
        aux_fake_labels = torch.argmax(gen_labels_onehot, dim=1)

        gen_labels_onehot_long = aux_fake_labels.type(torch.long)
        real_labels_onehot_long = aux_real_labels.type(torch.long)
        
        fake_image = generator(latent_input,gen_labels_onehot_long)
        fake_images.append(fake_image.detach().cpu())


        real_images.append(real_image.detach().cpu())
        
        if (idx+1) * batch_size > n_samples:
            break 


    fake_images = torch.cat(fake_images, dim=0)
    fake_images = fake_images.type(torch.uint8)
    real_images = torch.cat(real_images, dim=0)
    real_images = real_images.type(torch.uint8)

    fid.update(fake_images, real=True)
    fid.update(real_images, real=False)
    fid_score = fid.compute()
    
    inception = InceptionScore()
    # generate some images

    inception.update(fake_images)
    incep_score = inception.compute()


    return fid_score.item(),  incep_score[0].item()

fid_score, incep_score = calculate_fid_is_score(generator, num_classes)
print("FID Score:", fid_score)
print("IS Score:", incep_score)

