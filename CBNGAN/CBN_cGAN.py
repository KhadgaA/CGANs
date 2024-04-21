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

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
args = parser.parse_args()
print(args)


ngpu = torch.cuda.device_count()
print('num gpus available: ', ngpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

wandb.init(project = 'DL_Assignment_4', entity='m23csa017')

image_dir = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train_data"
sketch_dir = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train/Contours"
labels_df = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train/Train_labels.csv"

image_dir_test = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test" 
sketch_dir_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test_countours " 
labels_df_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test_Labels.csv" 

lambda_seg = 2.0 # controls segmentation loss weight

image_size = 128
batch_size = args.batch_size
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

def denorm(img_tensors):
    return img_tensors * stats_image[1][0] + stats_image[0][0]




num_classes = len(train_ds.labels_df.columns) - 1
print('number of classes in dataset: ',num_classes)
# num_classes = 7




discriminator = Discriminator(num_classes=num_classes, ngpu=ngpu).to(device)
generator = Generator(ngpu=ngpu, num_classes=num_classes).to(device)


model_unfreeze = models.mobilenet_v2(pretrained=True)
model_unfreeze.classifier = nn.Identity()
model_unfrozen = model_unfreeze.features
decoder = Decoder(num_encoder_features=1280, num_classes=1)
seg_model = SegmentationModel(encoder=model_unfrozen, decoder=decoder)

seg_model_saved = 'segmentation_model.pth'
seg_model.load_state_dict(torch.load(seg_model_saved))
seg_model.to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
    seg_model = nn.DataParallel(seg_model, list(range(ngpu)))




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



sample_dir = "generated_SBNGAN_images"
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, generator, train_dl, show=True):
    real_images, sketches, real_labels = next(iter(train_dl))
    latent_input, gen_labels = Generate_Fakes(sketches=sketches)

    aux_fake_labels = torch.argmax(gen_labels, dim=1)
    aux_fake_labels = aux_fake_labels.type(torch.long)

    fake_images = generator(latent_input.to(device),aux_fake_labels)

    fake_fname = "generated-images-{0:0=4d}.png".format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


# fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
adversarial_loss = torch.nn.BCELoss()
aux_criterion = nn.NLLLoss()
seg_criterion = nn.BCEWithLogitsLoss()
Tensor = torch.cuda.FloatTensor if (device.type == 'cuda') else torch.FloatTensor

# mask_gen = mobilenet_v2(pretrained=True)
# mask_gen.to(device)

def fit(mask_gen, epochs, lr, start_idx=1):

    torch.cuda.empty_cache()
    generator.train()
    discriminator.train()
    mask_gen.eval()
    # Losses and scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    k = 2
    p = 6

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):

        for idx, (real_images, sketches, real_labels_onehot) in tqdm(enumerate(train_dl), 
                                                              desc= "Training", dynamic_ncols=True,total=len(train_dl)):  # Ensure that real_labels are provided
            # Configure input
            real_images  = Variable(real_images.type(Tensor).to(device), requires_grad=True)
            sketches = sketches.to(device)
            real_labels_onehot = real_labels_onehot.to(device)
            # real_labels = torch.argmax(real_labels.to(device), dim=1)
            # Adversarial ground truths
            batch_size = real_images.shape[0]
            
            valid  = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            fake = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

            # generate fake input
            latent_input, gen_labels_onehot = Generate_Fakes(sketches=sketches)
            
            latent_input =  Variable(latent_input.to(device))
            # ----------------------
            # Train Discriminator
            # ----------------------
            
            opt_d.zero_grad()

            # convert one-hot to labels
            aux_real_labels = torch.argmax(real_labels_onehot, dim=1)
            aux_fake_labels = torch.argmax(gen_labels_onehot, dim=1)

            gen_labels_onehot_long = aux_fake_labels.type(torch.long)
            real_labels_onehot_long = aux_real_labels.type(torch.long)
            
            fake_images = generator(latent_input,gen_labels_onehot_long)

            #  real images
            validity_real, real_aux_output = discriminator(real_images)
            #  fake images
            validity_fake, fake_aux_output = discriminator(fake_images)

            loss_d_validity = adversarial_loss(validity_real, valid) + adversarial_loss(validity_fake, fake)

 # print(fake_aux_output.shape,gen_labels.shape,real_aux_output.shape,real_labels.shape)
            loss_d_aux = aux_criterion(fake_aux_output, aux_fake_labels) + aux_criterion(real_aux_output, aux_real_labels)
            
            loss_d = loss_d_validity + loss_d_aux

            # real_loss_d = adversarial_loss(validity_real, valid)
            real_score =torch.mean(validity_real).item()

            

            # fake_loss_d = adversarial_loss(validity_fake, fake)
            fake_score = torch.mean(validity_fake).item()
            
            # Total discriminator loss
            # loss_d = (real_loss_d + fake_loss_d) / 2
            loss_d.backward()
            opt_d.step()

            # Train the generator every n_critic steps
            if idx % args.n_critic == 0:
                # ------------------
                # Train generator
                # ------------------
                opt_g.zero_grad()
                fake_images = generator(latent_input,gen_labels_onehot_long)
                validity_fake, fake_aux_output = discriminator(fake_images)
                generated_mask = mask_gen(fake_images)
                # loss_g = -torch.mean(validity_fake) + aux_criterion(fake_aux_output, aux_fake_labels) 
                loss_g_adv = adversarial_loss(validity_fake, valid) + aux_criterion(fake_aux_output, aux_fake_labels)
                generated_mask = generated_mask.squeeze(1)
                loss_g_seg = lambda_seg * seg_criterion(generated_mask, sketches)
                loss_g = loss_g_adv + loss_g_seg
                loss_g.backward()
                opt_g.step()

                wandb.log(
                {
                    "loss_g": loss_g,
                    "loss_d":loss_d,
                    'real_score': real_score,
                    'fake_score': fake_score,
                    
                }
            )
                print(
                    "Epoch [{}/{}], Batch [{}/{}], loss_g:{:.4f}, loss_d:{:.4f}, real_scores:{:.4f}, fake_score:{:.4f}".format(
                        epoch + 1, epochs, idx, len(train_dl), loss_g, loss_d, real_score, fake_score
                    )
                )
                batches_done = epoch * len(train_dl) + idx
                if batches_done % args.sample_interval == 0:
                    save_samples(batches_done, generator, train_dl, show=False)
                
                batches_done += args.n_critic
                
                losses_d.append(loss_d.item())
                losses_g.append(loss_g.item())
                real_scores.append(real_score)
                fake_scores.append(fake_score)

        if (epoch+1) % 20 == 0:
            save_model_path_task2 = f'generator_model_{epoch+1}.pth'
            torch.save(generator.state_dict(), save_model_path_task2)
    return losses_g, losses_d, real_scores, fake_scores


lr = args.lr 
epochs = args.n_epochs

history = fit(seg_model,epochs, lr)

losses_g, losses_d, real_scores, fake_scores = history

