import glob
import random
import os
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import glob
from models.Generator import Generator
import torchvision.transforms as T
import torch
sketch_dir_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Test/Test_contours" 
ngpu = torch.cuda.device_count()
all_sketches = glob.glob1(sketch_dir_val, "*.png")  

num_samples = 60

lambda_seg = 2.0 # controls segmentation loss weight
num_classes = 7
image_size = 128
batch_size = 32
stats_image = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
stats_sketch = (0,5), (0.5)
transform_sketch = T.Compose(
    [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        # T.ToTensor(),
        # T.Normalize(*stats_sketch)
    ]
)
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

def Generate_Fakes(sketches,classof):
    # noisy_sketchs = add_gaussian_noise(sketches)
    noisy_sketchs = sketches
    noisy_sketchs_ = []
    fake_labels = torch.ones(sketches.size(0) , device=sketches.device,dtype=torch.long) * classof
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

def accuracy( outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def load_sketch(sketch_path):
    sketch = transform_sketch(Image.open(sketch_path))
    sketch_np = np.zeros_like(sketch)
    sketch_np[np.all(sketch) == 255] = 1.0
    sketch_np = sketch_np.astype(np.float32)
    return torch.from_numpy(sketch_np).unsqueeze(0)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

classifier = MNISTCNNModel(num_classes=7)
classifier.load_state_dict(torch.load("/teamspace/studios/this_studio/DL_Assignment_4/CBNGAN/classifier_9.pth"))
classifier.to(device)
classifier.eval()

generator = Generator(ngpu=1, num_classes=7).to(device)


for class_label, classes in enumerate(["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]):
    sketch_filenames = np.random.choice(all_sketches,num_samples)
    sketches =[]
    for sketch_filename in sketch_filenames:
        sketches.append(load_sketch(os.path.join(sketch_dir_val, sketch_filename)))
    # print(sketches[0].shape)
    sketches = torch.cat(sketches)
    # print(sketches.shape)
    latent_input,gen_labels = Generate_Fakes(sketches,class_label)
    aux_fake_labels = torch.argmax(gen_labels, dim=1)
    aux_fake_labels = aux_fake_labels.type(torch.long).to(device)
    fake_images = generator(latent_input.to(device),aux_fake_labels)

    pred_class = classifier(fake_images)
    acc = accuracy(pred_class, aux_fake_labels)
    print(f"acc of class {class_label}: {acc}")