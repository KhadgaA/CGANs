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
from models.segmenatation_model import SegmentationModel
from models.Generator import Generator
from models.Discriminator import Discriminator  
from scipy.linalg import sqrtm
from torchvision.models import inception_v3

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

image_dir = "../dataset/dl_assignment_4/Train_data"
sketch_dir = "../dataset/dl_assignment_4/Train/Contours"
labels_df = "../dataset/dl_assignment_4/Train/Train_labels.csv"
image_dir_test = "../dataset/dl_assignment_4/Test/Test" 
sketch_dir_val = "../dataset/dl_assignment_4/Test/Test_countours " 
labels_df_val = "../dataset/dl_assignment_4/Test/Test_labels.csv" 

lambda_seg = 2.0 # controls segmentation loss weight

image_size = 128
batch_size = args.batch_size
stats_image = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
stats_sketch = (0,5), (0.5)


def add_gaussian_noise(image, mean=0, stddev=1):

    noise = torch.randn_like(image)

    noisy_image = image + noise

    return noisy_image


# class ImageSketchDataset(torch.utils.data.Dataset):
#     def __init__(
#         self, image_dir, sketch_dir, labels_df, transform_image, transform_sketch
#     ):
#         self.image_dir = image_dir
#         self.sketch_dir = sketch_dir
#         self.labels_df = pd.read_csv(labels_df)
#         self.transform_image = transform_image
#         self.transform_sketch = transform_sketch
#         self.all_sketches = glob.glob1(
#             self.sketch_dir, "*.png"
#         )  # return .jpg or .png files

#     def __len__(self):
#         return len(self.labels_df)

#     def __getitem__(self, index):
#         # print(self.labels_df,"here")
#         image_filename = self.labels_df.iloc[index]["image"]  # Get image filename

#         label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
#         label = self.labels_df.loc[index, label_cols].values.astype(
#             "float32"
#         )  # Load and convert labels

#         image_path = os.path.join(self.image_dir, image_filename + ".jpg")
#         sketch_filename = np.random.choice(
#             self.all_sketches
#         )  # Assuming sketch filenames start with 'sketch_'
#         sketch_path = os.path.join(self.sketch_dir, sketch_filename)

#         image = Image.open(image_path)
#         # print(image)

#         sketch = Image.open(sketch_path)

#         if self.transform_image:

#             image = self.transform_image(image)

#         if self.transform_sketch:
#             sketch = self.transform_sketch(sketch)

#         # Convert images to NumPy arrays
#         image_np = np.array(image)

#         sketch_np = np.zeros_like(sketch)
#         sketch_np[np.all(sketch) == 255] = 1.0
#         sketch_np = sketch_np.astype(np.float32)
#         # Add Gaussian noise to the sketch

#         # print(image_np, noisy_sketch_np)
#         # print(image_filename,label)
#         return (
#             torch.from_numpy(image_np),
#             torch.from_numpy(sketch_np),
#             torch.from_numpy(label),
#         )


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
    # num_workers=2,
    pin_memory=True,
)

val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=2,
    pin_memory=True,
)

def denorm(img_tensors):
    return img_tensors * stats_image[1][0] + stats_image[0][0]


# def show_images(images, nmax=64):
#   fig, ax = plt.subplots(figsize=(8, 8))
#   ax.set_xticks([]); ax.set_yticks([])
#   ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

# def show_batch(dl, nmax=64):/
#   for images, _ , _ in dl:
# # # #     show_images(images, nmax)
#     break


# class Discriminator(nn.Module):
#     def __init__(self, num_classes,ngpu=0):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False),
#         )

#         self.flatten = nn.Flatten()

#         # Output layer
#         self.fc = nn.Linear(56, 1)  # Add an extra dimension for the class labels

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, labels):

#         x = self.main(x)
#         x = self.flatten(x)

#         # Concatenate labels with the features
#         concatenated = torch.cat((x, labels), dim=1)
#         # print(concatenated.shape, x.shape, labels.shape)
#         x = self.fc(concatenated)
#         # x = self.sigmoid(x)
#         return x









# class SandwichBatchNorm2d(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.num_features = num_features
#         self.bn = nn.BatchNorm2d(num_features, affine=True)
#         self.embed = nn.Embedding(num_classes, num_features * 2)
#         self.embed.weight.data[:, :num_features].normal_(
#             1, 0.02
#         )  # Initialise scale at N(1, 0.02)
#         self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         out = self.bn(x)
#         gamma, beta = self.embed(y).chunk(2, 1)
#         out = gamma.reshape(gamma.size(0), self.num_features, 1, 1) *out + beta.reshape(beta.size(0), self.num_features, 1, 1)

#         return out
    
# class CategoricalConditionalBatchNorm2d(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.num_features = num_features
#         self.bn = nn.BatchNorm2d(num_features, affine=False)
#         self.embed = nn.Embedding(num_classes, num_features * 2)
#         self.embed.weight.data[:, :num_features].normal_(
#             1, 0.02
#         )  # Initialise scale at N(1, 0.02)
#         self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

#     def forward(self, x, y):
#         out = self.bn(x)
#         gamma, beta = self.embed(y).chunk(2, 1)
#         out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
#             -1, self.num_features, 1, 1
#         )
#         return out
# class single_convolution(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes):
#         super(single_convolution, self).__init__()

#         self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.swn1 = SandwichBatchNorm2d(out_channels, num_classes)
#         self.act1 = nn.LeakyReLU(0.2,inplace=True)
        
#     def forward(self, x,y):
#         x = self.c1(x)
#         x = self.swn1(x,y)
#         x = self.act1(x)

#         return x

# class Discriminator(nn.Module):
#     def __init__(self, num_classes, ngpu=0):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu

#         # self.swn1 = SandwichBatchNorm2d(3, num_classes)
#         # self.swn2 = SandwichBatchNorm2d(64, num_classes)
#         # self.swn3 = SandwichBatchNorm2d(128, num_classes)
#         # self.swn4 = SandwichBatchNorm2d(256, num_classes)
#         self.des_block1 = single_convolution(3, 64, num_classes)
#         self.des_block2 = single_convolution(64, 128, num_classes)
#         self.des_block3 = single_convolution(128, 256, num_classes)
#         self.des_block4 = single_convolution(256, 512, num_classes)

#         self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False)
        
#         self.flatten = nn.Flatten()
        
#         # Output layers
        
#         self.fc_dis = nn.Linear(3969, 1)
#         self.fc_aux = nn.Linear(3969, num_classes)  # Classifier for auxiliary task
        
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim = 1)

#     def forward(self, x, y):
#         x = self.des_block1(x, y)
#         x = self.des_block2(x, y)
#         x = self.des_block3(x, y)
#         x = self.des_block4(x, y)
#         x = self.conv5(x)
#         x = self.flatten(x)
        
#         realfake = self.sigmoid(self.fc_dis(x)).view(-1, 1).squeeze(1)
#         # realfake = self.fc_dis(x)

#         classes = self.softmax(self.fc_aux(x))
        
#         return realfake, classes



num_classes = len(train_ds.labels_df.columns) - 1
print('number of classes in dataset: ',num_classes)
# num_classes = 7


# # """Generator Network"""


# class double_convolution(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes):
#         super(double_convolution, self).__init__()

#         self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.swn1 = SandwichBatchNorm2d(out_channels, num_classes)
#         self.act1 = nn.ReLU(inplace=True)
#         self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.swn2 = SandwichBatchNorm2d(out_channels, num_classes)
#         self.act2 = nn.ReLU(inplace=True)
#     def forward(self, x,y):
#         x = self.c1(x)
#         x = self.swn1(x,y)
#         x = self.act1(x)
#         x = self.c2(x)
#         x = self.swn2(x,y)
#         x = self.act2(x)
#         return x



# class Generator(nn.Module):
#     def __init__(self, ngpu, num_classes):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.swn1 = SandwichBatchNorm2d(512, num_classes)
#         self.swn2 = SandwichBatchNorm2d(256, num_classes)
#         self.swn3 = SandwichBatchNorm2d(128, num_classes)
#         self.swn4 = SandwichBatchNorm2d(64, num_classes)
#         # self.swn5 = SandwichBatchNorm2d(512, num_classes)
#         # self.swn6 = SandwichBatchNorm2d(1024, num_classes)
#         # self.swn7 = SandwichBatchNorm2d(1024, num_classes)

#         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Contracting path.
#         # Each convolution is applied twice.
#         self.down_convolution__2 = double_convolution(8, 4, num_classes)
#         self.down_convolution__1 = double_convolution(4, 8, num_classes)
#         self.down_convolution_0 = double_convolution(8, 1, num_classes)
#         self.down_convolution_1 = double_convolution(1, 64, num_classes)
#         self.down_convolution_2 = double_convolution(64, 128, num_classes)
#         self.down_convolution_3 = double_convolution(128, 256, num_classes)
#         self.down_convolution_4 = double_convolution(256, 512, num_classes)
#         self.down_convolution_5 = double_convolution(512, 1024, num_classes)
#         # Expanding path.

#         self.up_transpose_1 = nn.ConvTranspose2d(
#             in_channels=1024, out_channels=512, kernel_size=2, stride=2
#         )
#         # Below, `in_channels` again becomes 1024 as we are concatinating.
#         self.up_convolution_1 = double_convolution(1024, 512,num_classes)

#         self.up_transpose_2 = nn.ConvTranspose2d(
#             in_channels=512, out_channels=256, kernel_size=2, stride=2
#         )
#         self.up_convolution_2 = double_convolution(512, 256,num_classes)

#         self.up_transpose_3 = nn.ConvTranspose2d(
#             in_channels=256, out_channels=128, kernel_size=2, stride=2
#         )
#         self.up_convolution_3 = double_convolution(256, 128,num_classes)
#         self.up_transpose_4 = nn.ConvTranspose2d(
#             in_channels=128, out_channels=64, kernel_size=2, stride=2
#         )
#         self.up_convolution_4 = double_convolution(128, 64,num_classes)
#         # output => `out_channels` as per the number of classes.
#         self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

#     def forward(self, x, y):
#         down__2 = self.down_convolution__2(x, y)
#         down__1 = self.down_convolution__1(down__2, y)
#         down_0 = self.down_convolution_0(down__1,  y)
#         down_1 = self.down_convolution_1(down_0, y)
#         down_2 = self.max_pool2d(down_1)
#         down_3 = self.down_convolution_2(down_2, y)
#         down_4 = self.max_pool2d(down_3)
#         down_5 = self.down_convolution_3(down_4, y)
#         down_6 = self.max_pool2d(down_5)
#         down_7 = self.down_convolution_4(down_6, y)
#         down_8 = self.max_pool2d(down_7)
#         down_9 = self.down_convolution_5(down_8, y)

#         up_1 = self.up_transpose_1(down_9)
#         x = self.up_convolution_1(torch.cat([down_7, up_1], 1),y)
#         self.swn1(x, y)
#         up_2 = self.up_transpose_2(x)
#         x = self.up_convolution_2(torch.cat([down_5, up_2], 1),y)
#         self.swn2(x, y)
#         up_3 = self.up_transpose_3(x)
#         x = self.up_convolution_3(torch.cat([down_3, up_3], 1),y)
#         self.swn3(x, y)
#         up_4 = self.up_transpose_4(x)
#         x = self.up_convolution_4(torch.cat([down_1, up_4], 1),y)
#         self.swn4(x, y)
#         out = self.out(x)
#         return out
    


# class EncoderWithFeatures(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.features = encoder.features
#         self.feature_outputs = []

#     def forward(self, x):
#         for name, layer in self.features.named_children():
#             x = layer(x)
#             # print("Output of layer", name, ":", x.shape)
#             if name in ['3', '7', '11', '15']:
#                 self.feature_outputs.append(x)
#         return x
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class Decoder(nn.Module):
#     def __init__(self, num_encoder_features, num_classes):
#         super().__init__()
#         self.up1 = nn.ConvTranspose2d(num_encoder_features, num_encoder_features // 2, kernel_size=2, stride=2)
#         self.conv1 = DoubleConv(num_encoder_features // 2, num_encoder_features // 2)

#         self.up2 = nn.ConvTranspose2d(num_encoder_features // 2, num_encoder_features // 4, kernel_size=2, stride=2)
#         self.conv2 = DoubleConv(num_encoder_features // 4, num_encoder_features // 4)


#         self.up3 = nn.ConvTranspose2d(num_encoder_features // 4, num_encoder_features // 8, kernel_size=2, stride=2)
#         self.conv3 = DoubleConv(num_encoder_features // 8, num_encoder_features // 8)


#         self.up4 = nn.ConvTranspose2d(num_encoder_features // 8, num_encoder_features // 16, kernel_size=2, stride=2)
#         self.conv4 = DoubleConv(num_encoder_features // 16, num_encoder_features // 16)

#         self.up5 = nn.ConvTranspose2d(num_encoder_features // 16, num_encoder_features//16, kernel_size=2, stride=2)

#         self.final_conv = nn.Conv2d(num_encoder_features // 16, num_classes, kernel_size=1)

#     def forward(self, x):
#         x1 = self.up1(x)

#         x1 = self.conv1(x1)

#         x2 = self.up2(x1)

#         x2 = self.conv2(x2)

#         x3 = self.up3(x2)

#         x3 = self.conv3(x3)

#         x4 = self.up4(x3)

#         x4 = self.conv4(x4)

#         x5 = self.up5(x4)

#         output = self.final_conv(x5)

#         return output
    
# class SegmentationModel(nn.Module):
#     def __init__(self, encoder=None, decoder=None, num_classes=1,ngpu=0):
#         super().__init__()
#         self.ngpu = ngpu
#         if encoder is None:
#             base_model = models.mobilenet_v2(pretrained=True)
#             base_model.classifier = nn.Identity()
#             for param in base_model.parameters():
#                 param.requires_grad = False
#             self.encoder = EncoderWithFeatures(base_model)
#         else:
#             self.encoder = encoder

#         if decoder is None:
#             self.decoder = Decoder(num_encoder_features=1280, num_classes=num_classes)
#         else:
#             self.decoder = decoder

#     def forward(self, x):
#         x = self.encoder(x)
#         return self.decoder(x)


discriminator = Discriminator(num_classes=num_classes, ngpu=ngpu).to(device)
generator = Generator(ngpu=ngpu, num_classes=num_classes).to(device)
seg_model = SegmentationModel(ngpu=ngpu).to(device)

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

mask_gen = mobilenet_v2(pretrained=True)
mask_gen.to(device)

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

            # gen_labels_onehot_long = gen_labels_onehot.type(torch.long)
            # real_labels_onehot_long = real_labels_onehot.type(torch.long)
            
            fake_images = generator(latent_input,gen_labels_onehot_long)

            #  real images
            validity_real, real_aux_output = discriminator(real_images, real_labels_onehot_long)
            #  fake images
            validity_fake, fake_aux_output = discriminator(fake_images, gen_labels_onehot_long)


            
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
                validity_fake, fake_aux_output = discriminator(fake_images, gen_labels_onehot_long)
                generated_mask = mask_gen(fake_images)
                # loss_g = -torch.mean(validity_fake) + aux_criterion(fake_aux_output, aux_fake_labels) 
                loss_g_adv = adversarial_loss(validity_fake, valid) + aux_criterion(fake_aux_output, aux_fake_labels)
                loss_g_seg = lambda_seg * seg_criterion(generated_mask, sketches.unsqueeze(1))
                loss_g = loss_g_adv + loss_g_seg
                loss_g.backward()
                opt_g.step()


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

    return losses_g, losses_d, real_scores, fake_scores


lr = args.lr #0.0002
epochs = args.n_epochs

history = fit(seg_model,epochs, lr)

losses_g, losses_d, real_scores, fake_scores = history

# calculate the inception score for the model
def calculate_inception_score(generator, num_classes, n_split=10, eps=1E-16):
    # Generate fake images
    num_samples = 50000
    latent_size = 100
    fake_images = torch.randn(num_samples, latent_size, 1, 1)
    fake_labels = torch.randint(0, num_classes, (num_samples,))

    # Generate fake images using the generator
    fake_images = generator(fake_images, fake_labels)

    # Load the InceptionV3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.to(device)
    inception_model.eval()

    # Calculate the activations for the fake images
    fake_activations = []
    batch_size = 100
    num_batches = num_samples // batch_size
    for i in range(num_batches):
        batch = fake_images[i * batch_size : (i + 1) * batch_size]
        batch = batch.to(device)
        activations = inception_model(batch)
        fake_activations.append(activations.detach().cpu())

    fake_activations = torch.cat(fake_activations, dim=0)

    # Calculate the activations for the real images
    real_activations = []
    for i, (real_images, _, _) in enumerate(val_dl):
        real_images = real_images.to(device)
        activations = inception_model(real_images)
        real_activations.append(activations.detach().cpu())

    real_activations = torch.cat(real_activations, dim=0)

    # Calculate the mean and covariance of the real activations
    mu_real = torch.mean(real_activations, dim=0)
    sigma_real = torch_cov(real_activations, rowvar=False)

    # Calculate the mean and covariance of the fake activations
    mu_fake = torch.mean(fake_activations, dim=0)
    sigma_fake = torch_cov(fake_activations, rowvar=False)

    # Calculate the KL divergence between the real and fake distributions
    kl_divergence = 0.5 * (
        torch.trace(sigma_real @ torch.inverse(sigma_fake))
        + torch.trace(sigma_fake @ torch.inverse(sigma_real))
        + (mu_real - mu_fake).T @ torch.inverse(sigma_real) @ (mu_real - mu_fake)
        + (mu_fake - mu_real).T @ torch.inverse(sigma_fake) @ (mu_fake - mu_real)
        - 2 * num_classes
    )

    # Calculate the inception score
    inception_score = np.exp(kl_divergence.item() / num_samples) + eps

    return inception_score

def torch_cov(m, rowvar=False):
    if rowvar:
        m = m.t()
    m = m.type(torch.double)
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

inception_score = calculate_inception_score(generator, num_classes)
print("Inception Score:", inception_score)

# calculate the FID score for the model
def calculate_fid_score(generator, num_classes, n_samples=50000, eps=1e-6):
    # Generate fake images
    fake_images = []
    for _ in range(n_samples):
        latent = torch.randn(1, latent_size, 1, 1, device=device)
        fake_image = generator(latent, y)
        fake_images.append(fake_image)
    fake_images = torch.cat(fake_images, dim=0)

    # Calculate mean and covariance of fake images
    fake_mean = torch.mean(fake_images, dim=0)
    fake_cov = torch_cov(fake_images, rowvar=False)

    # Generate real images
    real_images = []
    for _ in range(n_samples):
        real_image, _ = next(iter(train_dl))
        real_images.append(real_image)
    real_images = torch.cat(real_images, dim=0)

    # Calculate mean and covariance of real images
    real_mean = torch.mean(real_images, dim=0)
    real_cov = torch_cov(real_images, rowvar=False)

    # Calculate squared Frobenius norm between means
    mean_diff = real_mean - fake_mean
    mean_diff_squared = torch.sum(mean_diff * mean_diff)

    # Calculate trace of product of covariances
    cov_product = torch.matmul(real_cov, fake_cov)
    cov_product_sqrt = torch.sqrt(torch.abs(cov_product))
    trace_cov_product = torch.trace(cov_product_sqrt)

    # Calculate FID score
    fid_score = mean_diff_squared + torch.trace(real_cov) + torch.trace(fake_cov) - 2 * trace_cov_product
    fid_score += eps

    return fid_score.item()

fid_score = calculate_fid_score(generator, num_classes)
print("FID Score:", fid_score)