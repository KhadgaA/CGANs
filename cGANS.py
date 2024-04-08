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
# %matplotlib inline
from tqdm.notebook import tqdm
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os
import cv2
import numpy as np

# from google.colab import drive
# drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = "/content/drive/MyDrive/DL_Assignment_4/train_data/Train_data"
sketch_dir = "/content/drive/MyDrive/DL_Assignment_4/train_data_sketches/Train/Train_sketch/Contours"
labels_file = "/content/drive/MyDrive/DL_Assignment_4/train_data_sketches/Train/Train_labels.csv"
labels_df = pd.read_csv(labels_file)


image_path = "/content/drive/MyDrive/DL_4/ISIC_0024307.jpg"
sketch_path = "/content/drive/MyDrive/DL_4/sketch_1082.png"

image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def add_gaussian_noise(image, mean=0, stddev=1):

    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)

    noisy_image = image + noise

    return noisy_image

class ImageSketchDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, sketch_dir, labels_df, transform):
        self.image_dir = image_dir
        self.sketch_dir = sketch_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        image_filename = self.labels_df.iloc[index]['image']  # Get image filename

        label_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        label = self.labels_df.loc[index, label_cols].values.astype('float32')  # Load and convert labels

        image_path = os.path.join(self.image_dir, 'Train_data', image_filename + '.jpg')
        sketch_filename = 'sketch_' + image_filename + '.png'  # Assuming sketch filenames start with 'sketch_'
        sketch_path = os.path.join(self.sketch_dir, 'Train_sketch', 'Contours', sketch_filename)

        image = Image.open(image_path)
        sketch = Image.open(sketch_path)

        if self.transform:
            image = self.transform(image)
            sketch = self.transform(sketch)

        # Convert images to NumPy arrays
        image_np = np.array(image)
        sketch_np = np.array(sketch)

        # Add Gaussian noise to the sketch
        noisy_sketch_np = add_gaussian_noise(sketch_np)

        return torch.from_numpy(image_np), torch.from_numpy(noisy_sketch_np), torch.from_numpy(label)



# Transformations
transform = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
])


train_ds = ImageSketchDataset(image_dir, sketch_dir, labels_df, transform=transform)

train_dl = DataLoader(train_ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=True)

def denorm(img_tensors):
  return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.set_xticks([]); ax.set_yticks([])
  ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
  for images, _ , _ in dl:
    show_images(images, nmax)
    break

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False),
        )

        self.flatten = nn.Flatten()

        # Output layer
        self.fc = nn.Linear(1 + num_classes, 1)  # Add an extra dimension for the class labels

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x = self.main(x)
        x = self.flatten(x)

        # Concatenate labels with the features
        concatenated = torch.cat((x, labels), dim=1)

        x = self.fc(concatenated)
        x = self.sigmoid(x)
        return x


num_classes = len(train_ds.labels_df.columns) - 1
discriminator = Discriminator(num_classes)

"""Generator Network"""

from PIL import Image

sketch_path = "/content/drive/MyDrive/DL_4/sketch_1082.png"
sketch_image = Image.open(sketch_path)

sketch_channels = sketch_image.mode

print("Number of channels in sketch image:", sketch_channels)

def double_convolution(in_channels, out_channels):

    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)
        # Expanding path.

        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2)
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2)
        self.up_convolution_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=3,
            kernel_size=1
        )

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)


        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        out = self.out(x)
        return out


generator = Generator()

# Generate a random input tensor with shape (batch_size, channels, height, width)
batch_size = 1
input_channels = 1
height, width = 256, 256  # input image size
input_tensor = torch.randn(batch_size, input_channels, height, width)


output_tensor = generator(input_tensor)
print("Output tensor shape:", output_tensor.shape)

output_image = output_tensor.squeeze().detach().cpu().numpy()
plt.imshow(output_image.transpose(1, 2, 0))  #(batch, channels, height, width)
plt.axis('off')
plt.show()

# xb = torch.randn(batch_size, latent_size, 1, 1)
# fake_images = generator(xb)
# print(fake_images.shape)
# show_images(fake_images)

def train_discriminator(real_images, real_labels, opt_d):

    opt_d.zero_grad()

    # Passing real images through discriminator
    real_preds = discriminator(real_images, real_labels)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generating fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent, real_labels)  # Condition the generator on real labels

    # Passing fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images, real_labels)  # Condition the discriminator on real labels
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()

    return loss.item(), real_score, fake_score

def train_generator(opt_g):
  opt_g.zero_grad()

  #Generate fake images
  latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
  loss = F.binary_cross_entropy(preds, targets)

  # Fool the discriminator
  preds = discriminator(fake_images)
  targets = torch.ones(batch_size, 1, device=device)
  loss = F.binary_cross_entropy(preds, targets)

  #Updatwe generator weights
  loss.backward()
  opt_g.step()

  return loss.item()

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
  fake_images = generator(latent_tensors)
  fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
  save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
  print('Saving', fake_fname)
  if show:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses and scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, real_labels in tqdm(train_dl):  # Ensure that real_labels are provided

            loss_d, real_score, fake_score = train_discriminator(real_images, real_labels, opt_d)

            loss_g = train_generator(opt_g, real_labels)  # Pass real_labels to the generator

        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        print("Epoch [{}/{}], loss_g:{:.4f}, loss_d:{:.4f}, real_scores:{:.4f}, fake_score:{:.4f}".format(
            epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

        save_samples(epoch + start_idx, fixed_latent, show=False)

    return losses_g, losses_d, real_scores, fake_scores

lr = 0.0002
epochs = 1

history = fit(epochs, lr)

losses_g, losses_d, real_scores, fake_scores = history

plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses')

plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores')