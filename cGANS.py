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

ngpu = torch.cuda.device_count()
print('num gpus available: ', ngpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

image_dir = "dataset/dl_assignment_4/Train_data"
sketch_dir = "dataset/dl_assignment_4/Train/Contours"

labels_df = "dataset/dl_assignment_4/Train/Train_labels.csv"



image_size = 256
batch_size = 64 * 2
stats_image = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
stats_sketch = (0,), (1)


def add_gaussian_noise(image, mean=0, stddev=1):

    noise = torch.randn_like(image)

    noisy_image = image + noise

    return noisy_image


class ImageSketchDataset(torch.utils.data.Dataset):
    def __init__(
        self, image_dir, sketch_dir, labels_df, transform_image, transform_sketch
    ):
        self.image_dir = image_dir
        self.sketch_dir = sketch_dir
        self.labels_df = pd.read_csv(labels_df)
        self.transform_image = transform_image
        self.transform_sketch = transform_sketch
        self.all_sketches = glob.glob1(
            self.sketch_dir, "*.png"
        )  # return .jpg or .png files

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        # print(self.labels_df,"here")
        image_filename = self.labels_df.iloc[index]["image"]  # Get image filename

        label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        label = self.labels_df.loc[index, label_cols].values.astype(
            "float32"
        )  # Load and convert labels

        image_path = os.path.join(self.image_dir, image_filename + ".jpg")
        sketch_filename = np.random.choice(
            self.all_sketches
        )  # Assuming sketch filenames start with 'sketch_'
        sketch_path = os.path.join(self.sketch_dir, sketch_filename)

        image = Image.open(image_path)
        # print(image)

        sketch = Image.open(sketch_path)

        if self.transform_image:

            image = self.transform_image(image)

        if self.transform_sketch:
            sketch = self.transform_sketch(sketch)

        # Convert images to NumPy arrays
        image_np = np.array(image)

        sketch_np = np.zeros_like(sketch)
        sketch_np[np.all(sketch) == 255] = 1.0
        sketch_np = sketch_np.astype(np.float32)
        # Add Gaussian noise to the sketch

        # print(image_np, noisy_sketch_np)
        # print(image_filename,label)
        return (
            torch.from_numpy(image_np),
            torch.from_numpy(sketch_np),
            torch.from_numpy(label),
        )


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


# for example in train_ds:
# 	print(example,"here")
# 	break


train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
)

# print(next(iter(train_dl)))

# import sys
# sys.exit()


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


class Discriminator(nn.Module):
    def __init__(self, num_classes,ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
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
        self.fc = nn.Linear(56, 1)  # Add an extra dimension for the class labels

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):

        x = self.main(x)
        x = self.flatten(x)

        # Concatenate labels with the features
        concatenated = torch.cat((x, labels), dim=1)
        # print(concatenated.shape, x.shape, labels.shape)
        x = self.fc(concatenated)
        x = self.sigmoid(x)
        return x


num_classes = len(train_ds.labels_df.columns) - 1


# """Generator Network"""


def double_convolution(in_channels, out_channels):

    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv_op


class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_0 = double_convolution(8, 1)
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)
        # Expanding path.

        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_convolution_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_convolution_4 = double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x):
        down_0 = self.down_convolution_0(x)
        down_1 = self.down_convolution_1(down_0)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)
        # import pdb
        # pdb.set_trace()
        # # break
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

discriminator = Discriminator(num_classes,ngpu).to(device)
generator = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
# generator = Generator()
# generator = generator.to(device)

# discriminator = discriminator.to(device)

# # # Generate a random input tensor with shape (batch_size, channels, height, width)
# batch_size = 1
# input_channels = 8
# height, width = 256, 256  # input image size
# input_tensor = torch.randn(batch_size, input_channels, height, width)


# output_tensor = generator(input_tensor)
# print("Output tensor shape:", output_tensor.shape)


# output_image = output_tensor.squeeze().detach().cpu().numpy()
# plt.imshow(output_image.transpose(1, 2, 0))  #(batch, channels, height, width)
# plt.axis('off')
# plt.show()
def sample_sketches(num_sketches):
    sketches = []
    for i in range(num_sketches):
        sketch = torch.zeros(1, 1, 256, 256)
        # Randomly sample 50% of the pixels
        sketch[
            0,
            0,
            torch.randint(0, 256, (int(0.5 * 256 * 256),)),
            torch.randint(
                0,
                256,
                (
                    int(
                        0.5 * 256 * 256,
                    )
                ),
            ),
        ] = 1.0
        sketches.append(sketch)
    return sketches


def Generate_Fakes(sketches):
    noisy_sketchs = add_gaussian_noise(sketches)
    noisy_sketchs_ = []
    fake_labels = torch.randint(0, 7, (sketches.size(0), 1), device=sketches.device)
    for noisy_sketch, fake_label in zip(noisy_sketchs, fake_labels):
        channels = torch.zeros(
            size=(7, *noisy_sketch.shape), device=noisy_sketch.device
        )
        channels[fake_label] = 1.0
        noisy_sketch = torch.cat((noisy_sketch.unsqueeze(0), channels), dim=0)
        noisy_sketchs_.append(noisy_sketch)

    noisy_sketchs = torch.stack(noisy_sketchs_)

    # convert fake_labels to one-hot encoding
    fake_labels = F.one_hot(fake_labels, num_classes=7).squeeze(1).float().to(device)

    return noisy_sketchs, fake_labels


def train_discriminator(
     real_images, real_labels, sketches, opt_d
):
    fake_images, fake_labels = Generate_Fakes(sketches, True)

    opt_d.zero_grad()
    # Passing real images through discriminator
    real_preds = discriminator(real_images, real_labels)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_loss.backward()
    real_score = torch.mean(real_preds).item()

    # Passing fake images through discriminator
    fake_preds = discriminator(fake_images, fake_labels)
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_loss.backward()
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    # loss.backward()
    opt_d.step()

    return loss.item(), real_score, fake_score


def train_generator( sketches, real_labels, opt_g):
    opt_g.zero_grad()
    # Generate fake images
    fake_images, fake_labels = Generate_Fakes( sketches, True)

    discriminator.eval()
    fake_preds = discriminator(fake_images, fake_labels)

    targets = torch.ones(fake_images.size(0), 1, device=device)

    # Fool the discriminator
    loss = F.binary_cross_entropy(fake_preds, targets)

    # Updatwe generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


sample_dir = "generated"
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, generator, train_dl, show=True):
    real_images, sketches, real_labels = next(iter(train_dl))
    latent_input, gen_labels = Generate_Fakes(sketches=sketches)
    fake_images = generator(latent_input)

    fake_fname = "generated-images-{0:0=4d}.png".format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


# fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
adversarial_loss = torch.nn.MSELoss()

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    generator.train()
    discriminator.train()

    # Losses and scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        # generator.eval()
        # discriminator.train()
        for idx, (real_images, sketches, real_labels) in tqdm(enumerate(train_dl), 
                                                              desc= "Training", dynamic_ncols=True,total=len(train_dl)):  # Ensure that real_labels are provided
            real_images = real_images.to(device)
            sketches = sketches.to(device)
            real_labels = real_labels.to(device)
            # Adversarial ground truths
            batch_size = real_images.shape[0]
            
            valid  = torch.full((batch_size,1), 1.0, dtype=torch.float, device=device)
            fake = torch.full((batch_size,1), 0.0, dtype=torch.float, device=device)

            # generate fake input
            latent_input, gen_labels = Generate_Fakes(sketches=sketches)
            # ------------------
            # Train generator
            # ------------------
            opt_g.zero_grad()
            fake_images = generator(latent_input)
            validity = discriminator(fake_images, gen_labels)
            loss_g = adversarial_loss(validity, valid)
            loss_g.backward()
            opt_g.step()

            # ----------------------
            # Train Discriminator
            # ----------------------

            opt_d.zero_grad()
            # Loss for real images
            validity_real = discriminator(real_images,real_labels)
            real_loss_d = adversarial_loss(validity_real, valid)
            real_score =torch.mean(validity_real).item()
            # Loss for fake images
            validity_fake = discriminator(fake_images.detach(), gen_labels)
            fake_loss_d = adversarial_loss(validity_fake, fake)
            fake_score = torch.mean(validity_fake).item()
            # Total discriminator loss
            loss_d = (real_loss_d + fake_loss_d) / 2
            loss_d.backward()
            opt_d.step()



            print(
                "Epoch [{}/{}], Batch [{}/{}], loss_g:{:.4f}, loss_d:{:.4f}, real_scores:{:.4f}, fake_score:{:.4f}".format(
                    epoch + 1, epochs, idx, len(train_dl), loss_g, loss_d, real_score, fake_score
                )
            )
            batches_done = epoch * len(train_dl) + idx
            if batches_done % 100 == 0:
                save_samples(epoch + start_idx, generator, train_dl, show=False)

    return losses_g, losses_d, real_scores, fake_scores


lr = 0.0002
epochs = 100

history = fit(epochs, lr)

losses_g, losses_d, real_scores, fake_scores = history

# plt.plot(losses_d, '-')
# plt.plot(losses_g, '-')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['Discriminator', 'Generator'])
# plt.title('Losses')

# plt.plot(real_scores, '-')
# plt.plot(fake_scores, '-')
# plt.xlabel('epoch')
# plt.ylabel('score')
# plt.legend(['Real', 'Fake'])
# plt.title('Scores')
