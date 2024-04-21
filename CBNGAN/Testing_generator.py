import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import glob
# import torchvision.transforms as transforms


# def to_rgb(image):
#     rgb_image = Image.new("RGB", image.size)
#     rgb_image.paste(image)
#     return rgb_image


class ImageSketchDataset(Dataset):
    def __init__(
        self, image_dir, sketch_dir, labels_df, transform_image, transform_sketch,
    classof = None, paired = False):
        self.image_dir = image_dir
        self.sketch_dir = sketch_dir
        self.labels_df = pd.read_csv(labels_df)
        self.transform_image = transform_image
        self.transform_sketch = transform_sketch
        self.all_sketches = glob.glob1(
            self.sketch_dir, "*.png"
        )  # return .jpg or .png files
        self.paired = paired
        self.classof = classof
        self.new_df = self.labels_df.loc[self.labels_df[self.classof] == 1]
        self.new_df.reset_index(inplace = True)
    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, index):
        # print(self.labels_df,"here")
        while True:
            image_filename = self.new_df.iloc[index]["image"]  # Get image filename

            label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
            label = self.new_df.loc[index, label_cols].values.astype(
                "float32"
            )  # Load and convert labels

            image_path = os.path.join(self.image_dir, image_filename + ".jpg")
            
            if self.paired:
                sketch_filename = image_filename + "_segmentation.png"
                sketch_path = os.path.join(self.sketch_dir, sketch_filename)
                if not os.path.exists(sketch_path):
                    index = (index + 1 ) % self.__len__()
                    continue
            else:
                sketch_filename = np.random.choice(self.all_sketches)
                sketch_path = os.path.join(self.sketch_dir, sketch_filename)
                break
            
            break
        
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


def Generate_Fakes(sketches,classof):
    # noisy_sketchs = add_gaussian_noise(sketches)
    noisy_sketchs = sketches
    noisy_sketchs_ = []
    fake_labels = torch.ones(sketches.size(0) , device=sketches.device) * classof
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

generator = Generator(ngpu=ngpu, num_classes=num_classes).to(device)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import os

# Define a function to calculate FID
def calculate_fid(real_images, generated_images, batch_size=50, device="cuda"):
    def calculate_activation_statistics(images, model, batch_size=50, dims=2048, device="cuda"):
        n_batches = len(images) // batch_size
        act = np.empty((len(images), dims))

        for i in range(n_batches):
            batch = images[i * batch_size: (i + 1) * batch_size].to(device)
            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            act[i * batch_size: i * batch_size + pred.size(0)] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    # Load pretrained InceptionV3 model
    inception_model = inception_v3(pretrained=True).to(device)
    inception_model.eval()
    inception_model.fc = nn.Identity()  # Remove the final classification layer

    # Calculate statistics
    mu_real, sigma_real = calculate_activation_statistics(real_images, inception_model, batch_size, device=device)
    mu_generated, sigma_generated = calculate_activation_statistics(generated_images, inception_model, batch_size, device=device)

    # Calculate FID
    eps = 1e-6
    mu_diff = mu_real - mu_generated
    cov_mean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
    
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = mu_diff.dot(mu_diff) + np.trace(sigma_real + sigma_generated - 2 * cov_mean)
    return fid

# Example usage:
# Assuming real_images and generated_images are PyTorch tensors
# real_images and generated_images should be of shape (N, C, H, W)
# N = number of images, C = number of channels, H = height, W = width

# Convert PyTorch tensors to numpy arrays and transpose to correct shape
real_images_numpy = real_images.cpu().detach().numpy().transpose((0, 2, 3, 1))
generated_images_numpy = generated_images.cpu().detach().numpy().transpose((0, 2, 3, 1))

# Calculate FID
fid_score = calculate_fid(real_images, generated_images)
print(f"FID score: {fid_score}")



for class_label, classes in enumerate(["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]):

    train_ds = ImageSketchDataset(
    image_dir,
    sketch_dir,
    labels_df,
    transform_image=transform_image,
    transform_sketch=transform_sketch,
    classof=classes
)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
)

real_images, sketches, labels = next(iter(train_dl))

latent_input,gen_labels = Generate_Fakes(sketches,class_label)

aux_fake_labels = torch.argmax(gen_labels, dim=1)
aux_fake_labels = aux_fake_labels.type(torch.long)

fake_images = generator(latent_input.to(device),aux_fake_labels)