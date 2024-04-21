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
    paired = False):
        self.image_dir = image_dir
        self.sketch_dir = sketch_dir
        self.labels_df = pd.read_csv(labels_df)
        self.transform_image = transform_image
        self.transform_sketch = transform_sketch
        self.all_sketches = glob.glob1(
            self.sketch_dir, "*.png"
        )  # return .jpg or .png files
        self.paired = paired
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        # print(self.labels_df,"here")
        while True:
            image_filename = self.labels_df.iloc[index]["image"]  # Get image filename

            label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
            label = self.labels_df.loc[index, label_cols].values.astype(
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
