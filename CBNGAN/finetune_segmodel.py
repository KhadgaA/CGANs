
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.segmenatation_model import SegmentationModel, Decoder

from tqdm import tqdm
from datasets import *
# /teamspace/studios/this_studio/DL_Assignment_4/Dataset
image_dir = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train_data"
sketch_dir = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Paired_train_sketches"
labels_df = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train/Train_labels.csv"

# df = pd.read_csv(labels_df)
# train=df.sample(frac=0.8,random_state=200)
# test=df.drop(train.index)

# train.to_csv('train_split.csv',index = False)
# test.to_csv('test_split.csv',index = False)

labels_df = "/teamspace/studios/this_studio/DL_Assignment_4/CBNGAN/train_split.csv"
image_dir_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Train_data"
sketch_dir_val = "/teamspace/studios/this_studio/DL_Assignment_4/Dataset/Paired_train_sketches"
labels_df_val = "/teamspace/studios/this_studio/DL_Assignment_4/CBNGAN/test_split.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def i_over_u(y_true, y_pred):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    return iou

def dice_coefficient(y_true, y_pred):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    smooth = 1.0
    dice = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    return dice


def accuracy(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == masks).sum().item()
    total = masks.numel()
    acc = correct / total
    return acc



image_size = 128
batch_size = 64 *2
stats_image = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
stats_sketch = (0,5), (0.5)




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
dataset = ImageSketchDataset(
    image_dir,
    sketch_dir,
    labels_df,
    transform_image=transform_image,
    transform_sketch=transform_sketch,
    paired=True
)

val_dataset = ImageSketchDataset(
    image_dir_val,
    sketch_dir_val,
    labels_df_val,
    transform_image=transform_image,
    transform_sketch=transform_sketch,
    paired = True
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)



model_unfreeze = models.mobilenet_v2(pretrained=True)
for param in model_unfreeze.parameters():
    param.requires_grad = True

model_unfreeze.classifier = nn.Identity()

model_unfrozen = model_unfreeze.features

decoder = Decoder(num_encoder_features=1280, num_classes=1)

model = SegmentationModel(encoder=model_unfrozen, decoder=decoder).to(device)
criterion = nn.BCEWithLogitsLoss()
encoder_lr = 0.0001
decoder_lr = 0.001

encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=encoder_lr)
decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=decoder_lr)

encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.2, patience=5)
decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.2, patience=5)

num_epochs = 10

train_losses = []
val_losses = []
val_accuracies = []
train_iou = []
val_iou = []
train_dice = []
val_dice = []

for epoch in range(num_epochs):
    running_loss = 0.0
    running_train_iou = 0.0
    running_train_dice = 0.0
    running_val_iou = 0.0
    running_val_dice = 0.0
    running_val_acc = 0.0

    model.train()
    for i, (images, masks, _) in tqdm(enumerate(dataloader),total = len(dataloader)):
        images = images.to(device)
        masks = masks.to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        outputs = model(images)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, masks)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        running_loss += loss.item()

        batch_iou = 0.0
        batch_dice = 0.0
        for j in range(len(images)):
            iou = i_over_u(masks[j].cpu().numpy(), outputs[j].detach().cpu().numpy() > 0.5)
            dice = dice_coefficient(masks[j].cpu().numpy(), outputs[j].detach().cpu().numpy() > 0.5)
            running_train_iou += iou
            running_train_dice += dice
            batch_iou += iou
            batch_dice += dice

        train_iou.append(batch_iou / len(images))
        train_dice.append(batch_dice / len(images))

    train_losses.append(running_loss / len(dataloader))

    avg_train_iou = running_train_iou / len(dataset)
    avg_train_dice = running_train_dice / len(dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, masks, _) in tqdm(enumerate(val_dataloader),total = len(val_dataloader)):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            encoder_scheduler.step(val_loss)
            decoder_scheduler.step(val_loss)
            batch_iou = 0.0
            batch_dice = 0.0
            batch_acc = 0.0
            for j in range(len(images)):
                iou = i_over_u(masks[j].cpu().numpy(), outputs[j].detach().cpu().numpy() > 0.5)
                dice = dice_coefficient(masks[j].cpu().numpy(), outputs[j].detach().cpu().numpy() > 0.5)
                acc = accuracy(outputs[j], masks[j])
                running_val_iou += iou
                running_val_dice += dice
                running_val_acc += acc
                batch_iou += iou
                batch_dice += dice
                batch_acc += acc

            val_iou.append(batch_iou / len(images))
            val_dice.append(batch_dice / len(images))
            val_accuracies.append(batch_acc / len(images))

    val_losses.append(val_loss / len(val_dataloader))

    avg_val_iou = running_val_iou / len(val_dataset)
    avg_val_dice = running_val_dice / len(val_dataset)
    avg_val_acc = running_val_acc / len(val_dataset)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Avg Train IoU : {avg_train_iou:.4f}, Avg Train Dice: {avg_train_dice:.4f}, Avg Val IoU : {avg_val_iou:.4f}, Avg Val Dice: {avg_val_dice:.4f}, Val Accuracy: {avg_val_acc:.4f}')
    # print(f' Epoch [{epoch + 1}/{num_epochs}], Avg Train IoU : {avg_train_iou:.4f}, Avg Train Dice: {avg_train_dice:.4f}')
    save_model_path_task2 = 'segmentation_model.pth'
    torch.save(model.state_dict(), save_model_path_task2)