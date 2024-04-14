
import torch.nn as nn
from torchvision import models
class EncoderWithFeatures(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.features = encoder.features
        self.feature_outputs = []

    def forward(self, x):
        for name, layer in self.features.named_children():
            x = layer(x)
            # print("Output of layer", name, ":", x.shape)
            if name in ['3', '7', '11', '15']:
                self.feature_outputs.append(x)
        return x
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, num_encoder_features, num_classes):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(num_encoder_features, num_encoder_features // 2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(num_encoder_features // 2, num_encoder_features // 2)

        self.up2 = nn.ConvTranspose2d(num_encoder_features // 2, num_encoder_features // 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(num_encoder_features // 4, num_encoder_features // 4)


        self.up3 = nn.ConvTranspose2d(num_encoder_features // 4, num_encoder_features // 8, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(num_encoder_features // 8, num_encoder_features // 8)


        self.up4 = nn.ConvTranspose2d(num_encoder_features // 8, num_encoder_features // 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(num_encoder_features // 16, num_encoder_features // 16)

        self.up5 = nn.ConvTranspose2d(num_encoder_features // 16, num_encoder_features//16, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(num_encoder_features // 16, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.up1(x)

        x1 = self.conv1(x1)

        x2 = self.up2(x1)

        x2 = self.conv2(x2)

        x3 = self.up3(x2)

        x3 = self.conv3(x3)

        x4 = self.up4(x3)

        x4 = self.conv4(x4)

        x5 = self.up5(x4)

        output = self.final_conv(x5)

        return output
    
class SegmentationModel(nn.Module):
    def __init__(self, encoder=None, decoder=None, num_classes=1,ngpu=0):
        super().__init__()
        self.ngpu = ngpu
        if encoder is None:
            base_model = models.mobilenet_v2(pretrained=True)
            base_model.classifier = nn.Identity()
            for param in base_model.parameters():
                param.requires_grad = False
            self.encoder = EncoderWithFeatures(base_model)
        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = Decoder(num_encoder_features=1280, num_classes=num_classes)
        else:
            self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
