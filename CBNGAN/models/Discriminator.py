import torch.nn as nn
import torch
class SandwichBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=True)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.reshape(gamma.size(0), self.num_features, 1, 1) *out + beta.reshape(beta.size(0), self.num_features, 1, 1)

        return out
    
class CategoricalConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )
        return out
class single_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(single_convolution, self).__init__()

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.swn1 = SandwichBatchNorm2d(out_channels, num_classes)
        self.swn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2,inplace=True)
        
    def forward(self, x):
        x = self.c1(x)
        x = self.swn1(x)
        x = self.act1(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # self.swn1 = SandwichBatchNorm2d(3, num_classes)
        # self.swn2 = SandwichBatchNorm2d(64, num_classes)
        # self.swn3 = SandwichBatchNorm2d(128, num_classes)
        # self.swn4 = SandwichBatchNorm2d(256, num_classes)
        self.des_block1 = single_convolution(3, 64, num_classes)
        self.des_block2 = single_convolution(64, 128, num_classes)
        self.des_block3 = single_convolution(128, 256, num_classes)
        self.des_block4 = single_convolution(256, 512, num_classes)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False)
        
        self.flatten = nn.Flatten()
        
        # Output layers
        
        self.fc_dis = nn.Linear(3969, 1)
        self.fc_aux = nn.Linear(3969, num_classes)  # Classifier for auxiliary task
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.des_block1(x)
        x = self.des_block2(x)
        x = self.des_block3(x)
        x = self.des_block4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        
        realfake = self.sigmoid(self.fc_dis(x)).view(-1, 1).squeeze(1)
        # realfake = self.fc_dis(x)

        classes = self.softmax(self.fc_aux(x))
        
        return realfake, classes
