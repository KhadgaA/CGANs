
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
    


# """Generator Network"""


class double_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(double_convolution, self).__init__()

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.swn1 = SandwichBatchNorm2d(out_channels, num_classes)
        self.act1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.swn2 = SandwichBatchNorm2d(out_channels, num_classes)
        self.act2 = nn.ReLU(inplace=True)
    def forward(self, x,y):
        x = self.c1(x)
        x = self.swn1(x,y)
        x = self.act1(x)
        x = self.c2(x)
        x = self.swn2(x,y)
        x = self.act2(x)
        return x



class Generator(nn.Module):
    def __init__(self, ngpu, num_classes):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.swn1 = SandwichBatchNorm2d(512, num_classes)
        self.swn2 = SandwichBatchNorm2d(256, num_classes)
        self.swn3 = SandwichBatchNorm2d(128, num_classes)
        self.swn4 = SandwichBatchNorm2d(64, num_classes)
        # self.swn5 = SandwichBatchNorm2d(512, num_classes)
        # self.swn6 = SandwichBatchNorm2d(1024, num_classes)
        # self.swn7 = SandwichBatchNorm2d(1024, num_classes)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution__2 = double_convolution(8, 4, num_classes)
        self.down_convolution__1 = double_convolution(4, 8, num_classes)
        self.down_convolution_0 = double_convolution(8, 1, num_classes)
        self.down_convolution_1 = double_convolution(1, 64, num_classes)
        self.down_convolution_2 = double_convolution(64, 128, num_classes)
        self.down_convolution_3 = double_convolution(128, 256, num_classes)
        self.down_convolution_4 = double_convolution(256, 512, num_classes)
        self.down_convolution_5 = double_convolution(512, 1024, num_classes)
        # Expanding path.

        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(1024, 512,num_classes)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_convolution_2 = double_convolution(512, 256,num_classes)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_convolution_3 = double_convolution(256, 128,num_classes)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_convolution_4 = double_convolution(128, 64,num_classes)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x, y):
        down__2 = self.down_convolution__2(x, y)
        down__1 = self.down_convolution__1(down__2, y)
        down_0 = self.down_convolution_0(down__1,  y)
        down_1 = self.down_convolution_1(down_0, y)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2, y)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4, y)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6, y)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8, y)

        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1),y)
        self.swn1(x, y)
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1),y)
        self.swn2(x, y)
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1),y)
        self.swn3(x, y)
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1),y)
        self.swn4(x, y)
        out = self.out(x)
        return out
    
