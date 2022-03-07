import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.activation = nn.ReLU()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.norm2(out)

        return out + x


class FeqeModel(nn.Module):
    def __init__(self, num_residual_blocks=5, channels=16):
        super(FeqeModel, self).__init__()

        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(3*4, channels//4, 1, padding='same'),
            nn.PixelUnshuffle(2),
        )

        self.residual_blocks = [ResidualConvBlock(channels) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*self.residual_blocks)

        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels*4, 1, padding='same'),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 3*4, 1, padding='same'),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        out = self.downsample(x)
        out = self.residual_blocks(out)
        out = self.upsample(out)

        output = x + out
        output = torch.clip(output, 0, 1)

        return output
