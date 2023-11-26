import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    1. Changed nn.ReLU() to nn.LeakyReLU()
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv
    1. Removed transposed upsampling, only using bilinear
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, ngf=8, output_size=384):
        super().__init__()

        self.ngf = ngf
        self.output_size = output_size

        self.n_channels = 1
        self.n_classes = 1
        self.num_resolutions = int(np.floor(np.log2(self.output_size))) - 2

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for l in range(self.num_resolutions):
            if l == 0:
                self.encoder.append(DoubleConv(self.n_channels, self.ngf))
                self.decoder.append(OutConv(self.ngf, self.n_classes))
            else:
                small_chan = self.ngf * 2**(l-1)
                big_chan = small_chan if l == (self.num_resolutions - 1) else small_chan * 2
                up_chan = small_chan if l == 1 else small_chan//2

                self.encoder.append(Down(small_chan, big_chan))
                self.decoder.append(Up(small_chan * 2, up_chan))

    def forward(self, x):
        encoder_outputs = [x] #x, x_0 res, x_1 res, ..., x_{N-1} res
        for enc_layer in self.encoder:
            x = enc_layer(x)
            encoder_outputs.append(x)

        offset = 2
        for l, dec_layer in enumerate(self.decoder[::-1]):
            if l == (len(self.decoder) - 1):
                x = dec_layer(x)
            else:
                x = dec_layer(x, encoder_outputs[-(l + offset)])

        return x

class Fixed_Input_UNet(nn.Module):
    def __init__(self, ngf=8, output_size=384):
        super().__init__()

        self.ngf = ngf
        self.output_size = output_size

        self.model = UNet(ngf=self.ngf, output_size=self.output_size)
        self.latent = torch.randn((1, 1, self.output_size, self.output_size))
    
    def forward(self):
        return self.model(self.latent)
    