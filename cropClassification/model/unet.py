import torch
from torch import nn
import torch.nn.functional as F
###############################################################################
# Basic UNet Blocks
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff_y = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_x = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
# Out Conv Block
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class DoubleConvDrop(nn.Module):
    """(convolution => [BN] => ReLU => Dropout) * 2"""

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownDrop(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDrop(in_channels, out_channels, drop_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
###############################################################################

class originalUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(originalUNet, self).__init__()

        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if bilinear:
            factor = 2 
        else:
            factor = 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class encodeDropUNet(nn.Module):
    def __init__(self, n_channels, n_classes, drop_rate, bilinear=True):
        super(encodeDropUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_rate = drop_rate

        if bilinear:
            factor = 2 
        else:
            factor = 1

        self.inc = DoubleConvDrop(n_channels, 64, drop_rate)
        self.down1 = DownDrop(64, 128, drop_rate)
        self.down2 = DownDrop(128, 256, drop_rate)
        self.down3 = DownDrop(256, 512, drop_rate)
        self.down4 = DownDrop(512, 1024 // factor, drop_rate)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetWithAncillary(nn.Module):
    def __init__(self, n_channels, n_classes, ancillary_data_dim, bilinear=True):
        super(UNetWithAncillary, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if bilinear:
            factor = 2
        else:
            factor = 1

        # UNet Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # UNet Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)  # Concatenate ancillary data in the bottleneck
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Process ancillary data (fully connected layer)
        self.ancillary_fc = nn.Sequential(
            nn.Linear(ancillary_data_dim, 1024 // factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024 // factor, 1024 // factor),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ancillary_data):
        # UNet encoding path (downsampling)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Process ancillary data
        ancillary_processed = self.ancillary_fc(ancillary_data)  # Shape: [batch_size, 1024 // factor]
        ancillary_processed = ancillary_processed.unsqueeze(2).unsqueeze(3)  # Add spatial dimensions
        ancillary_processed = ancillary_processed.expand_as(x5)  # Expand to match the size of x5
        # print(f"Shape of ancillary_processed: {ancillary_processed.shape}")

        # Concatenate ancillary data with the bottleneck feature map
        x5_with_ancillary = torch.cat([x5, ancillary_processed], dim=1)
        # print(f"x5 shape: {x5.shape}")
        # print(f"ancillary_processed shape: {ancillary_processed.shape}")
        # print(f"x5_with_ancillary shape: {x5_with_ancillary.shape}")


        # UNet decoding path (upsampling)
        x = self.up1(x5_with_ancillary, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
