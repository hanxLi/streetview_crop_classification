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
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ensure padding sizes are integers
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

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

class FiLM(nn.Module):
    def __init__(self, input_dim, feature_map_channels):
        """
        FiLM Layer: Computes gamma and beta based on ancillary input.
        """
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(input_dim, feature_map_channels)
        self.beta = nn.Linear(input_dim, feature_map_channels)

    def forward(self, feature_map, ancillary_data):
        """
        Apply FiLM modulation: gamma * feature_map + beta
        """
        gamma = self.gamma(ancillary_data).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = self.beta(ancillary_data).unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        return gamma * feature_map + beta
    
##############################################################
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

class UNetWithUncertainty(nn.Module):
    def __init__(self, n_channels, n_classes, ancillary_data_dim, bilinear=True):
        super(UNetWithUncertainty, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # U-Net Encoder (downsampling)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # U-Net Decoder (upsampling with dropout for epistemic uncertainty)
        self.up1 = Up(1024, 512 // factor, bilinear, use_dropout=True)
        self.up2 = Up(512, 256 // factor, bilinear, use_dropout=True)
        self.up3 = Up(256, 128 // factor, bilinear, use_dropout=True)
        self.up4 = Up(128, 64, bilinear, use_dropout=True)

        # Final output layer for logits
        self.outc = OutConv(64, n_classes)

        # Output layer for aleatoric uncertainty (log variance)
        self.log_variance = OutConv(64, n_classes)

        # FiLM Layer for modulating feature maps with ancillary data
        self.film = FiLM(input_dim=ancillary_data_dim, feature_map_channels=512)

    def forward(self, x, ancillary_data):
        # U-Net encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # FiLM modulation at bottleneck
        ancillary_data = ancillary_data.float()
        x4 = self.film(x4, ancillary_data)  # Modulate with crop calendar

        x5 = self.down4(x4)

        # U-Net decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output predictions (logits) and aleatoric uncertainty (log variance)
        logits = self.outc(x)
        log_var = self.log_variance(x)

        return logits, log_var


    

