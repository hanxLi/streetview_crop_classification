import torch
import torch.nn as nn
import torch.nn.functional as F

### FiLM Layer ###
class FiLM(nn.Module):
    """
    FiLM Layer: Generates gamma and beta parameters to modulate the feature maps.
    """
    def __init__(self, input_dim, feature_map_channels):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(input_dim, feature_map_channels)
        self.beta = nn.Linear(input_dim, feature_map_channels)

    def forward(self, feature_map, ancillary_data):
        """
        Modulate feature map with FiLM parameters.
        Args:
            feature_map (Tensor): Shape [B, C, H, W]
            ancillary_data (Tensor): Shape [B, input_dim]
        Returns:
            Tensor: Modulated feature map.
        """
        gamma = self.gamma(ancillary_data).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = self.beta(ancillary_data).unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        return gamma * feature_map + beta

### DoubleConv Block ###
class DoubleConv(nn.Module):
    """
    Double convolution block: (conv => BN => ReLU) * 2 with optional dropout.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(p=0.1))  # Optional dropout for stochasticity

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

### Down Block (Encoder) ###
class Down(nn.Module):
    """
    Downscaling block with maxpool followed by double convolution.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_dropout=use_dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

### Up Block (Decoder) with Dropout ###
class Up(nn.Module):
    """
    Upscaling block with optional dropout to introduce uncertainty.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, use_dropout=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, use_dropout=use_dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ensure dimensions match by padding if necessary
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

### Output Convolution Layer ###
class OutConv(nn.Module):
    """
    Output convolution layer for logits and log variance.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

### U-Net Model with FiLM and Uncertainty ###
class UNetWithUncertainty(nn.Module):
    """
    U-Net model with FiLM layers for ancillary data and uncertainty estimation.
    """
    def __init__(self, n_channels, n_classes, ancillary_data_dim, bilinear=True, use_dropout=True):
        super(UNetWithUncertainty, self).__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder with dropout for epistemic uncertainty
        self.up1 = Up(1024, 512 // factor, bilinear, use_dropout=use_dropout)
        self.up2 = Up(512, 256 // factor, bilinear, use_dropout=use_dropout)
        self.up3 = Up(256, 128 // factor, bilinear, use_dropout=use_dropout)
        self.up4 = Up(128, 64, bilinear, use_dropout=use_dropout)

        # FiLM layer for modulating features with ancillary data
        self.film = FiLM(input_dim=ancillary_data_dim, feature_map_channels=512)

        # Output layers for logits and log variance
        self.logits_layer = OutConv(64, n_classes)
        self.log_var_layer = OutConv(64, n_classes)

    def forward(self, x, ancillary_data):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # FiLM modulation at bottleneck
        ancillary_data = ancillary_data.float()
        x4 = self.film(x4, ancillary_data)

        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Separate outputs for logits and log variance
        logits = self.logits_layer(x)
        log_var = self.log_var_layer(x)
        
        return logits, log_var

class UNetWithFiLM(nn.Module):
    """
    U-Net model with FiLM layers for ancillary data (no uncertainty).
    """
    def __init__(self, n_channels, n_classes, ancillary_data_dim, bilinear=True, use_dropout=False):
        super(UNetWithFiLM, self).__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear, use_dropout=use_dropout)
        self.up2 = Up(512, 256 // factor, bilinear, use_dropout=use_dropout)
        self.up3 = Up(256, 128 // factor, bilinear, use_dropout=use_dropout)
        self.up4 = Up(128, 64, bilinear, use_dropout=use_dropout)

        # FiLM layer for ancillary data modulation
        self.film = FiLM(input_dim=ancillary_data_dim, feature_map_channels=512)

        # Output layer for logits
        self.outc = OutConv(64, n_classes)

    def forward(self, x, ancillary_data):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # FiLM modulation at bottleneck
        ancillary_data = ancillary_data.float()
        x4 = self.film(x4, ancillary_data)

        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output logits
        logits = self.outc(x)

        return logits