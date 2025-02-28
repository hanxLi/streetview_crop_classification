import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_blocks import *

### Original U-Net Model ###
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

class UNetWithFiLM(nn.Module):
    """
    U-Net model with FiLM layers for ancillary data (no uncertainty).
    """
    def __init__(self, n_channels, n_classes, ancillary_data_dim, bilinear=True, dropout_rate=None):
        super(UNetWithFiLM, self).__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(512, 256 // factor, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(128, 64, bilinear, dropout_rate=dropout_rate)

        # FiLM layer for ancillary data modulation
        self.film = FiLM(ancillary_data_dim, feature_map_channels=512)

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
    
class UNetWithAttention(nn.Module):
    def __init__(self, n_channels, n_classes, ancillary_data_dim, dropout_rate, bilinear=True, ):
        super(UNetWithAttention, self).__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor, dropout_rate=dropout_rate)  # Dropout only in the last encoder block

        # FiLM Layers for Modulation
        self.film1 = FiLM(ancillary_data_dim, 128)
        self.film2 = FiLM(ancillary_data_dim, 256)
        self.film3 = FiLM(ancillary_data_dim, 512)

        # Self-Attention Layer at the Bottleneck
        self.attention = SelfAttention(512, dropout_rate)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output Layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x, ancillary_data):
        # Encoder path with FiLM
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.film1(x2, ancillary_data)

        x3 = self.down2(x2)
        x3 = self.film2(x3, ancillary_data)

        x4 = self.down3(x3)
        x4 = self.film3(x4, ancillary_data)

        # Bottleneck with Self-Attention
        x5 = self.down4(x4)
        x5 = self.attention(x5)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output logits
        logits = self.outc(x)

        return logits

class UNetWithAttentionDeep(nn.Module):
    def __init__(self, n_channels, n_classes, ancillary_data_dim, dropout_rate, bilinear=True):
        super(UNetWithAttentionDeep, self).__init__()
        factor = 2 if bilinear else 1

        # Encoder (Deeper)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048 // factor, dropout_rate=dropout_rate)  # Additional depth, with dropout

        # FiLM Layers for Modulation
        self.film1 = FiLM(ancillary_data_dim, 256)  # FiLM at 256 channels
        self.film2 = FiLM(ancillary_data_dim, 512)  # FiLM at 512 channels
        self.film3 = FiLM(ancillary_data_dim, 1024)  # FiLM at 1024 channels

        # Self-Attention Layers at Bottleneck and in Decoder
        self.attention_bottleneck = SelfAttention(1024, dropout_rate)
        self.attention_decoder1 = SelfAttention(256, dropout_rate)  # Adjusted to 256 channels in decoder
        self.attention_decoder2 = SelfAttention(128, dropout_rate)  # Adjusted to 128 channels in decoder

        # Decoder (Deeper)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128 // factor, bilinear)
        self.up5 = Up(128, 64, bilinear)  # Final layer to match deeper encoder

        # Output Layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x, ancillary_data):
        # Encoder path with FiLM
        x1 = self.inc(x)                   # 64 channels
        x2 = self.down1(x1)                # 128 channels
        x3 = self.down2(x2)
        x3 = self.film1(x3, ancillary_data)  # First FiLM at 256 feature level

        x4 = self.down3(x3)
        x4 = self.film2(x4, ancillary_data)  # Second FiLM at 512 feature level

        x5 = self.down4(x4)
        x5 = self.film3(x5, ancillary_data)  # Third FiLM at 1024 feature level

        # Bottleneck with Self-Attention
        x6 = self.down5(x5)                # 2048 channels
        x6 = self.attention_bottleneck(x6)  # Attention at bottleneck

        # Decoder path with skip connections and additional attentions
        x = self.up1(x6, x5)               # 1024 channels
        x = self.up2(x, x4)
        x = self.attention_decoder1(x)     # Adjusted to apply attention at 256 feature level in decoder
        x = self.up3(x, x3)
        x = self.attention_decoder2(x)     # Adjusted to apply attention at 128 feature level in decoder
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        # Output logits
        logits = self.outc(x)

        return logits
