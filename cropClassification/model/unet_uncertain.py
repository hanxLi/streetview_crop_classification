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

class SelfAttention(nn.Module):
    """
    Self-Attention Layer with dropout for regularization.
    """
    def __init__(self, in_channels, dropout_rate):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor
        self.dropout = nn.Dropout(dropout_rate)  # Dropout applied to attention output

    def forward(self, x):
        B, C, H, W = x.size()

        # Compute query, key, and value matrices
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C/8]
        key = self.key(x).view(B, -1, H * W)  # [B, C/8, HW]
        value = self.value(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C]

        # Compute attention weights and apply softmax
        attention = torch.bmm(query, key)  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)

        # Apply attention to the value matrix
        out = torch.bmm(attention, value)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        # Apply dropout to the output of the self-attention layer
        out = self.dropout(out)

        # Apply scaling with gamma and add the input (residual connection)
        out = self.gamma * out + x

        return out


### DoubleConv Block ###
class DoubleConv(nn.Module):
    """
    Double convolution block: (conv => BN => ReLU) * 2 with optional dropout.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate:
            layers.append(nn.Dropout2d(p=dropout_rate))  # Optional dropout for stochasticity

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

### Down Block (Encoder) ###
class Down(nn.Module):
    """
    Downscaling block with maxpool followed by double convolution.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

### Up Block (Decoder) with Dropout ###
class Up(nn.Module):
    """
    Upscaling block with optional dropout to introduce uncertainty.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

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
    def __init__(self, n_channels, n_classes, ancillary_data_dim, bilinear=True, dropout_rate=None):
        super(UNetWithUncertainty, self).__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder with dropout for epistemic uncertainty
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(512, 256 // factor, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(128, 64, bilinear, dropout_rate=dropout_rate)

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
        self.up4 = Up(128, 64, bilinear, use_dropout=dropout_rate)

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
