import torch
import torch.nn as nn
import torch.nn.functional as F

### FiLM Layer ###
class FiLM(nn.Module):
    """
    FiLM Layer: Generates gamma and beta parameters to modulate feature maps.
    Handles one-hot encoded `ancillary_data` efficiently using `nn.Embedding`.
    """
    def __init__(self, num_classes, feature_map_channels):
        super(FiLM, self).__init__()
        self.embedding = nn.Embedding(num_classes, feature_map_channels)
        self.gamma = nn.Linear(feature_map_channels, feature_map_channels)
        self.beta = nn.Linear(feature_map_channels, feature_map_channels)

    def forward(self, feature_map, ancillary_data):
        """
        Modulate feature map with FiLM parameters.

        Args:
            feature_map (Tensor): Shape [B, C, H, W]
            ancillary_data (Tensor): One-hot encoded input, Shape [B, num_classes]

        Returns:
            Tensor: Modulated feature map.
        """
        assert ancillary_data.dim() == 2, f"Expected ancillary_data shape [B, num_classes], got {ancillary_data.shape}"

        # Convert one-hot to class index for embedding lookup
        embedded_ancillary = self.embedding(ancillary_data.argmax(dim=1))  # Shape [B, feature_map_channels]

        gamma = self.gamma(embedded_ancillary).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = self.beta(embedded_ancillary).unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]

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
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention) 
        # Apply attention to the value matrix
        out = torch.bmm(attention, value)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

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
