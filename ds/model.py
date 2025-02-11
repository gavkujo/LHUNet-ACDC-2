import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spatial attention
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class HybridAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_attention = SpatialAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        # Apply spatial and channel attention
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        return x

class LHUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, init_features=32):
        super().__init__()
        features = init_features

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, features)
        self.enc2 = self.conv_block(features, features * 2)
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.enc4 = self.conv_block(features * 4, features * 8)

        # Hybrid Attention Blocks
        self.attention1 = HybridAttentionBlock(features)
        self.attention2 = HybridAttentionBlock(features * 2)
        self.attention3 = HybridAttentionBlock(features * 4)
        self.attention4 = HybridAttentionBlock(features * 8)

        # Decoder (Upsampling with skip connections)
        self.dec1 = self.upconv_block(features * 8, features * 4)
        self.dec2 = self.upconv_block(features * 4, features * 2)
        self.dec3 = self.upconv_block(features * 2, features)
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # (B, 32, H/2, W/2)
        e1 = self.attention1(e1)
        e2 = self.enc2(e1)  # (B, 64, H/4, W/4)
        e2 = self.attention2(e2)
        e3 = self.enc3(e2)  # (B, 128, H/8, W/8)
        e3 = self.attention3(e3)
        e4 = self.enc4(e3)  # (B, 256, H/16, W/16)
        e4 = self.attention4(e4)

        # Decoder with skip connections
        d1 = self.dec1(e4)  # (B, 128, H/8, W/8)
        d1 = d1 + e3  # Skip connection
        d2 = self.dec2(d1)  # (B, 64, H/4, W/4)
        d2 = d2 + e2  # Skip connection
        d3 = self.dec3(d2)  # (B, 32, H/2, W/2)
        d3 = d3 + e1  # Skip connection

        # Final layer
        output = self.final(d3)  # (B, out_channels, H, W)
        return output
