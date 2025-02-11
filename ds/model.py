import torch
import torch.nn as nn

class LHUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        # Decoder (Upsampling with skip connections)
        self.dec1 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Decoder
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        # Final layer
        return self.final(d2)
