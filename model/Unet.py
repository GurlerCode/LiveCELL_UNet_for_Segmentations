import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Middle
        self.middle1 = self.conv_block(256, 512)
        self.middle2 = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.output_layer = nn.Conv2d(64, num_classes, kernel_size=1)

        # Add dropout layers
        self.dropout_middle1 = nn.Dropout2d(p=0.5)  # Adjust dropout probability as needed
        self.dropout_middle2 = nn.Dropout2d(p=0.5)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))

        # Middle
        middle = self.middle1(F.max_pool2d(enc3, kernel_size=2, stride=2))
        middle = self.dropout_middle1(middle)

        # Decoder
        dec3 = self.dec3(F.interpolate(middle, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = self.dropout_middle2(dec1)

        # Output layer
        output = self.output_layer(dec1)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)