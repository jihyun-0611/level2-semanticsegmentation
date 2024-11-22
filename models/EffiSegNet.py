"""
EffiSegNet
code from https://github.com/ivezakis/effisegnet/blob/main/models/effisegnet.py
"""

import math
import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBNFeatures
from monai.networks.nets.efficientnet import get_efficientnet_image_size
from models.base_model import BaseModel

import math
import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBNFeatures
from monai.networks.nets.efficientnet import get_efficientnet_image_size
from models.base_model import BaseModel

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.BatchNorm2d(init_channels),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.BatchNorm2d(new_channels),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class EffiSegNet(BaseModel):
    def __init__(self, config):
        super().__init__()
        
        # Get parameters from config
        params = config.MODEL.PARAMS
        self.input_channels = params.IN_CHANNELS
        self.output_channels = params.OUT_CHANNELS
        self.starting_filters = 64  # Could be added to config if needed
        self.model_name = "efficientnet-b0"  # Could be added to config if needed
        
        # Initialize encoder
        self.encoder = EfficientNetBNFeatures(
            model_name=self.model_name,
            pretrained=True
        )
        
        # Remove unused layers
        del self.encoder._avg_pooling
        del self.encoder._dropout
        del self.encoder._fc

        # Get channel sizes based on model version
        b = int(self.model_name[-1])
        num_channels_per_output = [
            (16, 24, 40, 112, 320),
            (16, 24, 40, 112, 320),
            (16, 24, 48, 120, 352),
        ][min(b, 2)]

        # Initialize decoder components - conv layers and batch norm
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(ch_in, self.starting_filters, kernel_size=3, stride=1, padding=1, bias=False)
            for ch_in in num_channels_per_output
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(self.starting_filters) for _ in range(5)
        ])

        # Final layers
        self.relu = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(self.starting_filters)
        self.ghost1 = GhostModule(self.starting_filters, self.starting_filters)
        self.ghost2 = GhostModule(self.starting_filters, self.starting_filters)
        self.conv6 = nn.Conv2d(self.starting_filters, self.output_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # Get input size for upsampling target
        target_size = (x.size(2), x.size(3))
        
        # Get features from encoder
        features = self.encoder(x)
        
        processed_features = []
        for i, feat in enumerate(features):
            # Apply convolution and activation
            x = self.conv_layers[i](feat)
            x = self.relu(x)
            x = self.bn_layers[i](x)
            
            # Upsample to target size (input image size)
            x = nn.functional.interpolate(x, size=target_size, mode='nearest')
            processed_features.append(x)

        # Combine features
        x = sum(processed_features)
        
        # Apply final layers
        x = self.bn6(x)
        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.conv6(x)

        return x

    def get_model(self):
        return self