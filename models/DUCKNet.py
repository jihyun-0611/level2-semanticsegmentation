"""
DUCKNet
code from https://github.com/brainer3220/TransDUCKNet/blob/master/model/DUCK_NET.py
"""
import logging
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.DUCKNet_block import ConvBlock2D

class DUCKNet(BaseModel):
    def __init__(self, config):
        super(DUCKNet, self).__init__()
        params = config.MODEL.PARAMS
        self.starting_filters = 34
        self.input_channels = params.IN_CHANNELS
        self.classes = params.OUT_CHANNELS

        # Define layers directly
        self.conv1 = nn.Conv2d(self.input_channels, self.starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(self.starting_filters * 2, self.starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(self.starting_filters * 4, self.starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(self.starting_filters * 8, self.starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(self.starting_filters * 16, self.starting_filters * 32, kernel_size=2, stride=2, padding=0)

        self.conv_block1 = ConvBlock2D('duckv2', self.input_channels, self.starting_filters, repeat=1)
        self.conv_block2 = ConvBlock2D('duckv2', self.starting_filters * 2, self.starting_filters * 2, repeat=1)
        self.conv_block3 = ConvBlock2D('duckv2', self.starting_filters * 4, self.starting_filters * 4, repeat=1)
        self.conv_block4 = ConvBlock2D('duckv2', self.starting_filters * 8, self.starting_filters * 8, repeat=1)
        self.conv_block5 = ConvBlock2D('duckv2', self.starting_filters * 16, self.starting_filters * 16, repeat=1)
        self.conv_block6 = ConvBlock2D('resnet', self.starting_filters * 32, self.starting_filters * 32, repeat=2, dilation_rate=2)
        self.conv_block7 = ConvBlock2D('resnet', self.starting_filters * 32, self.starting_filters * 16, repeat=2, dilation_rate=2)

        self.li_conv1 = nn.Conv2d(self.starting_filters, self.starting_filters * 2, kernel_size=2, stride=2, padding=0)

        self.upconv4 = nn.ConvTranspose2d(self.starting_filters * 16, self.starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.conv_block8 = ConvBlock2D('duckv2', self.starting_filters * 16, self.starting_filters * 16, repeat=1)

        self.upconv3 = nn.ConvTranspose2d(self.starting_filters * 16, self.starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.conv_block9 = ConvBlock2D('duckv2', self.starting_filters * 8, self.starting_filters * 4, repeat=1)

        self.upconv2 = nn.ConvTranspose2d(self.starting_filters * 4, self.starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.conv_block10 = ConvBlock2D('duckv2', self.starting_filters * 4, self.starting_filters * 2, repeat=1)

        self.upconv1 = nn.ConvTranspose2d(self.starting_filters * 2, self.starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.conv_block11 = ConvBlock2D('duckv2', self.starting_filters * 2, self.starting_filters, repeat=1)

        self.upconv0 = nn.ConvTranspose2d(self.starting_filters, self.starting_filters, kernel_size=2, stride=2, padding=0)
        self.conv_block12 = ConvBlock2D('duckv2', self.starting_filters, self.starting_filters, repeat=1)

        self.final_conv = nn.Conv2d(self.starting_filters, self.classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        t0 = self.conv_block1(x)
        l1i = self.li_conv1(t0)
        s1 = p1 + l1i
        t1 = self.conv_block2(s1)

        l2i = self.conv2(t1)
        s2 = p2 + l2i
        t2 = self.conv_block3(s2)

        l3i = self.conv3(t2)
        s3 = p3 + l3i
        t3 = self.conv_block4(s3)

        l4i = self.conv4(t3)
        s4 = p4 + l4i
        t4 = self.conv_block5(s4)

        l5i = self.conv5(t4)
        s5 = p5 + l5i
        t51 = self.conv_block6(s5)
        t53 = self.conv_block7(t51)

        # Decoder
        l5o = self.upconv4(t53)
        c4 = l5o + t4
        q4 = self.conv_block8(c4)

        l4o = self.upconv3(q4)
        c3 = l4o + t3
        q3 = self.conv_block9(c3)

        l3o = self.upconv2(q3)
        c2 = l3o + t2
        q2 = self.conv_block10(c2)

        l2o = self.upconv1(q2)
        c1 = l2o + t1
        q1 = self.conv_block11(c1)

        l1o = self.upconv0(q1)
        c0 = l1o + t0
        z1 = self.conv_block12(c0)

        output = self.final_conv(z1)

        return output

    def get_model(self):
        return self