"""
custom model = [encoder : Efficientnet-b0, decoder : DUCKNet]
code from https://github.com/brainer3220/TransDUCKNet/blob/master/model/DUCK_NET.py
"""
import logging
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.DUCKNet_block import ConvBlock2D
import segmentation_models_pytorch as smp

class DUCKNetDecoder(BaseModel):
    def __init__(self, config):
        super(DUCKNetDecoder, self).__init__()

        # SMP Encoder 설정
        self.encoder = smp.Unet(encoder_name="timm-efficientnet-b8", encoder_weights="imagenet").encoder
        encoder_channels = self.encoder.out_channels


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
        # self.conv_block6 = ConvBlock2D('resnet', self.starting_filters * 32, self.starting_filters * 32, repeat=2, dilation_rate=2)
        self.conv_block6 = ConvBlock2D('resnet', encoder_channels[-1], self.starting_filters * 32, repeat=2, dilation_rate=2)

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

        self.upchannel1 = nn.Conv2d(in_channels=encoder_channels[4], out_channels=544, kernel_size=1, stride=1, padding=0)
        self.upchannel2 = nn.Conv2d(in_channels=encoder_channels[3], out_channels=272, kernel_size=1, stride=1, padding=0)
        self.upchannel3 = nn.Conv2d(in_channels=encoder_channels[2], out_channels=136, kernel_size=1, stride=1, padding=0)
        self.upchannel4 = nn.Conv2d(in_channels=encoder_channels[1], out_channels=68, kernel_size=1, stride=1, padding=0)
        self.upchannel5 = nn.Conv2d(in_channels=encoder_channels[0], out_channels=34, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # SMP 인코더 출력
        encoder_features = self.encoder(x)
        
        t0 = encoder_features[0]  # [2, 3, 512, 512] , 배치, 채널, h, w
        t1 = encoder_features[1]  # [2, 32, 256, 256]
        t2 = encoder_features[2]  # [2, 24, 128, 128]
        t3 = encoder_features[3]  # [2, 40, 64, 64]
        t4 = encoder_features[4]  # [2, 112, 32, 32]
        t5 = encoder_features[-1] # [2, 320, 16, 16]
        
        # DUCKNet 디코더 블록
        t51 = self.conv_block6(t5) # [2, 1088, 16, 16]

        t53 = self.conv_block7(t51) # [2, 544, 16, 16]

        # 디코더 부분 출력
        l5o = self.upconv4(t53) # [2, 544, 32, 32]


        t4 = self.upchannel1(t4) # t4의 채널은 114, l5o의 채널은 554 이므로 맞춰줌.
        c4 = l5o + t4
        q4 = self.conv_block8(c4) # [2, 544, 32, 32]

        l4o = self.upconv3(q4) # [2, 272, 64, 64]


        t3 = self.upchannel2(t3) # t3의 채널은 40, l4o의 채널은 272 이므로 맞춰줌.
        c3 = l4o + t3
        q3 = self.conv_block9(c3) # [2, 136, 64, 64]

        l3o = self.upconv2(q3) # [2, 136, 128, 128]

        t2 = self.upchannel3(t2) # t2의 채널은 24, l3o의 채널은 136 이므로 맞춰줌.
        c2 = l3o + t2
        q2 = self.conv_block10(c2) # [2, 68, 128, 128]

        l2o = self.upconv1(q2) # [2, 68, 256, 256]

        t1 = self.upchannel4(t1) # t1의 채널은 32, l2o의 채널은 68 이므로 맞춰줌.
        c1 = l2o + t1
        q1 = self.conv_block11(c1) # [2, 34, 256, 256]

        l1o = self.upconv0(q1) # [2, 34, 512, 512]

        t0 = self.upchannel5(t0) # t0의 채널은 3, l1o의 채널은 34 이므로 맞춰줌.
        c0 = l1o + t0
        z1 = self.conv_block12(c0)

        output = self.final_conv(z1)

        return output

    def get_model(self):
        return self
    