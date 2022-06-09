import torch as t
from torch import nn
import numpy as np


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        
        self.unit_layers = nn.Sequential(
            NewConv(3, 32, kernel_size=9, stride=1),
            #nn.Batchnorm2d(32)
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            NewConv(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64),
            #nn.Batchnorm2d(64)
            nn.ReLU(inplace=True),
            NewConv(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128),
            #nn.Batchnorm2d(128)
            nn.ReLU(inplace=True),
        )

        
        self.residual_block = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        
        self.up_layer = nn.Sequential(
            UpsampleNewConv(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(64),
            #nn.Batchnorm2d(64)
            nn.ReLU(inplace=True),
            UpsampleNewConv(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(32),
            #nn.Batchnorm2d(128)
            nn.ReLU(inplace=True),
            NewConv(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        out = self.unit_layers(x)
        out= self.residual_block(out)
        out= self.up_layer(out)
        return out


class NewConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(NewConv, self).__init__()
        n_pad= int(np.floor(kernel_size / 2))
        self.ref_pad = nn.ReflectionPad2d(ref_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.ref_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleNewConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleNewConv, self).__init__()
        self.upsample = upsample
        n_pad= int(np.floor(kernel_size / 2))
        self.ref_pad = nn.ReflectionPad2d(ref_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = t.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.ref_pad(x_in)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = NewConv(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = NewConv(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + res