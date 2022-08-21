import time

import numpy as np
import torch
import torch.nn as nn
import torchgan
import torch.nn.functional as F
from MBDiscLayer import MinibatchDiscrimination

LRelU_slope = 0.2


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz, num_of_chanells, features_generator):
        super(Generator, self).__init__()
        self.features_generator = features_generator
        self.num_of_chanells = num_of_chanells
        self.nz = nz
        self.ngpu = ngpu

        self.generator = nn.Sequential(
            self.G_Block_A(nz, features_generator * 32, 1, 1, 0),
            self.G_Block_A(features_generator * 32, features_generator * 16, 4, 4, 1),
            self.G_Block_A(features_generator * 16, features_generator * 8, 4, 4, 1),
            self.G_Block_A(features_generator * 8, features_generator * 4, 4, 4, 1),
            self.G_Block_A(features_generator * 4, features_generator * 2, 4, 4, 1),
            self.G_Block_A(features_generator * 2, features_generator, 4, 4, 1),
            nn.ConvTranspose2d(features_generator, num_of_chanells, 4, 4, 1),
            nn.Tanh(),  # [-1,1]
        )

    def G_Block_A(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU(LRelU_slope, inplace=True),  # TODO : Modify activation function
                             )

    # TODO : Create Adaptivepool layers Blocks
    # TODO : Add Dropout and residuals
    def forward(self, input):
        return self.generator(input)


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, num_of_chanells, features_discriminator):
        super(Discriminator, self).__init__()
        self.num_of_chanells = num_of_chanells
        self.features_discriminator = features_discriminator
        self.ngpu = ngpu

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.num_of_chanells, self.features_discriminator, 2, 2, 1, bias=False),
            nn.LeakyReLU(LRelU_slope, inplace=True),
            self.D_Block_A(self.features_discriminator, self.features_discriminator * 2, 4, 4, 1),
            self.D_Block_A(self.features_discriminator * 2, self.features_discriminator * 4, 4, 4, 1),
            self.D_Block_A(self.features_discriminator * 4, self.features_discriminator * 8, 4, 4, 1),
            self.D_Block_A(self.features_discriminator * 8, self.features_discriminator * 16, 4, 4, 1),
            self.D_Block_A(self.features_discriminator * 16, self.features_discriminator * 32, 3, 3, 1),
        )
        # Preventing of mode collapse by corelate many same data on output of sequential
        self.mbd = MinibatchDiscrimination(self.features_discriminator * 32, self.features_discriminator * 32, 1)
        self.c2d = nn.Conv2d(self.features_discriminator * 64, 1, 1, 1, 0, bias=False)

    def D_Block_A(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU(LRelU_slope, inplace=True),
                             )

    def forward(self, input):
        a = self.discriminator(input)
        b = self.mbd(a)
        c = self.c2d(b)
        x = torch.sigmoid(c)
        return x
