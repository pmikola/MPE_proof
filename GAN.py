import time

import numpy as np
import torch
import torch.nn as nn
import torchgan
import torch.nn.functional as F
from MBDiscBlock import MinibatchDiscrimination
from ResidualBlockD import ResidualBlockD
from ResidualBlockG import ResidualBlockG

LRelU_slope = 0.2


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz, num_of_chanells, features_generator, resolution, gen_output):
        super(Generator, self).__init__()
        self.gen_output = gen_output
        self.features_generator = features_generator
        self.num_of_chanells = num_of_chanells
        self.nz = nz
        self.ngpu = ngpu
        self.resolution = resolution
        self.generator_input = self.G_Block_A(nz, self.features_generator * 64, 1, 1, 0)
        self.GBA0 = self.G_Block_A(self.features_generator * 64, self.features_generator * 32, 4, 4, 1)
        # self.ResNet_GA1 = self.G_Block_Residual(self.features_generator * 32, self.features_generator * 32, 1, 1, 0)
        self.GBA1 = self.G_Block_A(self.features_generator * 32, self.features_generator * 16, 4, 4, 1)
        # self.GBB1 = self.G_Block_B(self.features_generator * 32, self.features_generator * 16, 2, 2, 1)
        # self.ResNet_GA2 = self.G_Block_Residual(self.features_generator * 16, self.features_generator * 16, 2, 2, 1)
        #self.GBA2 = self.G_Block_A(self.features_generator * 16, self.features_generator * 8, 4, 4, 1)
        self.GBB2 = self.G_Block_B(self.features_generator * 16, self.features_generator * 8, 2, 2, 1)
        # self.ResNet_GA3 = self.G_Block_Residual(self.features_generator * 8, self.features_generator * 8, 1, 1, 0)
        # self.GBA3 = self.G_Block_A(self.features_generator * 8, self.features_generator * 4, 4, 4, 1)
        self.GBB3 = self.G_Block_B(self.features_generator * 8, self.features_generator * 4, 2, 2, 1)
        # self.ResNet_GA4 = self.G_Block_Residual(self.features_generator * 4, self.features_generator * 4, 1, 1, 0)
        # self.GBA4 = self.G_Block_A(self.features_generator * 4, self.features_generator * 2, 4, 4, 1)
        self.GBB4 = self.G_Block_B(self.features_generator * 4, self.features_generator * 2, 2, 2, 1)
        # self.ResNet_GA5 = self.G_Block_Residual(self.features_generator * 2, self.features_generator * 2, 1, 1, 0)
        # self.GBA5 = self.G_Block_A(self.features_generator * 2, self.gen_output, 4, 4, 1)
        self.GBB5 = self.G_Block_B(self.features_generator * 2, self.gen_output, 2, 2, 1)
        # self.ct2d = nn.ConvTranspose2d(self.gen_output, num_of_chanells, 4, 4, 1)
        # self.GBB6 = self.G_Block_B(self.gen_output, num_of_chanells, 1, 1, 0)
        self.GBB6 = nn.Sequential(
            nn.UpsamplingBilinear2d(size=None, scale_factor=2),
            nn.Conv2d(self.gen_output, num_of_chanells, 3, 3, 0, bias=False),
        )

    def G_Block_A(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.2, inplace=False),
            # nn.FeatureAlphaDropout(p=0.1, inplace=False),
            nn.Tanh(),
            # nn.ReLU(inplace=False),
            # nn.Hardtanh(-2, 2),
            # nn.Softsign(),
        )

    def G_Block_B(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.UpsamplingBilinear2d(size=None, scale_factor=4),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.2, inplace=False),
            nn.Tanh(),
        )

    def G_Block_Residual(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(ResidualBlockG(in_channels, out_channels, kernel, stride, padding),
                             # ResidualBlockG(out_channels, out_channels, kernel, stride, padding),
                             # ResidualBlockG(out_channels, out_channels, kernel, stride, padding),
                             )

    def resize_output(self, input, res, gen_output):
        out = torch.reshape(torch.squeeze(F.interpolate(input, size=res)), (gen_output, 1, res, res))
        return out

    def forward(self, input):
        a = self.generator_input(input)
        b = self.GBA0(a)
        # a_res = self.ResNet_GA1(a)
        c = self.GBA1(b)
        # b_res = self.ResNet_GA2(b)
        d = self.GBB2(c)
        # c_res = self.ResNet_GA3(c)
        e = self.GBB3(d)
        # d_res = self.ResNet_GA4(d)
        f = self.GBB4(e)
        # e_res = self.ResNet_GA5(e)
        g = self.GBB5(f)
        # h = self.ct2d(f)
        h = self.GBB6(g)
        # print(h.shape)
        i = self.resize_output(h, self.resolution, self.gen_output)
        out = torch.tanh(i)

        return out


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, num_of_chanells, features_discriminator):
        super(Discriminator, self).__init__()
        self.num_of_chanells = num_of_chanells
        self.features_discriminator = features_discriminator
        self.ngpu = ngpu
        self.discriminator_input = nn.Sequential(
            nn.Conv2d(self.num_of_chanells, self.features_discriminator, 2, 2, 1, bias=False),
            nn.LeakyReLU(LRelU_slope, inplace=True),
        )
        # self.ResNet_DA1 = self.D_Block_Residual(features_discriminator, features_discriminator, 1, 1, 0)
        self.DBA1 = self.D_Block_A(self.features_discriminator, self.features_discriminator * 2, 2, 1, 1)
        # self.ResNet_DA2 = self.D_Block_Residual(features_discriminator * 2, features_discriminator * 2, 1, 1, 0)
        self.DBA2 = self.D_Block_A(self.features_discriminator * 2, self.features_discriminator * 4, 1, 2, 1)
        # self.ResNet_DA3 = self.D_Block_Residual(features_discriminator * 4, features_discriminator * 4, 1, 1, 0)
        self.DBA3 = self.D_Block_A(self.features_discriminator * 4, self.features_discriminator * 8, 4, 4, 1)
        # self.ResNet_DA4 = self.D_Block_Residual(features_discriminator * 8, features_discriminator * 8, 1, 1, 0)
        self.DBA4 = self.D_Block_A(self.features_discriminator * 8, self.features_discriminator * 16, 4, 4, 1)
        # self.ResNet_DA5 = self.D_Block_Residual(features_discriminator * 16, features_discriminator * 16, 1, 1, 0)
        self.DBA5 = self.D_Block_A(self.features_discriminator * 16, self.features_discriminator * 32, 3, 3, 1)
        # self.ResNet_DA6 = self.D_Block_Residual(features_discriminator * 32, features_discriminator * 32, 1, 1, 0)

        # Preventing of mode collapse by corelate many same data on output of sequential
        self.mbd = MinibatchDiscrimination(self.features_discriminator * 32, self.features_discriminator * 32, 1)
        self.c2d = nn.Conv2d(self.features_discriminator * 64, 1, 1, 1, 0, bias=True)

    def D_Block_A(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                             nn.BatchNorm2d(out_channels),
                             # Critic in WGAN-GP preferr LayerNorm over BatchNorm or without any norm scheme
                             nn.Dropout(p=0.1, inplace=False),
                             nn.LeakyReLU(LRelU_slope, inplace=True),
                             )

    def D_Block_Residual(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(ResidualBlockD(in_channels, out_channels, kernel, stride, padding),
                             )

    def forward(self, input):
        a = self.discriminator_input(input)
        # b_res = self.ResNet_DA1(a)
        b = self.DBA1(a)
        # c_res = self.ResNet_DA2(b)
        c = self.DBA2(b)
        # d_res = self.ResNet_DA3(c)
        d = self.DBA3(c)
        # e_res = self.ResNet_DA4(d)
        e = self.DBA4(d)
        # f_res = self.ResNet_DA5(e)
        f = self.DBA5(e)
        # g_res = self.ResNet_DA6(f)
        g = self.mbd(f)
        h = self.c2d(g)
        # i = self.discriminator_output(h)
        out = torch.sigmoid(h)
        return out

# # Discriminator Code
# class Discriminator(nn.Module):
#     def __init__(self, ngpu, num_of_chanells, features_discriminator):
#         super(Discriminator, self).__init__()
#         self.num_of_chanells = num_of_chanells
#         self.features_discriminator = features_discriminator
#         self.ngpu = ngpu
#         self.discriminator_input = nn.Sequential(
#             nn.Conv2d(self.num_of_chanells, self.features_discriminator, 2, 2, 1, bias=False),
#             nn.LeakyReLU(LRelU_slope, inplace=True),
#         )
#         #self.ResNet_DA1 = self.D_Block_Residual(features_discriminator, features_discriminator, 1, 1, 0)
#         self.DBA1 = self.D_Block_A(self.features_discriminator, self.features_discriminator * 2, 4, 4, 1)
#         #self.ResNet_DA2 = self.D_Block_Residual(features_discriminator * 2, features_discriminator * 2, 1, 1, 0)
#         self.DBA2 = self.D_Block_A(self.features_discriminator * 2, self.features_discriminator * 4, 4, 4, 1)
#         #self.ResNet_DA3 = self.D_Block_Residual(features_discriminator * 4, features_discriminator * 4, 1, 1, 0)
#         self.DBA3 = self.D_Block_A(self.features_discriminator * 4, self.features_discriminator * 8, 4, 4, 1)
#         #self.ResNet_DA4 = self.D_Block_Residual(features_discriminator * 8, features_discriminator * 8, 1, 1, 0)
#         self.DBA4 = self.D_Block_A(self.features_discriminator * 8, self.features_discriminator * 16, 4, 4, 1)
#         #self.ResNet_DA5 = self.D_Block_Residual(features_discriminator * 16, features_discriminator * 16, 1, 1, 0)
#         self.DBA5 = self.D_Block_A(self.features_discriminator * 16, self.features_discriminator * 32, 3, 3, 1)
#         #self.ResNet_DA6 = self.D_Block_Residual(features_discriminator * 32, features_discriminator * 32, 1, 1, 0)
#
#         # Preventing of mode collapse by corelate many same data on output of sequential
#         self.mbd = MinibatchDiscrimination(self.features_discriminator * 32, self.features_discriminator * 32, 1)
#         self.c2d = nn.Conv2d(self.features_discriminator * 64, 1, 1, 1, 0, bias=True)
#         # self.discriminator_output = nn.Sequential(
#         #     nn.LeakyReLU(LRelU_slope, inplace=True),
#         #     nn.Dropout(p=0.2, inplace=False),
#         #     nn.Linear(1, self.features_discriminator),
#         #     nn.LeakyReLU(LRelU_slope, inplace=True),
#         #     nn.Dropout(p=0.1, inplace=False),
#         #     nn.Linear(self.features_discriminator, 1),
#         #
#         # )
#
#     def D_Block_A(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#                              nn.BatchNorm2d(out_channels), # Critic in WGAN-GP preferr LayerNorm over BatchNorm or without any norm scheme
#                              nn.Dropout(p=0.1, inplace=False),
#                              nn.LeakyReLU(LRelU_slope, inplace=True),
#                              )
#
#     def D_Block_Residual(self, in_channels, out_channels, kernel, stride, padding):
#         return nn.Sequential(ResidualBlockD(in_channels, out_channels, kernel, stride, padding),
#                              )
#
#     def forward(self, input):
#         a = self.discriminator_input(input)
#         #b_res = self.ResNet_DA1(a)
#         b = self.DBA1(a)
#         #c_res = self.ResNet_DA2(b)
#         c = self.DBA2(b)
#         #d_res = self.ResNet_DA3(c)
#         d = self.DBA3(c)
#         #e_res = self.ResNet_DA4(d)
#         e = self.DBA4(d)
#         #f_res = self.ResNet_DA5(e)
#         f = self.DBA5(e)
#         #g_res = self.ResNet_DA6(f)
#         g = self.mbd(f)
#         h = self.c2d(g)
#         #i = self.discriminator_output(h)
#         out = torch.sigmoid(h)
#         return out
#
# # Generator Code
# class Generator(nn.Module):
#     def __init__(self, ngpu, nz, num_of_chanells, features_generator, resolution):
#         super(Generator, self).__init__()
#         self.features_generator = features_generator
#         self.num_of_chanells = num_of_chanells
#         self.nz = nz
#         self.ngpu = ngpu
#         self.resolution = resolution
#         self.generator_input = self.G_Block_A(nz, self.features_generator * 32, 1, 1, 0)
#         self.ResNet_GA1 = self.G_Block_Residual(self.features_generator * 32, self.features_generator * 32, 1, 1, 0)
#         self.GBA1 = self.G_Block_A(self.features_generator * 32, self.features_generator * 16, 4, 4, 1)
#         self.ResNet_GA2 = self.G_Block_Residual(self.features_generator * 16, self.features_generator * 16, 2, 2, 1)
#         self.GBA2 = self.G_Block_A(self.features_generator * 16, self.features_generator * 8, 4, 4, 1)
#         self.ResNet_GA3 = self.G_Block_Residual(self.features_generator * 8, self.features_generator * 8, 1, 1, 0)
#         self.GBA3 = self.G_Block_A(self.features_generator * 8, self.features_generator * 4, 4, 4, 1)
#         self.ResNet_GA4 = self.G_Block_Residual(self.features_generator * 4, self.features_generator * 4, 1, 1, 0)
#         self.GBA4 = self.G_Block_A(self.features_generator * 4, self.features_generator * 2, 4, 4, 1)
#         self.ResNet_GA5 = self.G_Block_Residual(self.features_generator * 2, self.features_generator * 2, 1, 1, 0)
#         self.GBA5 = self.G_Block_A(self.features_generator * 2, self.features_generator, 4, 4, 1)
#         nn.Tanh(),
#         self.dout = nn.Dropout(p=0.1, inplace=False)
#         self.ct2d = nn.ConvTranspose2d(features_generator, num_of_chanells, 4, 4, 1)
#
#     def G_Block_A(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#                              nn.BatchNorm2d(out_channels),
#                              nn.Dropout(p=0.1, inplace=False),
#                              # nn.FeatureAlphaDropout(p=0.1, inplace=False),
#                              nn.Tanh(),
#                              # nn.ReLU(inplace=False),
#                              # nn.Hardtanh(-2, 2),
#                              # nn.Softsign(),
#                              )
#
#     def G_Block_Residual(self, in_channels, out_channels, kernel, stride, padding):
#         return nn.Sequential(ResidualBlockG(in_channels, out_channels, kernel, stride, padding),
#                              ResidualBlockG(out_channels, out_channels, kernel, stride, padding),
#                              ResidualBlockG(out_channels, out_channels, kernel, stride, padding),
#                              )
#
#     def resize_output(self, input, res, features):
#         out = torch.reshape(torch.squeeze(F.interpolate(input, size=res)), (features, 1, res, res))
#         return out
#
#     def forward(self, input):
#         a = self.generator_input(input)
#         a_res = self.ResNet_GA1(a)
#         b = self.GBA1(a_res)
#         b_res = self.ResNet_GA2(b)
#         c = self.GBA2(b_res)
#         c_res = self.ResNet_GA3(c)
#         d = self.GBA3(c_res)
#         d_res = self.ResNet_GA4(d)
#         e = self.GBA4(d_res)
#         e_res = self.ResNet_GA5(e)
#         f = self.GBA5(e_res)
#         g = self.dout(f)
#         h = self.ct2d(g)
#         i = torch.tanh(h)
#         out = self.resize_output(i, self.resolution, self.features_generator)
#         return out
