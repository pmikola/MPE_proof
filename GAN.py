import torch.nn as nn


# DCGAN ARCHITECTURE https://arxiv.org/pdf/1511.06434.pdf

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz,  batch_size):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.nz = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.batch_size, self.nz, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                               bias=False),
            nn.BatchNorm1d(self.nz),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.nz, self.nz * 5, kernel_size=(5, 5), stride=(5,5), padding=(0, 0),
                               bias=False),
            nn.BatchNorm1d(self.nz * 5),
            nn.ReLU(True),
            nn.Conv2d(self.nz * 5, int(self.nz * 2.5), kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm1d(int(self.nz * 2.5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(self.nz * 2.5), self.batch_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, batch_size):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.ngpu = ngpu
        self.nc = nc
        self.main = nn.Sequential(
            nn.Conv2d(self.batch_size, self.nc, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc, self.nc * 2, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc*64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc * 2, self.nc * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc * 4, self.nc * 8, kernel_size=(12, 12), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc * 8, self.nc * 16, kernel_size=(8, 8), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc * 16, self.nc * 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc * 32, self.nc * 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc * 64, self.batch_size, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
