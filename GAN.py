import torch.nn as nn


# DCGAN ARCHITECTURE https://arxiv.org/pdf/1511.06434.pdf

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.ngpu = ngpu
        self.nc = nc
        # self.structure = structure
        # self.meta = meta
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(1, 10, (250, 124750), bias=False),
            nn.ConvTranspose2d(1, 10, (250, 124750)),
            # nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(self.ngf * 8),
            # nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf * 4),
            # nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf * 2),
            # nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf),
            # nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.nc = nc
        # self.fields = fields
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(10, 1, (250, 124750)),
            # nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 32 x 32
            # nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 16 x 16
            # nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
