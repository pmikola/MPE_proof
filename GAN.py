import torch.nn as nn

# DCGAN variation

LRelU_slope = 0.2


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz, batch_size):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.nz = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # nn.ConvTranspose2d(self.batch_size, self.nz, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            #                    bias=False),
            # nn.BatchNorm1d(self.nz),
            # nn.Softsign(),
            # nn.AdaptiveAvgPool2d((int(self.nz / 10), (int(self.nz / 10)))),
            # nn.ConvTranspose2d(self.nz, int(self.nz / 5), kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
            #                    bias=False),
            # nn.BatchNorm1d(int(self.nz / 5)),
            # nn.Softsign(),
            # nn.AdaptiveAvgPool2d((int(self.nz / 5), (int(self.nz)))),
            # nn.ConvTranspose2d(int(self.nz / 5), int(self.nz), kernel_size=(5, 5), stride=(5, 5), padding=(0, 0),
            #                    bias=False),
            # nn.BatchNorm1d(int(self.nz)),
            # nn.Softsign(),
            # nn.AdaptiveAvgPool2d(self.nz * 2),
            # nn.ConvTranspose2d(self.nz, int(self.nz * 2), kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
            #                    bias=False),
            # nn.BatchNorm1d(int(self.nz * 4)),
            # nn.Softsign(),
            # # nn.Threshold(0, 0),
            # # nn.Hardtanh(0, 1),
            # nn.AdaptiveMaxPool2d(250),
            # nn.ConvTranspose2d(self.nz * 2, self.batch_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            #                    bias=False),
            # #nn.Softsign(),
            # nn.Threshold(0, 0),
            # nn.Hardtanh(0, 1),
            # nn.Softmax(),

            nn.Conv2d(self.batch_size, self.nz, kernel_size=(5, 5), stride=(self.nz, self.nz), padding=(1, 1),
                      bias=False),
            # nn.Softsign(),
            #nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.ConvTranspose2d(self.nz, int(self.nz / 100), kernel_size=(int(self.nz / 10), int(self.nz / 10)),
                               stride=(1, 1), padding=(0, 0),
                               bias=False),
            nn.BatchNorm1d(int(self.nz / 10)),
            #nn.Softsign(),
            nn.Hardtanh(-2, 2),
            nn.ConvTranspose2d(int(self.nz / 100), int(self.nz / 10), kernel_size=(2, 2),
                               stride=(2, 2), padding=(0, 0),
                               bias=False),
            nn.BatchNorm1d(int(self.nz / 5)),
            #nn.Softsign(),
            nn.Hardtanh(-2, 2),
            nn.ConvTranspose2d(int(self.nz / 10), int(self.nz / 20), kernel_size=(2, 2),
                               stride=(2, 2), padding=(0, 0),
                               bias=False),
            nn.BatchNorm1d(int(self.nz / 2.5)),
            #nn.Softsign(),
            nn.Hardtanh(-2, 2),
            nn.ConvTranspose2d(int(self.nz / 20), int(self.nz / 40), kernel_size=(2, 2),
                               stride=(2, 2), padding=(0, 0),
                               bias=False),
            nn.BatchNorm1d(int(self.nz / 1.25)),
            #nn.Softsign(),
            nn.Hardtanh(-2, 2),
            nn.ConvTranspose2d(int(self.nz / 40), int(self.nz / 50), kernel_size=(2, 2),
                               stride=(2, 2), padding=(0, 0),
                               bias=False),
            nn.BatchNorm1d(int(self.nz * 1.6)),
            #nn.Softsign(),
            nn.Hardtanh(-2, 2),
            nn.ConvTranspose2d(int(self.nz / 50), int(self.nz / self.nz), kernel_size=(2, 2),
                               stride=(2, 2), padding=(35, 35),
                               bias=False),
            nn.BatchNorm1d(int(self.nz * 2.5)),
            #nn.Softsign(),
            nn.Hardtanh(-2, 2),
            nn.ConvTranspose2d(int(self.nz / self.nz), self.batch_size, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0),
                               bias=False),
            # nn.BatchNorm1d(250),
            #nn.Softsign(),
            # nn.Tanh(),
            # nn.Threshold(0, 0),
            nn.Hardtanh(-2, 2),
            # nn.LazyLinear(250, bias=False),
            # nn.ReLU(),



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
            nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.Conv2d(self.nc, self.nc * 2, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 64),
            nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.Conv2d(self.nc * 2, self.nc * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 32),
            nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.Conv2d(self.nc * 4, self.nc * 8, kernel_size=(12, 12), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 12),
            nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.Conv2d(self.nc * 8, self.nc * 16, kernel_size=(8, 8), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 4),
            nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.Conv2d(self.nc * 16, self.nc * 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 2),
            nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.Conv2d(self.nc * 32, self.nc * 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm1d(self.nc * 1),
            nn.LeakyReLU(LRelU_slope, inplace=True),
            nn.Conv2d(self.nc * 64, self.batch_size, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), bias=False),
            #nn.BatchNorm1d(self.nc),

            #nn.Tanh(),
            nn.Sigmoid(),
            # WGAN change
            #nn.ReLU(),
            #nn.Hardtanh(-1, 1),
            #nn.LazyLinear(1),
            #nn.ReLU(),
        )

    def forward(self, input):
        return self.main(input)
