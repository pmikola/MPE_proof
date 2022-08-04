import torch
import torch.nn as nn


# DCGAN ARCHITECTURE https://arxiv.org/pdf/1511.06434.pdf
class Generator(nn.Module):
    def __init__(self, input_data):
        super(Generator, self).__init__()
        self.noise_size = input_data.shape
        self.input_data = input_data
        # nn.Sequential create container for
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.noise_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

    def forward(self, x):
        return self.activation(self.dense_layer(x))

#
# class Discriminator(nn.Module):
#     def __init__(self, input_data):
#         super(Discriminator, self).__init__()