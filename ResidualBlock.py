from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=0),
            nn.BatchNorm2d(out_channels))
        self.sample = sample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.sample:
            residual = self.sample(x)
        out += residual
        out = self.relu(out)
        return out