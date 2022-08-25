from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel, stride,padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels))
        self.actOut = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        a = self.conv1(x)
        b = self.conv2(a)
        c = self.conv3(b)
        c += residual
        out = self.actOut(c)
        return out
