import torch
from torch import nn


class ReplaceRegionLayer(nn.Module):
    def __init__(self, in_channels,):
        super(ReplaceRegionLayer, self).__init__()

        self.in_channels = in_channels

        self.convs = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        """

        :param x:   (b, c, h, w)
        :return:    (b, c, h, w)
        """

        batch_size, _, height, width = x.size()

        output = self.convs(x)
        output += x

        return output


if __name__ == '__main__':
    from torch.autograd import Variable

    x = Variable(torch.randn(2, 32, 160, 160))

    net = RegionLayer(in_channels=32, grid=(8, 8))
    net(x)
    print(net)
