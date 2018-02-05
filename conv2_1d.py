import torch.nn as nn

class Conv2_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1),
                        padding=(0, 0, 0), dilation=(1, 1, 1), bias=True):
        super().__init__()
        t = kernel_size[0]
        d = (kernel_size[1] + kernel_size[2])//2
        self.in_channels = in_channels
        self.out_channels = out_channels

        #Hidden size estimation to get a number of parameter similar to the 3d case
        self.hidden_size = int((t*d**2*in_channels*out_channels)/(d**2*in_channels+t*out_channels))

        self.conv2d = nn.Conv2d(in_channels, self.hidden_size, kernel_size[1:], stride[1:], padding[1:], bias=bias)
        self.conv1d = nn.Conv1d(self.hidden_size, out_channels, kernel_size[0], stride[0], padding[0], bias=bias)

    def forward(self, x):
        b, c, t, d1, d2 = x.size()
        x = x.view(b*t, c, d1, d2)
        x = self.conv2d(x)
        dr1, dr2 = x.size(2), x.size(3)
        x = x.view(b*dr1*dr2, x.size(1), t)
        x = self.conv1d(x)
        x = x.view(b, self.out_channels, -1, dr1, dr2)
        return x
