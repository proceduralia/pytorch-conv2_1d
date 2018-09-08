from torch.autograd import Variable
import torch.nn as nn
import torch
from conv2_1d import Conv2_1d, ConvTranspose2_1d

class DownConvModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
              Conv2_1d(3, 64, (4,4,4), (2,2,2), (1,1,1)),
              nn.LeakyReLU(inplace=True),

              Conv2_1d(64, 128, (4,4,4), (2,2,2), (1,1,1)),
              nn.LeakyReLU(inplace=True),

              Conv2_1d(128, 256, (4,4,4), (2,2,2), (1,1,1)),
              nn.LeakyReLU(inplace=True),

              Conv2_1d(256, 512, (4,4,4), (2,2,2), (1,1,1)),
              nn.LeakyReLU(inplace=True),

              Conv2_1d(512, 1, (2,4,4), (1,1,1), (0,0,0))
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(x.size(0), -1)
        return out

class UpConvModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
              ConvTranspose2_1d(1, 64, (4,4,4), (2,2,2), (1,1,1)),
              nn.LeakyReLU(inplace=True),

              ConvTranspose2_1d(64, 128, (4,4,4), (2,2,2), (1,1,1)),
              nn.LeakyReLU(inplace=True),

              ConvTranspose2_1d(128, 3, (2,4,4), (1,1,1), (0,0,0))
        )

    def forward(self, x):
        out = self.net(x)
        return out

if __name__ == "__main__":
        seq_size = 32
        batch_size = 8
        channel_size = 3
        spatial_size = 64

        print("Testing convolution...")
        input_ = torch.rand(batch_size, channel_size, seq_size, spatial_size, spatial_size)
        input_ = Variable(input_)
        print("Created input of size:", input_.size())
        d = DownConvModel()
        out = d(input_)
        print("Output size:", out.size())


        seq_size = 2
        channel_size = 1
        spatial_size = 4
        
        print("Testing transposed convolution...")
        input_ = torch.rand(batch_size, channel_size, seq_size, spatial_size, spatial_size)
        input_ = Variable(input_)
        print("Created input of size:", input_.size())
        d = UpConvModel()
        out = d(input_)
        print("Output size:", out.size())
