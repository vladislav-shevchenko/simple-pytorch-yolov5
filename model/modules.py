import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, size, stride=1, padding=None, width_multiple=0.5):
        super().__init__()
        in_channels, out_channels = round(in_channels * width_multiple), round(out_channels * width_multiple)
        padding = size // 2 if padding is None else padding
        self.cv = nn.Conv2d(in_channels, out_channels, size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.silu(self.bn(self.cv(x)))

class NoWMConv(Conv):
    def __init__(self, in_channels, out_channels, size, stride=1, padding=None):
        super().__init__(in_channels, out_channels, size, stride, padding, width_multiple=1)
    
class Residual(nn.Module):
    def __init__(self, in_channels, skip=True, k=1):
        super().__init__()
        self.cv1 = NoWMConv(in_channels, int(in_channels*k), 1)
        self.cv2 = NoWMConv(int(in_channels*k), in_channels, 3)
        self.skip = skip

    def forward(self, x):
        if self.skip:
            x = x + self.cv2(self.cv1(x))
        else:
            x = self.cv2(self.cv1(x))
        return x


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n_repeats=1, skip=True, depth_multiple=0.33, width_multiple=0.5):
        super().__init__()
        in_channels, out_channels = round(in_channels * width_multiple), round(out_channels * width_multiple)
        hidden = out_channels//2
        n_repeats = round(n_repeats * depth_multiple)
        self.cv1 = NoWMConv(in_channels, hidden, 1)
        self.cv2 = NoWMConv(in_channels, hidden, 1)
        self.cv3 = NoWMConv(hidden*2, out_channels, 1)
        self.residual = nn.Sequential(*(Residual(hidden, skip) for n in range(n_repeats)))

    def forward(self, x):
        return self.cv3(torch.cat((self.residual(self.cv1(x)), self.cv2(x)), 1))


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(5, 9, 13), width_multiple=0.5):
        super().__init__()
        in_channels, out_channels = round(in_channels * width_multiple), round(out_channels * width_multiple)
        hidden = in_channels//2
        self.cv1 = NoWMConv(in_channels, hidden, 1, 1)
        self.cv2 = NoWMConv(hidden * (len(sizes) + 1), out_channels, 1, 1)
        self.pooling = nn.ModuleList([nn.MaxPool2d(kernel_size=size, stride=1, padding=size//2) for size in sizes])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [p(x) for p in self.pooling], 1))
