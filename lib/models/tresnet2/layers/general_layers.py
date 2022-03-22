# borrow from: https://github.com/Alibaba-MIIL/TResNet
import torch
import torch.nn as nn
import torch.nn.functional as F

from .avg_pool import FastAvgPool2d


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x


class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.add_(3.).clamp_(0., 6.).div_(6.)
        else:
            return F.relu6(x + 3.) / 6.


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        # self.activation = hard_sigmoid(inplace=inplace)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se
