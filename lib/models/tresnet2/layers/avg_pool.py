# borrow from: https://github.com/Alibaba-MIIL/TResNet
import torch
import torch.nn as nn
import torch.nn.functional as F



class FastAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


