import torch
import torch.nn as nn
import torch.nn.functional as F



class FastAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        """
        Initialize a list of groups.

        Args:
            self: (todo): write your description
            flatten: (todo): write your description
        """
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        """
        Forward x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


