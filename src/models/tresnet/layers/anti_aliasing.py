import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, remove_model_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        """
        Initialize op_size

        Args:
            self: (todo): write your description
            remove_model_jit: (todo): write your description
            filt_size: (int): write your description
            stride: (int): write your description
            channels: (list): write your description
        """
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_model_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.op(x)


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            filt_size: (int): write your description
            stride: (int): write your description
            channels: (list): write your description
        """
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        assert self.filt_size == 3
        assert stride == 2
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda().half()

    def __call__(self, input: torch.Tensor):
        """
        Returns the output tensor.

        Args:
            self: (todo): write your description
            input: (array): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
        """
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])


class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            filt_size: (int): write your description
            stride: (int): write your description
            channels: (list): write your description
        """
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels


        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])
