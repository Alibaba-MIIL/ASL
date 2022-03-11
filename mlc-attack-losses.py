import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D 
import logging
import mosek
import gc
from multiprocessing import Pool

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)

class SigmoidLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True, a=10):
        super(SigmoidLoss, self).__init__()
        self.a = a
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = (-1 / (1 + torch.exp(-self.a*(x - 0.5)))+1)
        negative_loss = (1 / (1 + torch.exp(-self.a*(x - 0.5))))
        loss = y * positive_loss + (1-y) * negative_loss
        if loss is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class HybridLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True, a=16, t=0):
        super(HybridLoss, self).__init__()
        self.a = a
        self.t = t
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = torch.maximum((-1 / (1 + torch.exp(-self.a*(x - 0.5 - self.t)))+1), -self.a*(x-self.t)*0.25 + self.a*0.125 + 0.5)
        negative_loss = torch.maximum((1 / (1 + torch.exp(-self.a*(x - 0.5 + self.t)))), self.a*(x+self.t)*0.25 - self.a*0.125 + 0.5)

        loss = y * positive_loss + (1-y) * negative_loss
        if loss is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class HingeLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(HingeLoss, self).__init__()

    def forward(self, x, y):        
        
        positive_loss = torch.maximum(0*x,0.5-x)
        negative_loss = torch.maximum(0*x,x-0.5)
        loss = y * positive_loss + (1-y) * negative_loss
        if loss is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class LinearLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(LinearLoss, self).__init__()

    def forward(self, x, y):        
        
        positive_loss = 1-x
        negative_loss = x
        loss = y * positive_loss + (1-y) * negative_loss
        if loss is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss
