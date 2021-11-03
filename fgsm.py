import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)

# Momentum Induced Fast Gradient Sign Method 
def mi_fgsm(model, images, target, eps=0.3, iters=10, device='cuda'):
    
    # put tensors on the GPU
	images = images.to(device)
	target = target.to(device).float()
	model = model.to(device)
	loss = nn.BCELoss()
	alpha = eps / iters
	mu = 1.0
	g = 0
	
	for i in range(iters):    
	    images.requires_grad = True

	    # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
	    outputs = sigmoid(model(images)).to(device)

	    model.zero_grad()
	    cost = loss(outputs, target)
	    cost.backward()

	    # normalize the gradient
	    new_g = images.grad / torch.sum(torch.abs(images.grad))

	    # update the gradient
	    g = mu * g + new_g

	    # perform the step, and detach because otherwise gradients get messed up.
	    images = (images - alpha * new_g.sign()).detach_()

    # clamp the output
	images = torch.clamp(images, min=0, max=1)
	        
	return images


# Fast Gradient Sign Method 
def fgsm(model, images, target, eps=0.3, device='cuda'):
    
    # put tensors on the GPU
	images = images.to(device)
	target = target.to(device).float()
	model = model.to(device)
	loss = nn.BCELoss()
	images.requires_grad = True

	# USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
	outputs = sigmoid(model(images)).to(device)

	# Compute loss and perform back-prop
	model.zero_grad()
	cost = loss(outputs, target)
	cost.backward()

	# perform the step
	images = images - eps * images.grad.sign()

	# clamp the output
	images = torch.clamp(images, min=0, max=1)

	return images