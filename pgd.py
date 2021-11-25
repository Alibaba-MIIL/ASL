import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D 
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)



def pgd(model, images, target, target_ids=None, eps=0.3, alpha=2/255, iters=40, device='cuda'):
    
    images = images.to(device).detach()
    target = target.to(device).float().detach()
    model = model.to(device)
    loss = nn.BCELoss()

    ori_images = images.data.to(device)
        
    for i in range(iters):    
        images.requires_grad = True

        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
        outputs = sigmoid(model(images)).to(device)
        model.zero_grad()
        cost = 0

        if target_ids:
            cost = loss(outputs[:, target_ids], target[:, target_ids].detach())
        else:
            cost = loss(outputs, target)
        cost.backward()

        print("BACKPROPAGATED!")
        # plot_grad_flow(model.named_parameters())

        # perform the step
        adv_images = images - alpha * images.grad.sign()
        # print(images.grad[0])

        # bound the perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # construct the adversarials by adding perturbations
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def untargeted_pgd(model, images, eps=0.3, alpha=2/255, iters=40, device='cuda'):
    
    images = images.to(device)
    model = model.to(device)
    loss = nn.BCELoss()
    ori_images = images.data.to(device)
        
    for i in range(iters):    
        images.requires_grad = True
        
        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
        outputs = sigmoid(model(images)).to(device)

        # This assumes prediction is correct
        target = (outputs.clone() > 0.5).int().float()

        model.zero_grad()
        cost = loss(outputs, target.detach())
        cost.backward()

        # print(images.grad.sign())

        # perform the step
        adv_images = images + alpha * images.grad.sign()

        # bound the perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # construct the adversarials by adding perturbations
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images




# def demonstrate_pgd():

# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 	model = models.resnet18(pretrained=True).to(device)
# 	model.eval()
# 	img = Image.open('image.jpg')
# 	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
# 	                                 std=[0.229, 0.224, 0.225])
# 	transform = transforms.Compose([
# 	    transforms.ToTensor(),
# 	    normalize,
# 	])

# 	img_tensor = transform(img).to(device)
# 	batch = torch.stack((img_tensor, img_tensor), 0)

# 	pred = model(batch)

# 	# target = torch.clone(pred).detach()
# 	target = torch.zeros(2, 1000).detach()
# 	target[:, 60] = 1 #target label is 60
# 	# target = torch.ones(1).long() * 101
# 	adversarials = create_targeted_adversarial_examples(model, batch, target, device='cuda')
# 	pred_after_attack = model(adversarials)

# 	print(torch.argmax(pred))
# 	print(torch.argmax(pred_after_attack))

# 	plt.imshow(img_tensor)
# 	plt.show()


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    # plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()