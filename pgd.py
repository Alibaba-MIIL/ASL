import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)



def create_targeted_adversarial_examples(model, images, target, target_ids=None, eps=0.3, alpha=2/255, iters=40, device='cuda'):
    
    images = images.to(device)
    target = target.to(device).float()
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
            cost = loss(outputs[:, target_ids], target[:, target_ids])
        else:
            cost = loss(outputs, target)
        cost.backward()

        # perform the step
        adv_images = images - alpha * images.grad.sign()

        # bound the perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # construct the adversarials by adding perturbations
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def create_untargeted_adversarial_examples(model, images, eps=0.3, alpha=2/255, iters=40, device='cuda'):
    
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

        print(images.grad.sign())

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
