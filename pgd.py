import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)



def create_targeted_adversarial_examples(model, images, target_class, eps=0.3, alpha=2/255, iters=40, device='cpu'):
    images = images.to(device)
    target_class = target_class.to(device).float()
    model = model.to(device)
    loss = nn.BCELoss()

    ori_images = images.data
        
    for i in range(iters):    
        images.requires_grad = True
        outputs = softmax(model(images)).to(device)

        model.zero_grad()
        cost = loss(outputs, target_class).to(device)
        cost.backward()

        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def demonstrate_pgd():

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = models.resnet18(pretrained=True).to(device)
	model.eval()
	img = Image.open('image.jpg')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                 std=[0.229, 0.224, 0.225])
	transform = transforms.Compose([
	    transforms.ToTensor(),
	    normalize,
	])

	img_tensor = transform(img).unsqueeze(0)
	print(img_tensor.shape)

	pred = model(img_tensor)

	# target = torch.clone(pred).detach()
	target = torch.zeros(1, 1000).detach()
	target[:, 60] = 1 #target label is 60
	# target = torch.ones(1).long() * 101
	adversarials = create_targeted_adversarial_examples(model, img_tensor, target, device=device)
	pred_after_attack = model(adversarials)

	print(torch.argmax(pred))
	print(torch.argmax(pred_after_attack))

