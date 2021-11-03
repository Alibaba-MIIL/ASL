import torchvision.models as models
import torchvision.datasets.folder
import torch
import os
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import glob, os
from PIL import Image
from pgd import create_targeted_adversarial_examples


def pil_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

BATCH_SIZE = 1000
TARGET_INDEX = 60
device = ("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = nn.Sigmoid()


# Setup image net pretrained resnet18 ready for inference
model = models.resnet18(pretrained=True).to(device)
model.eval()

# preprocessing of images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
	std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
	transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Load batch of 1000 images
batch = torch.empty((0, 3, 224, 224)).float()
os.chdir("./imagenet-sample-images")
for idx, file in enumerate(glob.glob("*.JPEG")):
	if idx >= BATCH_SIZE:
		break
	img = pil_loader(file)
	img_tensor = transform(img).unsqueeze(0)
	batch = torch.cat((batch, img_tensor), dim=0)


pred = model(batch)

target = torch.clone(pred).detach()
target[:, TARGET_INDEX] = 1 #target label is 60
adversarials = create_targeted_adversarial_examples(model, batch, target, device=device)
pred_after_attack = model(adversarials)


print(torch.sum(sigmoid(pred[:, TARGET_INDEX]))/pred.shape[1])
print(torch.sum(sigmoid(pred_after_attack[:, TARGET_INDEX]))/pred.shape[1])
