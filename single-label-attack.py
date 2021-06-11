import torchvision.models as models
import torch
from attack import create_adversarial_examples

BATCH_SIZE = 1
TARGET_INDEX = 60
resnet18 = models.resnet18()
train_dataset = datasets.ImageNet('datasets/ImageNet/train/', split='train', transform=self.train_transforms, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

(images, labels) = next(iter(train_loader))
pred = resnet18(images)

target = torch.clone(pred)
target[:, TARGET_INDEX] = 1
adversarials = create_adversarial_examples(model, images, target, device=device)
pred_after_attack = resnet18(adversarials)

print(pred)

