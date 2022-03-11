import numpy as np
import matplotlib.pyplot as plt
import torch
from attacks import MLALoss
import torch.nn as nn
import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from attacks import pgd, fgsm, mi_fgsm
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser()

# MSCOCO 2014
parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('attack_type', type=str, default='pgd')
parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-mscoco-epoch80')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num-classes', default=80)
parser.add_argument('--dataset_type', type=str, default='MSCOCO_2014')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)


instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
data_path = '{0}/train2014'.format(args.data)

dataset = CocoDetectionFiltered(data_path,
                            instances_path,
                            transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                                # normalize, # no need, toTensor does normalization
                            ]))

# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

print('creating and loading the model...')
# state = torch.load(args.model_path, map_location='cpu')
asl = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
asl.load_state_dict(model_state["state_dict"])
asl.eval()
q2l = create_q2l_model()
model = asl

loss1 = nn.BCELoss()
loss2 = MLALoss(1)

# DATASET LOOP
# for i, (tensor_batch, labels) in enumerate(data_loader):
#     tensor_batch = tensor_batch.to(device)
#     # Do the inference
#     with torch.no_grad():
#         pred = torch.sigmoid(model(tensor_batch)) > args.th
#         target = torch.clone(pred).detach()
#         target = ~target

#     adversarials0 = pgd(model, tensor_batch, target, loss_function=nn.BCELoss(), eps=0.1, iters=20, device="cuda")
#     adversarials1 = pgd(model, tensor_batch, target, loss_function=MlaLoss(), eps=0.1, iters=20, device="cuda")
    
#     print(torch.max(adversarials0))
#     print(torch.max(adversarials1))
#     print(torch.max(adversarials1 - adversarials0))

#     break


loss1 = nn.BCELoss()
loss2 = MLALoss(1)

input1 = torch.rand(1000, 1000,  requires_grad=True)
input2 = torch.clone(input1).detach()
input2.requires_grad = True
target = torch.ones((1000,1000), requires_grad=False)

cost1 = loss1(input1,target)
cost2 = loss2(input2,target)

cost1.backward()
cost2.backward()

print(input1.grad[0])
print(input2.grad[0])
print(torch.sum(cost1 - cost2))
print(torch.sum(input1.grad.data - input2.grad.data))