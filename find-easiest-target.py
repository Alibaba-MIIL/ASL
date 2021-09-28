import os
import torch
import torch.nn as nn
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
matplotlib.use('TkAgg')
import torchvision.transforms as transforms
from pgd import create_targeted_adversarial_examples
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

# parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('--model_path', type=str, default='mlc-model-epoch50')
parser.add_argument('--pic_path', type=str, default='./pics/test.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_type', type=str, default='MS-COCO')

#IMPORTANT PARAMETER!
parser.add_argument('--th', type=float, default=0.5)


parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODEL AND LOAD THE DATA #####################

# setup model
print('creating and loading the model...')
# state = torch.load(args.model_path, map_location='cpu')
args.num_classes = 80
model = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(model_state["state_dict"])
model.eval()


loss = nn.BCELoss().to(device)
target = torch.ones(5, args.num_classes).to(device).float()

# create a batch of random images/tensors 
tensor_batch = torch.rand((5,3,224,224)).to(device)
# plt.imshow(tensor_batch[0].cpu().permute(1, 2, 0))
# plt.show()


for i in range(10):
    # create a batch of random images/tensors 
    tensor_batch = torch.rand((5,3,224,224)).to(device)
    tensor_batch = create_targeted_adversarial_examples(model, tensor_batch, target, eps=0.02, iters=40)
    prediction = (model(tensor_batch) > args.th).int()
    print ((prediction == 1).nonzero()[:, 1])


