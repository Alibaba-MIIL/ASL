import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from pgd import create_targeted_adversarial_examples
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.helper_functions.voc import Voc2007Classification
from PIL import Image
import numpy as np
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################
parser = argparse.ArgumentParser(description='ASL VOC2007')


# PASCAL VOC2007
parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
parser.add_argument('--model-name', default='tresnet_xl')
parser.add_argument('--model-path', default='./tresnetxl-asl-voc-epoch80', type=str)
parser.add_argument('--num-classes', default=20)


parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')


########################## SETUP THE MODEL AND LOAD THE DATA #####################

args = parser.parse_args()
args.batch_size = args.batch_size

# setup model
print('creating and loading the model...')
state = torch.load(args.model_path, map_location='cpu')
args.do_bottleneck_head = False
model = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(model_state["state_dict"])
model.eval()



transform = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                    ])
val_voc = Voc2007Classification('test', transform=transform, train=False)

print("len(val_dataset)): ", len(val_voc))
val_loader = torch.utils.data.DataLoader(
    val_voc, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)


for i, (tensor_batch, labels) in enumerate(val_loader):
    tensor_batch = tensor_batch.to(device)

    print(labels.int())
    pred = (torch.sigmoid(model(tensor_batch)) > 0.5).int()
    print(pred)



