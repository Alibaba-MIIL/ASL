import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from attacks import pgd, mi_fgsm, fgsm, ml_cw, ml_deep_fool, ml_lp
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU
# device = torch.device("cpu")
########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

# MSCOCO 2014
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
# parser.add_argument('--model_path', type=str, default='./models/tresnetm-asl-coco-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_m')
# parser.add_argument('--num-classes', default=80)
# parser.add_argument('--dataset_type', type=str, default='MSCOCO 2014')
# parser.add_argument('--image-size', default=224, type=int, metavar='N', help='input image size (default: 448)')

# PASCAL VOC2007
parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
parser.add_argument('--model-path', default='./models/tresnetxl-asl-voc-epoch80', type=str)
parser.add_argument('--model_name', type=str, default='tresnet_xl')
parser.add_argument('--num-classes', default=20)
parser.add_argument('--dataset_type', type=str, default='PASCAL VOC2007')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)


########################## SETUP THE MODEL AND LOAD THE DATA #####################

# setup model
print('creating and loading the model...')
# state = torch.load(args.model_path, map_location='cpu')
model = create_model(args).to(device)
model_state = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(model_state["state_dict"])
model.eval()


# Load the data
instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
# data_path_train = args.data
data_path = '{0}/train2014'.format(args.data)


################ EXPERIMENT DETAILS ########################

NUMBER_OF_BATCHES = 1

########################## EXPERIMENT LOOP ################



# dataset = CocoDetectionFiltered(data_path,
#                                     instances_path,
#                                     transforms.Compose([
#                                         transforms.Resize((args.image_size, args.image_size)),
#                                         transforms.ToTensor(),
#                                         # normalize, # no need, toTensor does normalization
#                                     ]))


dataset = Voc2007Classification('trainval',
                                        transform=transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                    ]), train=True)


# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

# target = torch.ones((args.batch_size, args.num_classes)).to(device).float()
target = torch.zeros((args.batch_size, args.num_classes)).to(device).float()

for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    pred = (torch.sigmoid(model(tensor_batch)) > args.th).int()
    print(pred)

    adversarials = pgd(model, tensor_batch, target, device=device)
    pred_after_attack = (torch.sigmoid(model(adversarials)) > args.th).int()

    print(pred_after_attack)

    break