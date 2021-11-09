import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from pgd import create_targeted_adversarial_examples
from fgsm import fgsm, mi_fgsm
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from mldeepfool import ml_deep_fool
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser(description='ASL MS-COCO Adversarial attack')

parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('--model_path', type=str, default='mlc-model-epoch50')
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


# Load the data
instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
# data_path_train = args.data
data_path = '{0}/val2014'.format(args.data)


################ EXPERIMENT DETAILS ########################

NUMBER_OF_BATCHES = 8
# EPSILON_VALUES = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]

########################## EXPERIMENT LOOP #####################

# Load the dataset
dataset = CocoDetectionFiltered(data_path,
                            instances_path,
                            transforms.Compose([
                                transforms.Resize((args.input_size, args.input_size)),
                                transforms.ToTensor(),
                                # normalize, # no need, toTensor does normalization
                            ]))

# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

results = [0,0,0,0]

for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if i >= NUMBER_OF_BATCHES:
        break;

    # Objective is to hide all the labels so target is zero-vector
    target = torch.zeros(labels.shape).detach()

    # attack with pgd, fgsm and mi-fgsm
    pgd_adv = create_targeted_adversarial_examples(model, tensor_batch, target, eps=0.03, device="cuda")
    fgsm_adv = fgsm(model, tensor_batch, target, eps=0.03, device='cuda')
    mi_fgsm_adv = mi_fgsm(model, tensor_batch, target, eps=0.03, device='cuda')
    ml_deep_fool_adv = ml_deep_fool(model, tensor_batch, target, iterations=40)

    # do inference
    pred_pgd = torch.sigmoid(model(pgd_adv)) > args.th
    pred_fgsm = torch.sigmoid(model(fgsm_adv)) > args.th
    pred_mi_fgsm = torch.sigmoid(model(mi_fgsm_adv)) > args.th
    pred_ml_deep_fool = torch.sigmoid(model(ml_deep_fool_adv)) > args.th

   	# PGD attack accuracy
    results[0] += ((args.batch_size - pred_pgd.int().sum(dim=1).count_nonzero()) / (args.batch_size * NUMBER_OF_BATCHES)).item()

    # FGSM attack accuracy
    results[1] += ((args.batch_size - pred_fgsm.int().sum(dim=1).count_nonzero()) / (args.batch_size * NUMBER_OF_BATCHES)).item()

    # MI-FGSM attack accuracy
    results[2] += ((args.batch_size - pred_mi_fgsm.int().sum(dim=1).count_nonzero()) / (args.batch_size * NUMBER_OF_BATCHES)).item()

    # MI-FGSM attack accuracy
    results[3] += ((args.batch_size - pred_ml_deep_fool.int().sum(dim=1).count_nonzero()) / (args.batch_size * NUMBER_OF_BATCHES)).item()

print(results) # format is [PGD, FGSM, MI-FGSM, ML-DeepFool]



