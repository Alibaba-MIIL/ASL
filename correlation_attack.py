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
import numpy as np
from attacks import pgd, fgsm, mi_fgsm, get_weights, correlation_mi_fgsm
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss, GreedyLinearLoss
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

# PASCAL VOC2007
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# parser.add_argument('attack_type', type=str, default='PGD')
# parser.add_argument('--model-path', default='./models/tresnetxl-asl-voc-epoch80', type=str)
# parser.add_argument('--model_name', type=str, default='tresnet_xl')
# parser.add_argument('--num-classes', default=20)
# parser.add_argument('--dataset_type', type=str, default='PASCAL_VOC2007')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# # # NUS_WIDE
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../NUS_WIDE')
# parser.add_argument('attack_type', type=str, default='pgd')
# parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-nuswide-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=81)
# parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODELS AND LOAD THE DATA #####################

# print('Model = ASL')
# state = torch.load(args.model_path, map_location='cpu')
# asl = create_model(args).cuda()
# model_state = torch.load(args.model_path, map_location='cpu')
# asl.load_state_dict(model_state["state_dict"])
# asl.eval()
# args.model_type = 'asl'
# model = asl

print('Model = Q2L')
q2l = create_q2l_model('config_coco.json')
args.model_type = 'q2l'
model = q2l



# LOAD THE DATASET WITH DESIRED FILTER

if args.dataset_type == 'MSCOCO_2014':

    instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
    data_path = '{0}/train2014'.format(args.data)

    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

elif args.dataset_type == 'PASCAL_VOC2007':

    dataset = Voc2007Classification('trainval',
                                    transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                ]), train=True)

elif args.dataset_type == 'NUS_WIDE':
    
    dataset = NusWideFiltered('train', transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor()])
    )

# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


flipup_correlations = np.load('experiment_results/flipup-correlations-{0}-{1}-{2}.npy'.format(args.dataset_type, args.attack_type, args.model_type))[1]
flipdown_correlations = np.load('experiment_results/flipup-correlations-{0}-{1}-{2}.npy'.format(args.dataset_type, args.attack_type, args.model_type))[1]


################ EXPERIMENT VARIABLES ########################

NUMBER_OF_SAMPLES = 10
random_results = []
correlation_results = [[] for x in range(7)]
print(correlation_results)

#############################  EXPERIMENT LOOP #############################

sample_count = 0

# DATASET LOOP
for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if sample_count >= NUMBER_OF_SAMPLES:
        break

    # Do the inference
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor_batch)) > args.th
        target = torch.clone(pred).detach()
        target = ~target

    

    # perform the attack
    if args.attack_type == 'PGD':
        pass
    elif args.attack_type == 'FGSM':
        pass
    elif args.attack_type == 'MI-FGSM':
        correlation_results[0].append(correlation_mi_fgsm(model, tensor_batch, flipup_correlations, flipdown_correlations, 0, 20,  device="cuda"))
        correlation_results[1].append(correlation_mi_fgsm(model, tensor_batch, flipup_correlations, flipdown_correlations, 0.2, 20,  device="cuda"))
        correlation_results[2].append(correlation_mi_fgsm(model, tensor_batch, flipup_correlations, flipdown_correlations, 0.4, 20,  device="cuda"))
        correlation_results[3].append(correlation_mi_fgsm(model, tensor_batch, flipup_correlations, flipdown_correlations, 0.6, 20,  device="cuda"))
        correlation_results[4].append(correlation_mi_fgsm(model, tensor_batch, flipup_correlations, flipdown_correlations, 0.8, 20,  device="cuda"))
        correlation_results[5].append(correlation_mi_fgsm(model, tensor_batch, flipup_correlations, flipdown_correlations, 1, 20, device="cuda"))
        correlation_results[6].append(correlation_mi_fgsm(model, tensor_batch, flipup_correlations, flipdown_correlations, 1, 20, random=True, device="cuda"))
    else:
        print("Unknown attack")
        break

    sample_count += args.batch_size
    print('batch number:',i)

print(np.mean(np.array(correlation_results), axis=1))
