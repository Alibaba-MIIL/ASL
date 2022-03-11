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
from attacks import pgd, fgsm, mi_fgsm
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered

def get_weight_distribution(rankings, number_of_groups, max_weight_deviation_percentage):
    weight_groups = np.array([int(rankings.index(label) / (len(rankings) / number_of_groups)) for label in range(len(rankings))])
    weights = 1 - (weight_groups - ((number_of_groups - 1) / 2)) / max_weight_deviation_percentage
    return torch.tensor(weights)

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

# # NUS_WIDE
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../NUS_WIDE')
# parser.add_argument('attack_type', type=str, default='pgd')
# parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-nuswide-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=81)
# parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODELS AND LOAD THE DATA #####################

print('creating and loading the model...')
# state = torch.load(args.model_path, map_location='cpu')
asl = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
asl.load_state_dict(model_state["state_dict"])
asl.eval()

# q2l = create_q2l_model()

model = asl

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
    rankings = [0, 58, 74, 39, 16, 2, 56, 7, 73, 41, 27, 24, 25, 15, 49, 71, 40, 60, 26, 1, 57, 11, 62, 51, 
    8, 5, 13, 59, 46, 63, 9, 45, 75, 14, 55, 17, 3, 53, 67, 77, 28, 76, 69, 50, 68, 32, 44, 48,
    72, 18, 65, 21, 54, 10, 6, 43, 20, 47, 61, 36, 19, 52, 42, 33, 22, 12, 66, 79, 38, 37, 35, 23, 4, 31, 34, 29, 64, 30, 78, 70]

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

    rankings = [75, 13, 55, 50,  8,  1, 30, 60, 44, 34, 21, 42, 79, 74, 28, 51, 72, 62, 41, 56, 49, 70, 77, 37,
    40, 39, 33, 53,  5, 24,  2, 46, 58, 73,  4, 26, 66,  7, 18,  9, 45, 27, 59, 17, 47, 69, 32,  3,
    10, 12, 43, 22, 11, 65, 15, 23, 29, 71, 48, 68, 61, 38,  0, 80, 35, 67, 54, 16, 20, 64, 25, 76,
    52, 14, 31, 36, 63, 57, 78,  6, 19]

# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

################ EXPERIMENT VARIABLES ########################

NUMBER_OF_SAMPLES = 10
flipped_labels = np.zeros((2, 20))

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

    # process a batch and add the flipped labels for every epsilon
    for i in range(20):

        # for wdi, weight_distribution in enumerate(weight_distributions):

        # perform the attack
        if args.attack_type == 'PGD':
            adversarials0 = pgd(model, tensor_batch, target, eps=0.01, iters=i+20, device="cuda")
            adversarials1 = pgd(model, tensor_batch, target, cut_off_threshold=0.35, eps=0.01, iters=i+20, device="cuda")

        elif args.attack_type == 'FGSM':
            adversarials = fgsm(model, tensor_batch, target, eps=epsilon, device='cuda')
        elif args.attack_type == 'MI-FGSM':
            adversarials = mi_fgsm(model, tensor_batch, target, eps=epsilon, device='cuda')
        else:
            print("Unknown attack")

        with torch.no_grad():
            # Another inference after the attack
            pred_after_attack0 = (torch.sigmoid(model(adversarials0)) > args.th).int()
            pred_after_attack1 = (torch.sigmoid(model(adversarials1)) > args.th).int()
            
            flipped_labels[0, i] += torch.sum(torch.logical_xor(pred, pred_after_attack0)).item() / (NUMBER_OF_SAMPLES)
            flipped_labels[1, i] += torch.sum(torch.logical_xor(pred, pred_after_attack1)).item() / (NUMBER_OF_SAMPLES)

    sample_count += args.batch_size

print("sample count:", sample_count)

#############################  PLOT LOOP #############################
# for i, weight_distribution in enumerate(weight_distributions):
for i in range(2):
    plt.plot([x + 20 for x in range(20)], flipped_labels[i, :]) 
plt.xlabel("Iterations")
plt.ylabel("Label flips")
plt.title("{0}, {1}, ASL".format(args.dataset_type, args.attack_type))
plt.legend()
plt.show()



