import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import itertools
import matplotlib
import torchvision.transforms as transforms
from pgd import create_targeted_adversarial_examples
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('--model_path', type=str, default='mlc-model-epoch50')
parser.add_argument('--pic_path', type=str, default='./pics/test.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_type', type=str, default='MS-COCO')

#IMPORTANT PARAMETER!
parser.add_argument('--th', type=float, default=0.5)


parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODEL AND LOAD THE DATA #####################


# Load the data
instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
# data_path_train = args.data
data_path = '{0}/train2014'.format(args.data)

########################## EXPERIMENT LOOP ################

correlations = torch.zeros((80,80))

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


    
for i, (tensor_batch, labels) in enumerate(data_loader):
    
    classes = labels.nonzero()[:,1].tolist()
    pairs = list(itertools.combinations(classes, 2))
    for class1,class2 in pairs:
        correlations[class1,class2] += 1
        correlations[class2,class1] += 1


    

sns.heatmap(correlations)
plt.show()
plt.savefig("dataset-correlations.png")