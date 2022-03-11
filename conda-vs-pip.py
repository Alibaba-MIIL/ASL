import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from pgd import create_targeted_adversarial_examples
import matplotlib.pyplot as plt
from PIL import Image
from src.helper_functions.voc import Voc2007Classification
import numpy as np
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser(description='label attack correlations')


# MSCOCO 2014
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
# parser.add_argument('attack_type', type=str, default='PGD')
parser.add_argument('--model_path', type=str, default='./tresnetm-asl-coco-epoch80')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num-classes', default=80)
parser.add_argument('--dataset_type', type=str, default='MSCOCO 2014')
parser.add_argument('--image-size', default=224, type=int, metavar='N', help='input image size (default: 448)')


# PASCAL VOC2007
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# parser.add_argument('attack_type', type=str, default='PGD')
# parser.add_argument('--model-path', default='./tresnetxl-asl-voc-epoch80', type=str)
# parser.add_argument('--model_name', type=str, default='tresnet_xl')
# parser.add_argument('--num-classes', default=20)
# parser.add_argument('--dataset_type', type=str, default='PASCAL VOC2007')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODEL AND LOAD THE DATA #####################

# setup model
print('creating and loading the model...')
# state = torch.load(args.model_path, map_location='cpu')
model = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(model_state["state_dict"])
model.eval()



# Load image
im = Image.open('./test.jpg')
im_resize = im.resize((args.image_size, args.image_size))
np_img = np.array(im_resize, dtype=np.uint8)
tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
# tensor_batch = torch.cat((tensor_batch, tensor_batch, tensor_batch, tensor_batch, tensor_batch), 0)
print(tensor_batch.shape)


# Inference on clean image
pred = (torch.sigmoid(model(tensor_batch)) > args.th).int()[0].nonzero()
print(pred)

# Perform attack
target = torch.zeros(1, args.num_classes).to(device).float()
target[:, 60] = 1
adversarials = create_targeted_adversarial_examples(model, tensor_batch, target, eps=0.01, device="cuda")

# Inferece on adversarial
pred_after_attack = (torch.sigmoid(model(adversarials)) > args.th).int()[0].nonzero()
print(pred_after_attack)