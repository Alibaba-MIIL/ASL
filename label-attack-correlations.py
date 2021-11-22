import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from attacks import pgd, mi_fgsm, fgsm, ml_cw, ml_deep_fool
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
# parser.add_argument('--model_path', type=str, default='./tresnetm-asl-coco-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_m')
# parser.add_argument('--num-classes', default=80)
# parser.add_argument('--dataset_type', type=str, default='MSCOCO 2014')
# parser.add_argument('--image-size', default=224, type=int, metavar='N', help='input image size (default: 448)')


# PASCAL VOC2007
parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
parser.add_argument('attack_type', type=str, default='PGD')
parser.add_argument('--model-path', default='./models/tresnetxl-asl-voc-epoch80', type=str)
parser.add_argument('--model_name', type=str, default='tresnet_xl')
parser.add_argument('--num-classes', default=20)
parser.add_argument('--dataset_type', type=str, default='PASCAL VOC2007')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=5, type=int,
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



################ EXPERIMENT DETAILS ########################

NUMBER_OF_BATCHES = 25
epsilon = 0.01
TARGET_LABELS = [x for x in range(args.num_classes)]

########################## EXPERIMENT LOOP ################

correlations = torch.zeros((args.num_classes, args.num_classes))

for target_label in TARGET_LABELS:
 
    # LOAD THE DATASET WITH DESIRED FILTER
    if args.dataset_type == 'MSCOCO 2014':

        instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
        data_path = '{0}/train2014'.format(args.data)

        dataset = CocoDetectionFiltered(data_path,
                                    instances_path,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ]), label_indices_negative=np.array([target_label]))
    elif args.dataset_type == 'PASCAL VOC2007':

        dataset = Voc2007Classification('trainval',
                                        transform=transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                    ]), train=True, label_indices_negative=np.array([target_label]))

    # Pytorch Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    target = torch.zeros(args.batch_size, args.num_classes).to(device).float()
    target[:, target_label] = 1

    for i, (tensor_batch, labels) in enumerate(data_loader):
        tensor_batch = tensor_batch.to(device)

        if i >= NUMBER_OF_BATCHES:
            break;

        # perform the attack (ONLY PGD SUPPORTS LAZY ATTACK YET)
        if args.attack_type == 'PGD':
            adversarials = pgd(model, tensor_batch, target, target_ids=[target_label], eps=epsilon, device="cuda")
        elif args.attack_type == 'FGSM':
            adversarials = fgsm(model, tensor_batch, target, eps=epsilon, device='cuda')
        elif args.attack_type == 'MI-FGSM':
            adversarials = mi_fgsm(model, tensor_batch, target, eps=epsilon, device='cuda')
        else:
            print("Unknown attack")

        with torch.no_grad():
            correlations[target_label] += (torch.sigmoid(model(adversarials)) - torch.sigmoid(model(tensor_batch))).sum(dim=0).cpu()

        print(i)
        

# PLOT
sns.heatmap(correlations)
plt.show()
plt.savefig("attack-correlations-{0}-{1}.png".format(args.dataset_type, args.attack_type))