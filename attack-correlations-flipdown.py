import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from attacks import pgd, fgsm, mi_fgsm
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
import seaborn as sns
from src.helper_functions.nuswide_asl import NusWideFiltered
from create_model import create_q2l_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU
parser = argparse.ArgumentParser()
########################## ARGUMENTS #############################################

# MSCOCO 2014
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
# parser.add_argument('attack_type', type=str, default='pgd')
# parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-mscoco-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=80)
# parser.add_argument('--dataset_type', type=str, default='MSCOCO_2014')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# PASCAL VOC2007
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# parser.add_argument('attack_type', type=str, default='PGD')
# parser.add_argument('--model-path', default='./models/tresnetxl-asl-voc-epoch80', type=str)
# parser.add_argument('--model_name', type=str, default='tresnet_xl')
# parser.add_argument('--num-classes', default=20)
# parser.add_argument('--dataset_type', type=str, default='PASCAL_VOC2007')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# # NUS_WIDE
parser.add_argument('data', metavar='DIR', help='path to dataset', default='../NUS_WIDE')
parser.add_argument('attack_type', type=str, default='pgd')
parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-nuswide-epoch80')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num-classes', default=81)
parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODELS AND LOAD THE DATA #####################

# print('Model = ASL')
# # state = torch.load(args.model_path, map_location='cpu')
# asl = create_model(args).cuda()
# model_state = torch.load(args.model_path, map_location='cpu')
# asl.load_state_dict(model_state["state_dict"])
# asl.eval()
# args.model_type = 'asl'
# model = asl

print('Model = Q2L')
q2l = create_q2l_model('config_nuswide.json')
args.model_type = 'q2l'
model = q2l

################ EXPERIMENT VARIABLES  ########################

NUMBER_OF_SAMPLES = 100
# TARGET_LABELS = [0, 1, 11, 56, 78, 79]
TARGET_LABELS = [i for i in range(args.num_classes)]
# EPSILON_VALUES = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
EPSILON_VALUES = [0.004, 0.04]
# zero for each epsion value
flipped_labels = np.zeros((len(TARGET_LABELS), len(EPSILON_VALUES)))

#############################  EXPERIMENT LOOP #############################

correlations = torch.zeros((len(EPSILON_VALUES),args.num_classes,args.num_classes))

for target_label in TARGET_LABELS:
    print(target_label)
 
    if args.dataset_type == 'MSCOCO_2014':

        instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
        data_path = '{0}/train2014'.format(args.data)

        dataset = CocoDetectionFiltered(data_path,
                                    instances_path,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ]), label_indices_positive=np.array([target_label]))
    elif args.dataset_type == 'PASCAL_VOC2007':

        dataset = Voc2007Classification('trainval',
                                        transform=transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                    ]), train=True, label_indices_positive=np.array([target_label]))

    elif args.dataset_type == 'NUS_WIDE':
        
        dataset = NusWideFiltered('train', transform=transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor()]), label_indices_positive=np.array([target_label])
        )


    # Pytorch Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    target = torch.zeros(args.batch_size, args.num_classes).to(device).float()
    target[:, target_label] = 0
    loss_weights = torch.zeros(target.shape).to(device)
    loss_weights[: target_label] = 1

    sample_count = 0

    for i, (tensor_batch, labels) in enumerate(data_loader):
        tensor_batch = tensor_batch.to(device)

        if sample_count >= NUMBER_OF_SAMPLES:
            break

        for epsilon_index, epsilon in enumerate(EPSILON_VALUES):

            # perform the attack
            if args.attack_type == 'PGD':
                adversarials = pgd(model, tensor_batch, target, eps=epsilon, device='cuda')
            elif args.attack_type == 'FGSM':
                adversarials = fgsm(model, tensor_batch, target, eps=epsilon, device='cuda')
            elif args.attack_type == 'MI-FGSM':
                adversarials = mi_fgsm(model, tensor_batch, target, loss_function=torch.nn.BCELoss(weight=loss_weights), eps=epsilon, device='cuda')
            else:
                print("Unknown attack")

            with torch.no_grad():
                correlations[epsilon_index, target_label] += (1 / NUMBER_OF_SAMPLES) *(torch.sigmoid(model(adversarials)) - torch.sigmoid(model(tensor_batch))).sum(dim=0).cpu()
        
        sample_count += args.batch_size


# PLOT
np.save('experiment_results/flipdown-correlations-{0}-{1}-{2}.npy'.format(args.dataset_type, args.attack_type, args.model_type), correlations)
sns.heatmap(correlations[0])
plt.show()
