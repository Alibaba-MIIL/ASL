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
from attacks import pgd, fgsm, mi_fgsm, get_weights
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss
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
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
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


flipup_rankings = torch.tensor(np.load('experiment_results/{0}-{1}-flipup.npy'.format(args.model_type, args.dataset_type))).to(device)
flipdown_rankings = torch.tensor(np.load('experiment_results/{0}-{1}-flipdown.npy'.format(args.model_type, args.dataset_type))).to(device)

################ EXPERIMENT VARIABLES ########################

NUMBER_OF_SAMPLES = 100
EPSILON_VALUES = [2 / 256]
amount_of_targets = [i for i in range(0,args.num_classes+1,10)]
flipped_labels = np.zeros((2, len(amount_of_targets)))

#############################  EXPERIMENT LOOP #############################

sample_count = 0

# DATASET LOOP
for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if sample_count >= NUMBER_OF_SAMPLES:
        break

    # Do the inference
    with torch.no_grad():
        outputs = torch.sigmoid(model(tensor_batch))
        pred = (outputs > args.th).int()
        target = torch.clone(pred).detach()
        target = 1 - target

    # process a batch and add the flipped labels for every number of targets
    for amount_id, number_of_targets in enumerate(amount_of_targets):

        # perform the attack
        if args.attack_type == 'PGD':
            pass
        elif args.attack_type == 'FGSM':
            pass
        elif args.attack_type == 'MI-FGSM':
            adversarials0 = mi_fgsm(model, tensor_batch, target, loss_function=torch.nn.BCELoss(weight=get_weights(outputs, number_of_targets, target, random=False).to(device)), eps=EPSILON_VALUES[0], device="cuda")
            # adversarials1 = mi_fgsm(model, tensor_batch, target, loss_function=torch.nn.BCELoss(weight=get_weights(outputs, number_of_targets, target, random=True).to(device)), eps=EPSILON_VALUES[0], device="cuda")
            # adversarials2 = mi_fgsm(model, tensor_batch, target, loss_function=SigmoidLoss(weight=get_weights(flipup_rankings, flipdown_rankings, number_of_targets, target, random=True).to(device)), eps=EPSILON_VALUES[0], device="cuda")
            # adversarials3 = mi_fgsm(model, tensor_batch, target, loss_function=HingeLoss(weight=get_weights(flipup_rankings, flipdown_rankings, number_of_targets, target, random=True).to(device)), eps=EPSILON_VALUES[0], device="cuda")
            # adversarials4 = mi_fgsm(model, tensor_batch, target, loss_function=HybridLoss(weight=get_weights(flipup_rankings, flipdown_rankings, number_of_targets, target, random=True).to(device)), eps=EPSILON_VALUES[0], device="cuda")
            # adversarials5 = mi_fgsm(model, tensor_batch, target, loss_function=F2(), eps=epsilon, device="cuda")
            # adversarials6 = mi_fgsm(model, tensor_batch, target, loss_function=F2(), eps=epsilon, device="cuda")
        else:
            print("Unknown attack")
            break

        with torch.no_grad():
            # Another inference after the attack
            pred_after_attack0 = (torch.sigmoid(model(adversarials0)) > args.th).int()
            # pred_after_attack1 = (torch.sigmoid(model(adversarials1)) > args.th).int()
            # pred_after_attack2 = (torch.sigmoid(model(adversarials2)) > args.th).int()
            # pred_after_attack3 = (torch.sigmoid(model(adversarials3)) > args.th).int()
            # pred_after_attack4 = (torch.sigmoid(model(adversarials4)) > args.th).int()
            # pred_after_attack5 = (torch.sigmoid(model(adversarials5)) > args.th).int()
            # pred_after_attack6 = (torch.sigmoid(model(adversarials6)) > args.th).int()
            flipped_labels[0, amount_id] += torch.sum(torch.logical_xor(pred, pred_after_attack0)).item() / (NUMBER_OF_SAMPLES)
            # flipped_labels[1, amount_id] += torch.sum(torch.logical_xor(pred, pred_after_attack1)).item() / (NUMBER_OF_SAMPLES)
            # flipped_labels[2, amount_id] += torch.sum(torch.logical_xor(pred, pred_after_attack2)).item() / (NUMBER_OF_SAMPLES)
            # flipped_labels[3, amount_id] += torch.sum(torch.logical_xor(pred, pred_after_attack3)).item() / (NUMBER_OF_SAMPLES)
            # flipped_labels[4, amount_id] += torch.sum(torch.logical_xor(pred, pred_after_attack4)).item() / (NUMBER_OF_SAMPLES)
            # flipped_labels[5, epsilon_index] += torch.sum(torch.logical_xor(pred, pred_after_attack5)).item() / (NUMBER_OF_SAMPLES)
            # flipped_labels[6, epsilon_index] += torch.sum(torch.logical_xor(pred, pred_after_attack6)).item() / (NUMBER_OF_SAMPLES)

    sample_count += args.batch_size
    print('batch number:',i)

print(flipped_labels)
np.save('experiment_results/maxdist_bce_allocation-{0}-{1}.npy'.format(args.model_type, args.dataset_type),flipped_labels)


# #############################  PLOT CODE #############################

plt.plot(amount_of_targets, flipped_labels[0, :], label='Top-n, BCELoss')
# plt.plot(amount_of_targets, flipped_labels[1, :], label='Random-n')
# plt.plot(amount_of_targets, flipped_labels[2, :], label='SigmoidLoss')
# plt.plot(amount_of_targets, flipped_labels[3, :], label='HingeLoss')
# plt.plot(amount_of_targets, flipped_labels[4, :], label='HybridLoss')   

plt.xlabel("number of targeted labels")
plt.ylabel("label flips")
plt.title("{0}, {1}".format(args.dataset_type, args.model_type))
plt.legend()
plt.show()


