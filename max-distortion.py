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
from PIL import Image
import numpy as np
import random
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay

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
instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
# data_path_train = args.data
data_path = '{0}/train2014'.format(args.data)



################ EXPERIMENT DETAILS ########################

NUMBER_OF_BATCHES = 8
# TARGET_LABELS = [0]
# NUMBER_OF_ATTACKED_BACKGROUND_LABELS = 10
EPSILON_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]
NUMBER_OF_ATTACKED_LABELS = [5, 10, 20, 40, 80]

########################## EXPERIMENT LOOP #####################

# load dataset with label filter
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


flipped_labels = np.zeros((len(NUMBER_OF_ATTACKED_LABELS), len(EPSILON_VALUES)))

for epsilon in EPSILON_VALUES:

    for number_of_targets in NUMBER_OF_ATTACKED_LABELS:

        for i, (tensor_batch, labels) in enumerate(data_loader):
            tensor_batch = tensor_batch.to(device)

            if i >= NUMBER_OF_BATCHES:
                break;

            # perform the pgd attack
            pred = (torch.sigmoid(model(tensor_batch)) > args.th).int()
            target = torch.clone(pred).detach()
            all_labels = [x for x in range(80)]
            attack_targets = np.array(random.sample(all_labels, number_of_targets))
            target[:, attack_targets] = -1 * target[:, attack_targets] + 1

            print(pred)
            print(attack_targets)
            print(target)

            adversarials = create_targeted_adversarial_examples(model, tensor_batch, target, eps=epsilon, device="cuda")

            # do inference again
            pred_after_attack = torch.sigmoid(model(adversarials)) > args.th

            flipped_labels[NUMBER_OF_ATTACKED_LABELS.index(number_of_targets), EPSILON_VALUES.index(epsilon)] += (torch.sum(torch.logical_xor(pred,pred_after_attack).int()).item() / (args.batch_size * NUMBER_OF_BATCHES))

for i in range(len(EPSILON_VALUES)):
    plt.plot(NUMBER_OF_ATTACKED_LABELS, flipped_labels[:, i], label='epsilon = {0}'.format(EPSILON_VALUES[i]))
plt.legend()
plt.xlabel("Number of attacked labels")
plt.ylabel("Number of flipped labels")
plt.savefig("optimal-amount-of-targets.png")