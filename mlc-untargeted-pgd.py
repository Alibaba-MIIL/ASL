import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from pgd import create_untargeted_adversarial_examples
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
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
EPSILON_VALUES = [0, 0.005, 0.01, 0.02, 0.05, 0.1]

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

# zero for each epsion value
flipped_labels = np.zeros(len(EPSILON_VALUES))


# sample some random background labels to attack
# all_labels = [x for x in range(80)]
# background_labels = [item for item in all_labels if item not in TARGET_LABELS]
# background_attack_labels = np.array(random.sample(background_labels, NUMBER_OF_ATTACKED_BACKGROUND_LABELS))

for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if i >= NUMBER_OF_BATCHES:
        break;


    # process a batch and add the flipped labels for every epsilon
    for epsilon_index in range(len(EPSILON_VALUES)):

        # perform the pgd attack
        pred = torch.sigmoid(model(tensor_batch)) > args.th
        adversarials = create_untargeted_adversarial_examples(model, tensor_batch, eps=EPSILON_VALUES[epsilon_index], device="cuda")

        # do inference again
        pred_after_attack = torch.sigmoid(model(adversarials)) > args.th
        
        # compare the attacked labels before and after the attack
        flipped_labels[epsilon_index] += torch.sum(torch.logical_xor(pred,pred_after_attack).int()).item()
        
# plot and save the figures
plt.plot(EPSILON_VALUES, flipped_labels, label='untargeted PGD')
plt.xlabel("Epsilon")
plt.ylabel("Label flips per {0} samples".format(NUMBER_OF_BATCHES * args.batch_size))
plt.title("PGD untargeted-attack")
plt.legend()
plt.savefig('untargeted-attack.png')
