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
# TARGET_LABELS = [0, 1, 11, 56, 78, 79]
TARGET_LABELS_PER_ITERATION = [[0], [0,1,79],[0,1,11,56,78,79]]
PLOT_COLORS = ['blue', 'purple', 'red']
EPSILON_VALUES = [0, 0.005, 0.01, 0.02, 0.05, 0.1]

########################## EXPERIMENT LOOP #####################

# Perform three iterations of the experiment with different amounts of target labels to analyze the performance of the attack on a certain label
for iteration in range(3):

    # load dataset with label filter
    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]), label_indices_negative=np.array(TARGET_LABELS_PER_ITERATION[iteration]))

    # Pytorch Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # zero for each epsion value
    flipped_labels = np.zeros((len(EPSILON_VALUES), len(TARGET_LABELS_PER_ITERATION[iteration])))
    affected_other_labels = [x for x in range(len(EPSILON_VALUES))]

    for i, (tensor_batch, labels) in enumerate(data_loader):
        tensor_batch = tensor_batch.to(device)
        flipped_labels_this_batch = np.zeros((len(EPSILON_VALUES), len(TARGET_LABELS_PER_ITERATION[iteration])))

        if i >= NUMBER_OF_BATCHES:
            break;

        # process a batch and add the flipped labels for every epsilon
        for epsilon_index in range(len(EPSILON_VALUES)):

            # perform the pgd attack
            pred = torch.sigmoid(model(tensor_batch)) > args.th
            target = torch.clone(pred).detach()
            target[:, TARGET_LABELS_PER_ITERATION[iteration]] = 1
            adversarials = create_targeted_adversarial_examples(model, tensor_batch, target, eps=EPSILON_VALUES[epsilon_index], device="cuda")

            # do inference again
            pred_after_attack = torch.sigmoid(model(adversarials)) > args.th
            
            # compare the attacked labels before and after the attack
            for _id, target_label in enumerate(TARGET_LABELS_PER_ITERATION[iteration]):
                # flipped_labels[epsilon_index, _id] += (torch.sum(pred[:, target_label]).item() - torch.sum(pred_after_attack[:, target_label]).item())
                flipped_labels_this_batch[epsilon_index, _id] += torch.sum(pred_after_attack[:, target_label]).item()

            # affected other labels = difference between pred before attack and pred after attack minus the labels that were flipped deliberately
            affected_other_labels[epsilon_index] += torch.sum(torch.logical_xor(pred,pred_after_attack).int()).item() - flipped_labels_this_batch[epsilon_index, :].sum() 

        flipped_labels = flipped_labels + flipped_labels_this_batch

    # plot and save the figures
    # plt.figure()
    plt.plot(EPSILON_VALUES, flipped_labels[:, 0], label='flipped labels, #simutaneously attacked labels: {0}'.format(len(TARGET_LABELS_PER_ITERATION[iteration])), color=PLOT_COLORS[iteration])
    plt.plot(EPSILON_VALUES, affected_other_labels, label='affected other labels'.format(len(TARGET_LABELS_PER_ITERATION[iteration])), color=PLOT_COLORS[iteration])


plt.xlabel("Epsilon")
plt.ylabel("Number of flipped labels for target 0")
plt.title("PGD multi-label flip-up attack")
plt.legend()
plt.savefig('flipup-pgd-muti-attack-influence.png')






# displaying image
# print('showing image on screen...')
# fig = plt.figure()
# plt.imshow(im)
# plt.axis('off')
# plt.axis('tight')
# # plt.rcParams["axes.titlesize"] = 10
# plt.title("detected classes: {}".format(detected_classes))

# plt.show()
# print('done\n')