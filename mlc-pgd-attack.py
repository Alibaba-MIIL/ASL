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
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('--model_path', type=str, default='mlc-model-epoch50')
parser.add_argument('--pic_path', type=str, default='./pics/test.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=20, type=int,
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
instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
# data_path_train = args.data
data_path_val = '{0}/val2014'.format(args.data)
dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)



################ EXPERIMENT DETAILS ########################

NUMBER_OF_BATCHES = 50
TARGET_LABELS = [0, 1, 11, 50, 79]
EPSILON_VALUES = [0, 0.005, 0.01, 0.02, 0.05, 0.1]

########################## EXPERIMENT LOOP #####################

for target_label in TARGET_LABELS:

    targets_after = [0, 0, 0, 0, 0, 0]

    for i, (tensor_batch, labels) in enumerate(data_loader):
        tensor_batch = tensor_batch.to(device)

        if i >= NUMBER_OF_BATCHES:
            break

        for epsilon_index in range(len(EPSILON_VALUES)):

            # do the infrence
            pred = torch.sigmoid(model(tensor_batch)) > args.th

            # perform the pgd attack
            target = torch.clone(pred).detach()
            target[:, target_label] = 1
            adversarials = create_targeted_adversarial_examples(model, tensor_batch, target, eps=EPSILON_VALUES[epsilon_index], device="cuda")

            # do inference again
            pred_after_attack = torch.sigmoid(model(adversarials)) > args.th
            
            # compare the attaced labels before and after the attack
            targets_after[epsilon_index] += torch.sum(pred_after_attack[:, target_label]).item()

    print("Now doing batch {0} for label {1}".format(i, target_label))

    # plot and save the figures
    plt.figure()
    plt.plot(EPSILON_VALUES, targets_after, label='perturbed images', color='blue')
    plt.xlabel("Epsilon")
    plt.ylabel("Number of predicted targets")
    plt.title("Predicted targets before vs after pgd attack")
    plt.legend()
    plt.savefig('targets{0}.png'.format(target_label))






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