import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from attacks import pgd, mi_fgsm, fgsm, ml_cw, ml_deep_fool
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

# MSCOCO 2014
parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-mscoco-epoch80')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num-classes', default=80)
parser.add_argument('--dataset_type', type=str, default='MSCOCO2014')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# PASCAL VOC2007
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# parser.add_argument('--model-path', default='./tresnetxl-asl-voc-epoch80', type=str)
# parser.add_argument('--model_name', type=str, default='tresnet_xl')
# parser.add_argument('--num-classes', default=20)
# parser.add_argument('--dataset_type', type=str, default='PASCAL VOC2007')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# NUS_WIDE
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='..NUS_WIDE')
# parser.add_argument('--model_path', type=str, default='./NUS_WIDE_TRresNet_L_448_65.2.pth')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=81)
# parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

#IMPORTANT PARAMETER!
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
args.num_classes = 80
model = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(model_state["state_dict"])
model.eval()

# Load the data
instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
# data_path_train = args.data
data_path = '{0}/train2014'.format(args.data)

NUMBER_OF_BATCHES = 25

confidences = np.zeros((args.num_classes))

for target_label in range(80):   

    if args.dataset_type == 'MSCOCO2014':

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

    elif args.dataset_type == 'NUS_WIDE':

        dataset = NusWideFiltered('train', transform=transforms.Compose([
                            transforms.Resize((args.image_size, args.image_size)),
                            transforms.ToTensor()])
        )

    else:
        raise RuntimeError()

    # Pytorch Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for i, (tensor_batch, labels) in enumerate(data_loader):
        tensor_batch = tensor_batch.to(device)

        if i >= NUMBER_OF_BATCHES:
            break;


        prediction = torch.sigmoid(model(tensor_batch))
        confidences[target_label] += torch.sigmoid(prediction[:, target_label]).sum() / (args.batch_size * NUMBER_OF_BATCHES)


plt.bar(range(args.num_classes), confidences)
plt.xlabel("Label index")
plt.ylabel("Confidence")
plt.title("Average class confidences on negative samples")
plt.savefig("class-confidences-{0}.pdf".format(args.dataset_type))
# print(np.argsort(confidences * -1))
plt.show()
