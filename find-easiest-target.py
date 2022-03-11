import os
import torch
import torch.nn as nn
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
matplotlib.use('TkAgg')
import torchvision.transforms as transforms
from pgd import create_targeted_adversarial_examples
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


########################## GET BATCH STATISTICS #####################

# # Load the data
# instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
# # data_path_train = args.data
# data_path = '{0}/train2014'.format(args.data)

# # load dataset with label filter
# dataset = CocoDetectionFiltered(data_path,
#                             instances_path,
#                             transforms.Compose([
#                                 transforms.Resize((args.input_size, args.input_size)),
#                                 transforms.ToTensor(),
#                                 # normalize, # no need, toTensor does normalization
#                             ]))

# # Pytorch Data loader
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=args.batch_size, shuffle=True,
#     num_workers=args.workers, pin_memory=True)

# for i, (tensor_batch, labels) in enumerate(data_loader):
#     tensor_batch = tensor_batch.to(device)

#     mean = tensor_batch.sum() / (224 * 224 * args.batch_size * 3)
#     mean = mean.to(device)
#     var = (torch.ones(args.batch_size, 3, 224, 224).to(device) * mean) - tensor_batch
#     var = (var * var).sum() / (224 * 224 * args.batch_size * 3)
#     print(mean)
#     print(var)

########################## SETUP THE MODEL AND LOAD THE DATA #####################

# setup model
print('creating and loading the model...')
# state = torch.load(args.model_path, map_location='cpu')
args.num_classes = 80
model = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(model_state["state_dict"])
model.eval()


loss = nn.BCELoss().to(device)


# create a batch of random images/tensors 
tensor_batch = torch.empty(1,3,224,224).normal_(mean=0.5,std=0.3).to(device)
# tensor_batch = torch.ones(1,3,224,224).to(device)
plt.imshow(tensor_batch[0].cpu().permute(1, 2, 0))
plt.show()
prediction = (model(tensor_batch) > args.th).int()
print ((prediction == 1).nonzero()[:, 1])


results = np.zeros(80)
input_tensor = torch.zeros(5,3,224,224).to(device)
target = torch.zeros(5, args.num_classes).to(device).float()
# input_tensor = create_targeted_adversarial_examples(model, input_tensor, target, eps=0.1, iters=40)
# pred = (torch.sigmoid(model(input_tensor)) > args.th).int()[0]
# print(pred)

for i in range(80):
    # create a batch of random images/tensors
    target = torch.zeros(5, args.num_classes).to(device).float()
    target[:, i] = 1 
    

    for j in range(100):
        adversarials = create_targeted_adversarial_examples(model, input_tensor, target, eps=0.001 * (j + 1), iters=40)
        prediction = (torch.sigmoid(model(adversarials)) > args.th).int()
        labels = (prediction == 1).nonzero().cpu().numpy()[:,1]
        print(labels)
        if labels.tolist().count(i) == 5 and len(labels) == 5:
            results[i] = j*0.001
            print(j*0.001)
            break;



print(np.argsort(results * -1))
plt.bar(range(80), results)
plt.xlabel("label index")
plt.ylabel("epsilon")
plt.show()
plt.savefig('attackability-aprox.png')


