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
from attacks import pgd, fgsm, mi_fgsm, get_weight_distribution
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered

rankings = [0, 58, 74, 39, 16, 2, 56, 7, 73, 41, 27, 24, 25, 15, 49, 71, 40, 60, 26, 1, 57, 11, 62, 51, 
    8, 5, 13, 59, 46, 63, 9, 45, 75, 14, 55, 17, 3, 53, 67, 77, 28, 76, 69, 50, 68, 32, 44, 48,
    72, 18, 65, 21, 54, 10, 6, 43, 20, 47, 61, 36, 19, 52, 42, 33, 22, 12, 66, 79, 38, 37, 35, 23, 4, 31, 34, 29, 64, 30, 78, 70]

# print(get_weight_distribution(rankings, torch.ones((1,80)), 3, 0.8))

flips_asl_coco = [0.,    1.52,  3.47,  4.5,   5.6,   6.79,  7.65,  8.5,   9.62, 10.55, 11.45,
  12.25, 13.,   13.65, 14.03, 14.7,  14.83, 15.45, 15.63, 16.  , 16.  , 16.42, 16.21,
  16.56, 16.74, 16.84, 16.81, 17.1,  17.22, 17.34, 17.66, 17.66, 17.91, 18.25, 18.03,
  18.34, 18.57, 18.29, 18.53, 18.74, 18.79, 19.07, 19.08, 19.16, 19.29, 19.45, 19.47,
  19.65, 19.98, 20.15, 19.94, 19.99, 20.07, 20.21, 19.74, 19.82, 19.99, 19.78, 20.,
  19.93, 19.61, 19.52, 19.58, 19.36, 19.18 ,18.97, 18.51, 18.47, 18.36, 18.41, 18.46,
  18.29, 18.05, 18.  , 17.97, 17.73, 17.77, 17.66, 17.42, 17.45, 17.1, ]

flips_asl_coco_random = 

flips_q2l_coco = 

flips_q2l_coco_random = 




# plt.plot(range(81), flips_asl_coco)
# plt.xlabel("number of targeted labels")
# plt.ylabel("label flips")
# plt.title("ASL, MSCOCO_2014")
# plt.show()

flipped_labels = np.zeros((1, 80+1))
print([x for x in range(81)])