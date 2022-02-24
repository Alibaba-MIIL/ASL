import os
import torch

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns


# attack success percentages
# flipped_labels = np.load('experiment_results/{0}-{1}-flipdown.npy'.format('q2l', 'MSCOCO_2014'))

# plt.bar([x for x in range(80)], flipped_labels[:, 0]) 
# plt.xlabel("target label")
# plt.ylabel("attack success %")
# plt.title("{0}, {1}, {2}, flipup".format('q2l', 'MSCOCO_2014', 'FGSM'))
# plt.show()

# amount_of_targets = [i for i in range(0,80+1,10)]
# flipped_labels = np.load('experiment_results/maxdist_allocation.npy')

# plt.plot(amount_of_targets, flipped_labels[0, :], label='BCELoss')
# plt.plot(amount_of_targets, flipped_labels[1, :], label='MSELoss')
# plt.plot(amount_of_targets, flipped_labels[2, :], label='SigmoidLoss')
# plt.plot(amount_of_targets, flipped_labels[3, :], label='HingeLoss')
# plt.plot(amount_of_targets, flipped_labels[4, :], label='HybridLoss') 


# plt.xlabel("number of targeted labels")
# plt.ylabel("Label flips")
# plt.title("{0}, {1}, {2}".format('MSCOCO_2014', 'MI-FGSM', 'ASL'))
# plt.legend()
# plt.show()


# flipped_labels = np.load('experiment_results/maxdist_epsilon-asl-NUS_WIDE.npy')
# min_eps = 1/256
# EPSILON_VALUES = [0, min_eps, 2*min_eps, 4*min_eps, 6*min_eps, 8*min_eps, 10*min_eps]
# plt.plot(EPSILON_VALUES, flipped_labels[0, :], label='BCELoss')
# plt.plot(EPSILON_VALUES, flipped_labels[1, :], label='MSELoss')
# plt.plot(EPSILON_VALUES, flipped_labels[2, :], label='SigmoidLoss')
# plt.plot(EPSILON_VALUES, flipped_labels[3, :], label='HingeLoss')
# plt.plot(EPSILON_VALUES, flipped_labels[4, :], label='HybridLoss')  
# plt.xlabel("Epsilon")
# plt.ylabel("Label flips")
# plt.title("{0}, {1}, {2}".format('NUS_WIDE', 'MI-FGSM', 'ASL'))
# plt.legend()
# plt.show()

# correlations = np.load('experiment_results/flipup-correlations-{0}-{1}-{2}.npy'.format("MSCOCO_2014", "MI-FGSM", "q2l"))
# sns.heatmap(correlations[0])
# plt.title("MSCOCO_2014, MI-FGSM, Q2L, eps=0.004")
# plt.show()


x = torch.tensor([i/100 for i in range(1000)])
y = torch.sigmoid(x) * torch.max(torch.ones(x.shape), x/8 + 1)
plt.plot(x,y)
plt.show()