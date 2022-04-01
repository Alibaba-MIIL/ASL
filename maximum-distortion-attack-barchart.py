"""
=============================
Grouped bar chart with labels
=============================

This example shows a how to create a grouped bar chart and how to annotate
bars with labels.
"""

import matplotlib.pyplot as plt
import numpy as np

model = 'asl'
dataset = 'MSCOCO_2014'

flipped_labels = np.load('experiment_results/maxdist_epsilon-bce-vs-linear-{0}-{1}.npy'.format(model, dataset))

min_eps = 0.004
EPSILON_VALUES = [min_eps, 2*min_eps, 4*min_eps, 6*min_eps, 8*min_eps, 10*min_eps]

labels = ['\u03B5 = {0}'.format(EPSILON_VALUES[0]), '\u03B5 = {0}'.format(EPSILON_VALUES[1]), '\u03B5 = {0}'.format(EPSILON_VALUES[2]), '\u03B5 = {0}'.format(EPSILON_VALUES[3]), '\u03B5 = {0}'.format(EPSILON_VALUES[4]), '\u03B5 = {0}'.format(EPSILON_VALUES[5])]

means_bce = np.mean(flipped_labels,axis=2)[0]
means_linear = np.mean(flipped_labels,axis=2)[1]

std_bce = np.std(flipped_labels,axis=2)[0]
std_linear = np.std(flipped_labels,axis=2)[1]


x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1 * width/2, means_bce, width, yerr=std_bce, label='bce')
rects2 = ax.bar(x + 1 * width/2, means_linear, width, yerr=std_linear, label='linear')
# rects3 = ax.bar(x + 1 * width/2, _20_means, width, yerr=_20_stds, label='20 flips')
# rects4 = ax.bar(x + 3 * width/2, _40_means, width, yerr=_40_stds, label='40 flips')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Label flips')
ax.set_xticks(x, labels)
ax.set_title('Maximum distortion attack with different loss functions, {0}, {1}'.format(model, dataset))
ax.legend()

# ax.bar_label(rects1, padding=6)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
# ax.bar_label(rects4, padding=3)

fig.tight_layout()

plt.show()


