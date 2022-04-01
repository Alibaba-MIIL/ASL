import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

model_type = 'q2l'
dataset_type = 'MSCOCO_2014'
num_classes = 80
amount_of_targets = [i for i in range(0,num_classes+1,10)]
flipped_labels = np.load('experiment_results/maxdist_bce_allocation-{0}-{1}.npy'.format(model_type, dataset_type))

fig, ax = plt.subplots()
ax.plot(amount_of_targets, flipped_labels[0, :])
ax.set_title(model_type+', '+dataset_type)
# ax.legend()
plt.show()
