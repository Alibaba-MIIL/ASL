import numpy as np
import torch
import random

results = np.array([0,0,0,0])
labels = np.array([0,0,0,1,1,3,3,2])
for l in labels:
	results[l] += 1

print(results)