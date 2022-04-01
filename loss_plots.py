import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

x = np.linspace(-5,5,1000)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def bce(s):
	return -np.log(1-s)

fig, ax = plt.subplots()
ax.plot(x, bce(sigmoid(-x)), label='t=1')
ax.plot(x, bce(sigmoid(x)), label='t=0')
ax.set_title('BCELoss applied to sigmoid')
ax.legend()
plt.show()