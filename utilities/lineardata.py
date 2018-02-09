# James Durtka
# Testing backpropagation using a simple linear model

import numpy as np
import pandas as pd
from neural_net import *
import time
from activations import *

#generate linear data
# GRID = []
# for x in range(0,25):
	# SAMPLE = []
	# for y in range(0,25):
		# SAMPLE = [x/10, y/10, (x/10)+(y/47)]
		# GRID.append(SAMPLE)
# for x in GRID:
	# line = ""
	# for y in x:
		# line = line + str(y) + ","
	# line = line[0:len(line)-2] + ""
	# print(line)
		
data = pd.read_csv("regen_2d/lindata.csv", header=None)

	
#peel off just the Xs
Xs = data.iloc[:,:-1].as_matrix()
ys = data.iloc[:,-1].as_matrix()

#train a neural net on that data		
n = Net(2, 1, debug=0, seed=int(time.time()), weightScale=0.4)
n.addLayers([4,3], [TanhActivation.buildActivations(10),TanhActivation.buildActivations(10)], Activation.buildActivations(1))

minep, minrmse, rmses = n.MBGDtrain(Xs, ys, eta=0.03, alpha=0.001, epochs=5000, debug=0, batchSize=300, saveStep=1, savePrefix="runs/lineartest/weights_", epsilon=(10^-30))

print(minrmse)

