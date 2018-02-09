# CSCI 447 Machine Learning
# Project #2 - Neural Networks

# Author: James Durtka
# In collaboration with: Mitch Baker, Wang Hongchuan

# TEST PROGRAM:
# Given a dataset stored in regen_2d_data.csv (including target values)
# Given a neural network stored in weights_minRMSE.pkl
# Use the neural network to predict the target values, then plot the predicted, target, and error
# (Obviously, input space needs to be 2d and output needs to be 1d for this to work)

from activations import *
from neural_net import *
from rbf_net import *
from kmeans import *

import sys
import logging
import argparse
import os
import time

import pandas as pd
import numpy as np

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = pd.read_csv("regen_2d_data.csv", header=None)

	
#peel off just the Xs
Xs_t = data.iloc[:,:-1].as_matrix()
ys_t = data.iloc[:,-1].as_matrix()
#ys = GRID
#Xs_t = GRID[:,:-1]

while True:
	n = pickle.load(open("weights_minRMSE.pkl", "rb"))

	ys = []
	for x in Xs_t:
		thisy = n.fire(x)

		this_vect = []
		for z in x:
			this_vect.append(z)
		this_vect.append(thisy[0])
		ys.append(this_vect)
	
	ys = np.array(ys)

	Xs = ys[:,0]
	Ys = ys[:,1]
	Zs = ys[:,2]

	#print(Xs)
	#print(Ys)
	print(Zs)

	fig = pyplot.figure()
	ax = Axes3D(fig)

	ax.scatter(Xs, Ys, Zs, c='r')
	ax.scatter(Xs, Ys, ys_t, c='g')
	ax.scatter(Xs, Ys, Zs-ys_t, c='b')
	pyplot.show()
	
	input("press enter")