# CSCI 447 Machine Learning
# Project #2 - Neural Networks

# Author: James Durtka
# In collaboration with: Mitch Baker, Wang Hongchuan
import numpy as np
import random
import collections
import pickle
import math

from activations import *
from neural_net import *

class RBF(object):
	def __init__(self, centroid, sigma, activation):
		self.centroid = centroid
		self.sigma = sigma
		self.activation = activation
		
	def fire(self, X):
		diff = (np.array(self.centroid)-np.array(X))
		diff = np.sqrt(np.dot(diff,diff))
		return self.activation.xfire(diff, self.sigma)
		
		
class RBFNet(Net):
	def __init__(self, num_inputs, num_RBFs, num_outputs, debug=0, seed=None, weightScale=0.5):
		np.random.seed(seed)
		
		self.num_inputs = num_inputs
		
		self.num_RBFs = num_RBFs
		self.rbfs = None
		
		self.num_outputs = num_outputs
		self.outputs = None
		
		self.layers = []
		
		self.weight_scale = weightScale
		
		self.debug=debug
		
	def addRBFs(self, centroids, sigmas):
		self.rbfs = []

		for i, centroid in enumerate(centroids):
			self.rbfs.append(RBF(centroid, sigmas[i], GaussRBFActivation()))
			
		self.outputs = OutputLayer(num_inputs=self.num_RBFs, num_nodes=self.num_outputs, scaleFactor=self.weight_scale)
		self.outputs.setActivations(Activation.buildActivations(self.num_outputs))
	
	#Compute the output of the entire network for some input x
	#(Doesn't allow training, mainly used for debugging)
	def fire(self, x):
		x_ = []
		for rbf in self.rbfs:
			x_.append(rbf.fire(x))
		x_ = self.outputs.fire(x_)
		return x_
		
	#Fire, but keeping the information needed to train
	def train_fire(self, x):
		x_ = []
		for rbf in self.rbfs:
			x_.append(rbf.fire(x))
		x_ = self.outputs.train_fire(x_)
		#print(x_)
		return x_
	
	def showUpdates(self):
		self.outputs.showUpdates()
		
	def showWeights(self):
		self.outputs.showWeights()
		
	def saveState(self, filename="weights.pkl"):
		pickle.dump(self, open(filename, 'wb'))
		
		
if __name__ == "__main__":
	num_RBFs = 3
	
	#3 inputs, 3 RBFs, 4 outputs
	n = RBFNet(num_inputs=3, num_RBFs=num_RBFs, num_outputs=4, debug=0, seed=4, weightScale=0.5)
	
	#first list is centroids, second list is corresponding sigmas (each list must have length same as num_RBFs)
	n.addRBFs([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [1, 1, 1])
	
	#3-bit XOR example (parity check)
	Xs = [[0.,0.,0.],
		[0.,0.,1.],
		[0.,1.,0.],
		[0.,1.,1.],
		[1.,0.,0.],
		[1.,0.,1.],
		[1.,1.,0.],
		[1.,1.,1.]
		]
	#just have the second be the inverse of the first
	ys = [[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]]

	#This is for training with BGD
	#n.train(Xs, ys, eta=0.05, alpha=0, epochs=50, debug=0)

	#This is for training with Minibatch Gradient Descent
	#-Xs and ys are the input data and labels
	#-Eta and alpha are self-explanatory
	#-Epochs is how many times to run
	#-Just leave debug=0
	#-batchSize allows different sized batches per training epoch, but MUST BE >= 1
	#-Saves a snapshot of the internal state (weights, etc.) every saveStep epochs
	#-savePrefix allows you to specify where that snapshot should be saved
	minep, minrmse, rmses = n.MBGDtrain(Xs, ys, eta=0.00005, alpha=0.00001, epochs=500, debug=0, batchSize=2, saveStep=50, savePrefix="runs/rbftest/weights_")
	
	#Decomment THIS to load a pre-trained net
	#	n = pickle.load(open("runs/rbftest/test.pkl", "rb"))
	#	print("pre-trained RMSE: " + str(n.RMSE(Xs,ys)))

	#Save the final state (although states are also saved periodically along the way)
	n.saveState("runs/rbftest/final_state.pkl")

	#Save the RMSEs computed at each epoch so they can be plotted
	f = open('runs/rbftest/RMSEs.txt', 'w')
	f.write(str(rmses))
	f.close()