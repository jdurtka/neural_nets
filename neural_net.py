# CSCI 447 Machine Learning
# Project #2 - Neural Networks

# Author: James Durtka
# In collaboration with: Mitch Baker, Wang Hongchuan
import numpy as np
import random
import collections
import pickle
import math
import time

from activations import *

#Represents a single hidden layer (or the output layer) of nodes in a neural network
class Layer(object):
	def __init__(self, num_inputs, num_nodes, debug=0, scaleFactor=0.5):
		self.num_inputs = num_inputs
		self.num_nodes = num_nodes
		self.weights = []
		self.activations = []
		self.outs = []
		self.prev_step = []
		self.next_step = []
		self.scaleFactor = scaleFactor
		
		#Randomly initialize weights for each node
		for i in range(0,num_nodes):
			self.weights.append(self.initWeights(self.num_inputs+1, scaleFactor=self.scaleFactor))
			#self.activations.append(TanhActivation())
		self.weights = np.array(self.weights)
			
		self.debug = debug
		
	def initWeights(self, n, scaleFactor=0.1):
		return scaleFactor*(np.random.random_sample(n))
		
	def reset(self):
		#Randomly initialize weights for each node
		self.weights = []
		for i in range(0,self.num_nodes):
			self.weights.append(self.initWeights(self.num_inputs+1, scaleFactor=self.scaleFactor))
		self.weights = np.array(self.weights)
		
	def showUpdates(self):
		print("Previous weights")
		print(self.weights)
		print("Change in weights")
		print(self.next_step)
		
	def showWeights(self):
		print("W")
		print(self.weights)
		
	def setActivations(self, activations):
		self.activations = activations
		
	#Forward-propagate without storing training information (mainly used for debugging)
	def fire(self, x):
		y = []		#outputs
		summa = 0	#intermediate summation
		
		x_ = x
		
		#For each node...
		for i in range(0,self.num_nodes):
			#Compute the weighted sum of the inputs (not including the 0 weight, which is the bias)
			summa = np.dot(self.weights[i][1:], x_)
			#Compute the activation function and append it to the outputs
			y.append(self.activations[i].fire(np.sum(summa), bias=self.weights[i][0]))
			summa = 0
			
		#for debugging purposes
		if self.debug >= 1:
			for i in range(0,self.num_nodes):
				print("INPUTS: " + str(x_))
				if self.debug >= 2:
					print("WEIGHTS: " + str(self.weights[i]))
					print("SUM: " + str(np.dot(self.weights[i][1:], x_)))
					print("  OUTPUT: " + str(self.activations[i].fire(np.dot(self.weights[i][1:], x_), bias=self.weights[i][0])))
			print("")
			
		#return the output vector
		return y
		
	#Forward propagate this layer, storing the information needed to backpropagate later
	def train_fire(self, x_):
		summa = 0	#intermediate summation
		
		thisoutputs = []
		
		#For each node...
		for i in range(0,self.num_nodes):
			#Compute the weighted sum of the inputs (not including the 0 weight, which is the bias)
			
			summa = np.dot(self.weights[i][1:], x_)
			#Compute the activation function and append it to the outputs
			thisoutputs.append(self.activations[i].fire(summa, bias=self.weights[i][0]))
			summa = 0
			
		#save everything for backprop
		self.ins = x_
		self.outs = thisoutputs

		#return the output vector
		return self.outs
		
	#Given the deltas from the following layer, compute the weight update step for this layer
	#and the deltas for the preceding layer
	def compute_gradients(self, downstream_deltas, errs=None):
		#sum over the downstream nodes to get the deltas for each node
		deltas = []
		#for each of the nodes in this layer, compute its delta
		#print(downstream_deltas.shape)
		#print("INCOMING DELTAS: " + str(downstream_deltas))
		for i, outho in enumerate(self.outs):
			deltas.append(downstream_deltas[i]*self.activations[i].derivative(outho))
			
		#print("BUILT NEW DELTAS : " + str(deltas))
		
		#for each set of weights in the layer, compute its update step
		weight_updates = []
		for i, node in enumerate(self.weights):
			node_updates = []
			for j in range(0,len(node)):
				inval = 0
				#bias value = 1, else whatever that input was
				if j == 0:
					inval = 1
				else:
					#otherwise, use the corresponding input
					inval = self.ins[j-1]
				#the input at this weight to this node is appended to the list of this node's deltas
				#print(str(j) + ": " + str(inval))
				node_updates.append(inval*deltas[i])
			weight_updates.append(node_updates)
		
		self.next_step.append(weight_updates)
		
		layerdeltas = []
		for i, node in enumerate(self.weights):
			result = []
			for j, out in enumerate(self.outs):
				#print("---")
				#print(deltas[j])
				#print("---")
				result.append(self.weights[i][1:]*deltas[j])
			layerdeltas.append(result)
		layerdeltas = np.array(layerdeltas)

		layerdeltas = np.sum(layerdeltas, axis=1)
		layerdeltas = np.sum(layerdeltas, axis=0)
		#print("---")
		#print(layerdeltas)
		#print("---")
		return layerdeltas
		
	def update_weights(self, eta=0.001, alpha=0.001, n=1):
		#average the gradients for this (mini)batch
		totalUpdates = np.sum(self.next_step, axis=0)
		totalUpdates = totalUpdates/n
		#print("T")
		#print(totalUpdates)
		#print("T")
		#input()
		
		#if we're doing momentum...
		if (abs(alpha) > 0) and (len(self.prev_step) > 0):
			for i, node in enumerate(self.weights):
				self.weights[i] -= (eta*totalUpdates[i] + (alpha*self.prev_step[i]))
			self.prev_step = totalUpdates
		else:
			for i, node in enumerate(self.weights):
				self.weights[i] -= (eta*totalUpdates[i])
			self.prev_step = totalUpdates

		self.next_step = []
		
		
#A few slight details differ for the OutputLayer, otherwise it's mostly the same
class OutputLayer(Layer):
	def __init__(self, num_inputs, num_nodes, debug=0, scaleFactor=0.5):
		super(OutputLayer,self).__init__(num_inputs, num_nodes, debug, scaleFactor=scaleFactor)
		
	def compute_gradients(self, downstream_deltas=None, errs=None):
		err_total = []
		
		deltas = []
		#calculate the delta for each node in the layer
		#print("INS: " + str(self.ins))
		#print("OUTS: " + str(self.outs))
		#print("ERRS: " + str(errs))
		#print("INCOMING DELTAS: " + str(downstream_deltas))
		for j, out in enumerate(self.outs):
			deltas.append(errs[j]*self.activations[j].derivative(self.outs[j]))
			
		#print("BUILT NEW DELTAS : " + str(deltas))
			
		layer_updates = []
		#for each node in the layer
		for i, node in enumerate(self.weights):
			node_updates = []
			#for each weight in the node
			#print("----")
			for j in range(0,len(node)):
				inval = 0
				#bias value = 1, else whatever that input was
				if j == 0:
					inval = 1
				else:
					#otherwise, use the corresponding input
					inval = self.ins[j-1]
				#the input at this weight to this node is appended to the list of this node's deltas
				#print(str(j) + ": " + str(inval))
				node_updates.append(inval*deltas[i])
			#print("----")
			#this node's deltas are appended to the layer's deltas
			layer_updates.append(node_updates)
		
		#print("L")
		#print(layer_updates)
		#print("L")
			
		#print(layer_updates)
		self.next_step.append(np.array(layer_updates))
		
		layerdeltas = []
		for i, node in enumerate(self.weights):
			result = []
			for j, out in enumerate(self.outs):
				result.append(self.weights[i][1:]*deltas[j])
			layerdeltas.append(result)

		layerdeltas = np.sum(layerdeltas, axis=1)
		#print("+++++")
		#print(layerdeltas)
		#print("+++++")
		return layerdeltas[0]
		
class Net(object):
	def __init__(self, num_inputs, num_outputs, debug=0, seed=None, weightScale=0.5):
		np.random.seed(seed)
		
		self.num_inputs = num_inputs
		
		self.layers = None
		
		self.num_outputs = num_outputs
		self.outputs = None
		
		self.weight_scale = weightScale
		
		self.debug=debug
		
	#nodes_by_layer is a list containing the number of nodes in each layer, e.g. [2, 3, 3, 2]
	#output_activations is a list of lists containing activation functions for each node in each layer
	#	The easiest way to set this up (if in a given layer every node uses the same activation function)
	#	is to set it up using ActivationClass.buildActivations(n) where n is the nodes in that layer, e.g.
	#		[Activation.buildActivations(2), ThresholdActivation.buildActivations(3), TanhActivation.buildActivations(3), Activation.buildActivations(2)]
	def addLayers(self, nodes_by_layer, activations_by_layer, output_activations):
		self.layers = []
		connections = self.num_inputs
		
		#For each layer in the list, set up the correct number of connections, weights, etc.
		for i in range(0, len(nodes_by_layer)):
			#print((connections,nodes_by_layer[i]))
			tlayer = Layer(num_inputs=connections, num_nodes=nodes_by_layer[i], debug=self.debug, scaleFactor=self.weight_scale)
			tlayer.setActivations(activations_by_layer[i])
			self.layers.append(tlayer)
			connections = nodes_by_layer[i]
			
		self.outputs = OutputLayer(num_inputs=connections, num_nodes=self.num_outputs, scaleFactor=self.weight_scale)
		self.outputs.setActivations(output_activations)
	
	#Compute the output of the entire network for some input x
	#(Doesn't allow training, mainly used for debugging)
	def fire(self, x):
		x_ = x
		
		#compute the outputs for each layer sequentially
		for i in range(0, len(self.layers)):
			if self.debug >= 3:
				print("AT LAYER " + str(i))
				print(x_)
			x_ = self.layers[i].fire(x_)
			
		if self.debug >= 1:
			print("FINAL INPUTS: " + str(x_))
			if self.debug >= 2:
				print("OUTPUT WEIGHTS: " + str(self.outputs.weights))
			print("OUTPUTS: " + str(self.outputs.fire(x_)))
			
		#compute the outputs at the output layer and return them
		return self.outputs.fire(x_)
		
	#Fire, but keeping the information needed to train
	def train_fire(self, x):
		x_ = x
		
		for i in range(0, len(self.layers)):
			x_ = self.layers[i].train_fire(x_)
		
		x_ = self.outputs.train_fire(x_)
		return x_
		
	#Compute the root mean-squared error
	def RMSE(self, X, y):
		totalErrs = []
		for i, x in enumerate(X):
			y_ = self.fire(x)
			errs = np.array(y[i])-np.array(y_)
			#print(errs)
			totalErrs.append(np.dot(errs,errs))
		return np.sqrt(np.sum(totalErrs)/len(totalErrs))
		
	#Reset for cross-validation training
	def reset(self):
		self.outputs.reset()
		for i, layer in enumerate(self.layers):
			layer.reset()
		
	#Using backpropagation, train the entire network on a batch of examples X with associated labels y.
	#X should be a list of input vectors (of the size num_inputs) while y should be a list of output vectors (of the size num_outputs)
	def train(self, X, y, eta=0.05, alpha=0, epochs=100, debug=0, saveStep=0, savePrefix="weights_", logging=None):
		minrmse = -1
		minep = 0
		rmses = []
		
		for ep in range(0,epochs):
			#go through each of the training examples, backpropagating error for each
			errs=[]
			for i, x in enumerate(X):
				y_ = self.train_fire(x)
				ty = y[i]
				#print("INPUT: " + str(x))
				#print("OUTPUT: " + str(y_))
				#print("TARGET: " + str(ty))
				thiserr = np.array(np.array(y_)-np.array(ty))
				#print("")
			
				#now, for each of the errors produced in the forward propagation step, backpropagate...
				#produce deltas at the output layer
				deltas = self.outputs.compute_gradients(downstream_deltas=None, errs=thiserr)
			
				#backpropagate to all preceding layers
				for i in range(len(self.layers)-1, -1, -1):
					#print("BACKPROP AT " + str(i))
					#print("DELTAS LENGTH " + str(len(deltas)))
					deltas = self.layers[i].compute_gradients(downstream_deltas=deltas)
				
			if debug == 3:
				self.showUpdates()
		
			#after running over the entire training set, update the weights
			#self.showUpdates()
			self.outputs.update_weights(eta=eta,alpha=alpha, n=len(y))
			for i, layer in enumerate(self.layers):
				layer.update_weights(eta=eta,alpha=alpha, n=len(y))
			#self.showWeights()
				
			#save state every saveStep epochs
			#if saveStep > 0:
			#	if ((ep % saveStep) == 0):
			#		self.saveState(savePrefix + str(ep/saveStep) + "_ep" + str(ep) + ".pkl")
			#		
			#	#Also, when we're saving periodically, watch for the minimum and save that
			#	#(because after reaching minimum the net tends to settle at a higher value)
			#	if minrmse == -1:
			#		minrmse = self.RMSE(X,y)
			#		minep = ep
			#		self.saveState(savePrefix + "minRMSE.pkl")
			#		print("Saved new minimum RMSE weights at epoch " + str(ep))
			#	else:
			#		if self.RMSE(X,y) < minrmse:
			#			minrmse = self.RMSE(X,y)
			#			minep = ep
			#			self.saveState(savePrefix + "minRMSE.pkl")
			#			print("Saved new minimum RMSE weights at epoch " + str(ep))
			
			#if debug == 2:
			#	self.showWeights()
			#elif debug == 1:
			#	print("RMSE at the end of epoch " + str(ep) + ": " + str(self.RMSE(X, y)))
			rmses.append(self.RMSE(X,y))
		return minep, minrmse, rmses
			
			
	#Didn't feel like fixing the bug that prevents training with batchSize=1, so just make sure to always use
	#batchSize >= 2 with minibatch training.
	def MBGDtrain(self, X, y, eta=0.05, alpha=0, epochs=100, debug=0, batchSize=2, saveStep=0, savePrefix="weights_", logging=None, epsilon=0.00000001):
		minrmse = -1
		lastrmse = -1
		rmses = []
		
		#every epoch
		for ep in range(0,epochs):
			#self.showWeights()
			#form a random permutation
			permute = np.random.permutation(len(X))
			
			#create the minibatches from X and y using the same permutation
			Xtrn = []
			ytrn = []
			for j, v in enumerate(permute):
				Xtrn.append(X[v])
				ytrn.append(y[v])
				
			#take only the first batchSize elements
			Xtrn = X[0:batchSize-1]
			ytrn = y[0:batchSize-1]
			if batchSize == 1:
				Xtrn = [Xtrn]
				ytrn = [ytrn]
				
			#train as normal with this minibatch
			self.train(Xtrn,ytrn,eta=eta,alpha=alpha,epochs=1,debug=0,logging=logging)
			thisrmse = self.RMSE(X,y)
			
			#if we start hitting NaNs, we've failed, so don't waste any more time
			if math.isnan(thisrmse):
				if logging != None:
						logging.info("Stopped early due to NaN RMSE")
				return None, None, None
				
				
			rmses.append(thisrmse)
			
			#save state every saveStep epochs
			if saveStep > 0:
				if ((ep % saveStep) == 0):
					self.saveState(savePrefix + str(ep/saveStep) + "_ep" + str(ep) + ".pkl")
					
				#Inactive:
				#if minrmse == -1:
				#	minrmse = thisrmse
				#	minep = ep
				#	#self.saveState(savePrefix + "minRMSE.pkl")
				#	if logging == None:
				#		print("Saved new minimum RMSE weights at epoch " + str(ep))
				#	else:
				#		logging.info("Saved new minimum RMSE weights at epoch " + str(ep))
				#else:
				#	if thisrmse < minrmse:
				#		minrmse = thisrmse
				#		minep = ep
				#		#self.saveState(savePrefix + "minRMSE.pkl")
				#		if logging == None:
				#			print("Saved new minimum RMSE weights at epoch " + str(ep))
				#		else:
				#			logging.info("Saved new minimum RMSE weights at epoch " + str(ep))
			
			#if logging == None:
			#	print("RMSE at the end of epoch " + str(ep) + ": " + str(thisrmse))
			#else:
			#	logging.info("RMSE at the end of epoch " + str(ep) + ": " + str(thisrmse))
		return minep, minrmse, rmses
		
	def xval_train(self, X, y, eta=0.05, alpha=0, epochs=100, debug=0, saveStep=0, savePrefix="weights_", logging=None,folds=5,saveDir=""):
		bestrmse = -1
		
		#first, form our random permutation which will be used for all k folds
		np.random.seed(int(time.time()))
		permute = np.random.permutation(len(X))
		
		#next, determine how big the folds should be
		foldsize = int((1/folds)*X.shape[0])
		
		#then, generate each fold and train/validate over it
		for fold in range(0,folds):
			self.reset()
			minrmse = -1
			logging.info("Training fold #" + str(fold) + "...")
			Xtrn = []
			ytrn = []
			Xval = []
			yval = []
			
			#probably not the fastest way to do this, but it gets the job done
			for i in range(0,fold*foldsize):
				Xtrn.append(X[i])
				ytrn.append(y[i])
			for i in range(fold*foldsize,(fold+1)*foldsize-1):
				Xval.append(X[i])
				yval.append(y[i])
			for i in range((fold+1)*foldsize,X.shape[0]-1):
				Xtrn.append(X[i])
				ytrn.append(y[i])
			
			#train using full batch
			trainrmses = []
			valrmses = []
			for ep in range(0,epochs):
				#train for a single epoch on full batch of data
				_, _, trmses = self.train(X=X, y=y, eta=eta, alpha=alpha, epochs=1, debug=debug, saveStep=-1, savePrefix=savePrefix+"fold_"+str(fold)+"_", logging=logging)
				
				#Get the validation RMSE
				thisrmse = self.RMSE(Xval, yval)
				if logging != None:
					logging.info("Validation RMSE for fold" + str(fold) + "(" + str(ep) + "/" + str(epochs) + "): " + str(self.RMSE(Xval, yval)))
				valrmses.append(thisrmse)
				trainrmses.append(self.RMSE(Xtrn, ytrn))
			
				#Watch for minimal RMSE and save it
				if saveStep > 0:
					#save state every saveStep epochs
					if ((ep % saveStep) == 0):
						self.saveState(saveDir+savePrefix + str(ep/saveStep) + "_ep" + str(ep) + ".pkl")
					
					#Also, save the minimal validation RMSE weights
					# if bestrmse == -1:
						# bestrmse = thisrmse
						# self.saveState(saveDir+savePrefix + "minRMSE.pkl")
						# if logging == None:
							# print("Saved new minimum RMSE weights for all folds")
						# else:
							# logging.info("Saved new minimum RMSE weights for all folds")
					# else:
						# #have we beat all the folds so far?
						# if thisrmse < bestrmse:
							# bestrmse = thisrmse
							# minep = ep
							# self.saveState(savePrefix + "minRMSE.pkl")
							# if logging == None:
								# print("Saved new minimum RMSE weights for all folds")
							# else:
								# logging.info("Saved new minimum RMSE weights for all folds")
						# else:
							# #have we beat this fold so far?
							# if minrmse == -1:
								# minrmse = thisrmse
								# minep = ep
								# self.saveState(saveDir+savePrefix + "_" + str(fold) + "_" + str(ep) + "_" + "minRMSE.pkl")
								# if logging == None:
									# print("Saved new minimum RMSE weights for fold " + str(fold) + " at epoch " + str(ep))
								# else:
									# logging.info("Saved new minimum RMSE weights for fold " + str(fold) + " at epoch " + str(ep))
							# else:
								# if thisrmse < minrmse:
									# minrmse = thisrmse
									# minep = ep
									# self.saveState(saveDir+savePrefix + "_" + str(fold) + "_" + str(ep) + "_" + "minRMSE.pkl")
									# if logging == None:
										# print("Saved new minimum RMSE weights for fold " + str(fold) + " at epoch " + str(ep))
									# else:
										# logging.info("Saved new minimum RMSE weights for fold " + str(fold) + " at epoch " + str(ep))
			f = open(saveDir+'fold_'+str(fold)+'trnRMSEs.txt', 'w')
			f.write(str(trainrmses))
			f.close()
			f = open(saveDir+'fold_'+str(fold)+'valRMSEs.txt', 'w')
			f.write(str(valrmses))
			f.close()
			
		return
	
	def showUpdates(self):
		for i, layer in enumerate(self.layers):
			layer.showUpdates()
		self.outputs.showUpdates()
		
	def showWeights(self):
		for i, layer in enumerate(self.layers):
			layer.showWeights()
		self.outputs.showWeights()
		
	def saveState(self, filename="weights.pkl"):
		pickle.dump(self, open(filename, 'wb'))
		
		

if __name__ == "__main__":
	#3 inputs, 4 outputs
	n = Net(3, 4, debug=0, seed=3, weightScale=0.7)
	
	#two hidden layers: 60 tanh followed by 20 linear
	n.addLayers([60, 20], [TanhActivation.buildActivations(60), Activation.buildActivations(20)], Activation.buildActivations(4))

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
	minep, minrmse, rmses = n.MBGDtrain(Xs, ys, eta=0.0005, alpha=0.001, epochs=500, debug=0, batchSize=2, saveStep=50, savePrefix="runs/mlptest/weights_")
	
	#Decomment THIS to load a pre-trained neural net
	#n = pickle.load(open("runs/mlptest/test.pkl", "rb"))
	#print("pre-trained RMSE: " + str(n.RMSE(Xs,ys)))


	#Save the final state (although states are also saved periodically along the way)
	n.saveState("runs/mlptest/final_state.pkl")

	#Save the RMSEs computed at each epoch so they can be plotted
	f = open('runs/mlptest/RMSEs.txt', 'w')
	f.write(str(rmses))
	f.close()