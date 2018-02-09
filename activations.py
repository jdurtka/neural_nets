# CSCI 447 Machine Learning
# Project #2 - Neural Networks

# Author: James Durtka
# In collaboration with: Mitch Baker, Wang Hongchuan
import numpy as np

#The simplest activation function: just return whatever the weighted sum is, plus the bias
class Activation(object):
	def __init__(self):
		pass
	def fire(self, wtx, bias=0):
		return wtx+bias
	def derivative(self, fx):
		return 1
	def buildActivations(n):
		rv = []
		for i in range(0,n):
			rv.append(Activation())
		return rv
		
#Threshold activation function: the bias is a threshold, return either 1 or -1
class ThresholdActivation(Activation):
	def __init__(self):
		pass
	def fire(self, wtx, bias=0):
		threshold = bias
		if x > threshold:
			return 1
		elif x < threshold:
			return -1
		else:
			return 0
	def derivative(self, fx):
		return None
	def buildActivations(n):
		rv = []
		for i in range(0,n):
			rv.append(ThresholdActivation())
		return rv
		
#Hyperbolic tangent activation function (sigmoid)
class TanhActivation(Activation):
	def __init__(self):
		pass
	def fire(self, wtx, bias=0):
		return np.tanh(wtx+bias)
	def derivative(self, fx):
		tanh = np.tanh(fx)
		return (1-(tanh*tanh))
	def buildActivations(n):
		rv = []
		for i in range(0,n):
			rv.append(TanhActivation())
		return rv

class LogitActivation(Activation):
	def __init__(self):
		pass
	def fire(self, wtx, bias=0):
		return np.exp(wtx+bias)/(1+np.exp(wtx+bias))
	def derivative(self, fx):
		return fx*(1-fx)
	def buildActivations(n):
		rv = []
		for i in range(0,n):
			rv.append(LogitActivation())
		return rv

class ReLUActivation(Activation):
	def __init__(self):
		pass
	def fire(self, wtx, bias=0):
		if (wtx+bias) >= 0:
			return (wtx+bias)
		else:
			return 0
	def derivative(self, fx):
		if (fx) >= 0:
			return 1
		else:
			return 0
	def buildActivations(n):
		rv = []
		for i in range(0,n):
			rv.append(ReLUActivation())
		return rv
		
class GaussRBFActivation(Activation):
	def __init__(self):
		pass
	def xfire(self, diff, sigma):
		#print("D"+str(diff))
		#print("S"+str(sigma))
		#print(np.exp(-(diff*diff/(2*sigma*sigma))))
		return np.exp(-(diff*diff/(2*sigma*sigma)))
	def derivative(self, fx):
		return None
	def buildActivations(n):
		rv = []
		for i in range(0,n):
			rv.append(GaussRBFActivation())
		return rv