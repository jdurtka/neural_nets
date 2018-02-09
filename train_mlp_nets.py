# CSCI 447 Machine Learning
# Project #2 - Neural Networks

# Author: James Durtka
# In collaboration with: Mitch Baker, Wang Hongchuan
from activations import *
from neural_net import *
from rbf_net import *

import sys
import logging
import argparse
import os
import time

import pandas as pd
import numpy as np

#number of cross validations to perform for each dataset, each net
KFOLDS = 5

class NetInstance(object):
	def __init__(self, net, name):
		self.net = net
		self.name = name
		
def l2_norm(x1, x2):
	return np.sqrt(np.dot(np.abs(x2-x1), (np.abs(x2-x1))))

def l1_norm(x1, x2):
	return np.sum(np.abs(x2-x1))
		
def findMaxes(X, k, metric=l2_norm, debug=False):
	maxdist = 0
	for x in X:
		for y in X:
			if not (np.array_equal(x,y)):
				if(metric(x,y) > maxdist):
					maxdist = metric(x,y)
	return maxdist*np.ones(k)

def main():
	#parse the commandline (doc at https://docs.python.org/2/library/argparse.html#module-argparse)
	ap = argparse.ArgumentParser(description="Train neural and RBF nets on a dataset")
	ap.add_argument('-i', action='store', help='data from the indicated file (must be CSV with all numeric attributes and the rightmost or last column being the target value to predict)', required=True)
	ap.add_argument('-od', action='store', help='directory for output files', required=True)
	ap.add_argument('-epochs', action='store', help='number of epochs to run', required=True)
	ap.add_argument('--verbose', '-v', action='count', help='specify level of verbosity, up to 3 (i.e. -vvv)')
	
#	testing = False
#	if testing:
#		args = ap.parse_args('-fi test.csv -fo test.arff -fmt_in=c'.split())
#	else:
	args = ap.parse_args()
		
	inputFile = args.i
	outputDirectory = args.od
	
	try:
		os.stat(outputDirectory)
	except:
		os.mkdir(outputDirectory)
	
	#setup logging
	logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=outputDirectory + 'debug.log',
                    filemode='w')
	consoleLogFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s")
	rootLogger = logging.getLogger()
	
	consoleHandler = logging.StreamHandler()

	if args.verbose == None:
		verb = 0
	else:
		verb = args.verbose*10
		if verb > 40:
			verb = 40
		consoleHandler.setLevel(40 - verb)


	consoleHandler.setFormatter(consoleLogFormatter)
	
	rootLogger.addHandler(consoleHandler)

	#Log the inputs we've received so far
	logging.info("Input file: " + inputFile)
	logging.info("Verbosity level: " + str(verb))
	
	data = pd.read_csv(inputFile, header=None)
	
	#split off the Xs and ys
	#ASSUMPTION: here we assume there is only ONE target value and it is the very last column in the CSV
	#this assumption may not hold in other projects!
	ys_t = data.iloc[:,-1].as_matrix()
	Xs_t = data.iloc[:,:-1].as_matrix()
	
	#form a random permutation
	np.random.seed(int(time.time()))
	permute = np.random.permutation(len(Xs_t))
	Xs_t = Xs_t[permute]
	ys_t = ys_t[permute]
	
	#form a train-test split
	trn_size = int((0.8)*len(permute))
	tst_size = int((0.2)*len(permute))
	margin = (trn_size+tst_size-len(permute))
	trn_size += margin
	
	#create the train/test split (never actually used in this project)
	Xtst = Xs_t[trn_size:trn_size+tst_size-1]
	ytst = ys_t[trn_size:trn_size+tst_size-1]
	Xs = Xs_t[0:trn_size-1]
	ys = ys_t[0:trn_size-1]
	
	#each y is only a single element, but we need each to be separately packaged into its own matrix
	ys_ = []
	for y in ys:
		ys_.append([y])
	ys = ys_
	
	Xtrndf = pd.DataFrame(Xs)
	ytrndf = pd.DataFrame(ys)
	Xtstdf = pd.DataFrame(Xtst)
	ytstdf = pd.DataFrame(ytst)
	Xtrndf.to_csv(outputDirectory+'X_train.csv', header=None, index=None)
	ytrndf.to_csv(outputDirectory+'y_train.csv', header=None, index=None)
	Xtstdf.to_csv(outputDirectory+'X_test.csv', header=None, index=None)
	ytstdf.to_csv(outputDirectory+'y_test.csv', header=None, index=None)
	
	#Now, setup each neural net we wish to train:
	mlpnets = []
	rbfnets = []
	
	rndseed = int(time.time())
	
	starttime = time.time()
	
	logging.info("Building MLP networks...")
	#produce networks for each possible number of hidden layers
	for num_hiddens in range(0,3):
		#if we're building hidden layers, do so
		if num_hiddens > 0:
			if num_hiddens == 1:
				for i in [10, 100]:
					thismlp = NetInstance(net=Net(len(Xs[0]), 1, debug=0, seed=rndseed, weightScale=0.7), name='1h_'+str(i)+'/')
					logging.info("Building " + thismlp.name)
					thismlp.net.addLayers([i], [TanhActivation.buildActivations(i)], Activation.buildActivations(1))
					mlpnets.append(thismlp)
			elif num_hiddens == 2:
				for i in [10, 100]:
					for j in [10, 100]:
						thismlp = NetInstance(net=Net(len(Xs[0]), 1, debug=0, seed=rndseed, weightScale=0.7), name='2h_'+str(i)+'_'+str(j)+'/')
						logging.info("Building " + thismlp.name)
						thismlp.net.addLayers([i, j], [TanhActivation.buildActivations(i), TanhActivation.buildActivations(j)], Activation.buildActivations(1))
						mlpnets.append(thismlp)
						
		#no hidden layers
		else:
			thismlp = NetInstance(net=Net(len(Xs[0]), 1, debug=0, seed=rndseed, weightScale=0.7), name='0h/')
			logging.info("Building " + thismlp.name)
			thismlp.net.addLayers([], [], Activation.buildActivations(1))
			mlpnets.append(thismlp)

	for netinst in mlpnets:
		relstart = time.time()
		fulldir=outputDirectory+netinst.name
		try:
			os.stat(fulldir)
		except:
			os.mkdir(fulldir)
		logging.info("TRAINING NET: " + netinst.name)
		
		netinst.net.xval_train(Xs, ys, eta=0.03, alpha=0.008, epochs=int(int(args.epochs)/KFOLDS), debug=1, saveStep=int(args.epochs)/10, savePrefix='weights_', logging=logging, folds=KFOLDS, saveDir=fulldir)
		
		#Save the final state (although states are also saved periodically along the way)
		netinst.net.saveState(fulldir+"final_state.pkl")

		logging.info("MLP training time elapsed: " + str(time.time()-relstart))

	endtime = time.time()
	
	logging.info("COMPLETE")
	logging.info("\n")
	logging.info("--------------------------------------")
	
	
if __name__ == '__main__':
	main()