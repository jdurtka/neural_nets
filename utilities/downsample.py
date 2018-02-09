# CSCI 447 Machine Learning
# Project #2 - Neural Networks

# Author: James Durtka
# In collaboration with: Mitch Baker, Wang Hongchuan

import sys
import logging
import argparse
import os
import time

import pandas as pd
import numpy as np

def main():
	#parse the commandline (doc at https://docs.python.org/2/library/argparse.html#module-argparse)
	ap = argparse.ArgumentParser(description="Train neural and RBF nets on a dataset")
	ap.add_argument('-i', action='store', help='data from the indicated file (must be CSV with all numeric attributes and the rightmost or last column being the target value to predict)', required=True)
	ap.add_argument('-proportion', action='store', help='how much of the dataset to keep e.g. 0.05 for 5%', required=True)
	
	args = ap.parse_args()
		
	inputFile = args.i
	outputFile = 'ds_' + args.i
	proportion = float(args.proportion)
	
	#setup logging
	logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='debug.log',
                    filemode='w')
	consoleLogFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s")
	rootLogger = logging.getLogger()
	
	consoleHandler = logging.StreamHandler()

	consoleHandler.setFormatter(consoleLogFormatter)
	
	rootLogger.addHandler(consoleHandler)

	print("Reading input file...")
	data = pd.read_csv(inputFile, header=None)
	Xs_t = data.as_matrix()
	
	#form a random permutation
	print("Forming a permutation...")
	np.random.seed(int(time.time()))
	permute = np.random.permutation(len(Xs_t))
	Xs_t = Xs_t[permute]
	
	print("Selecting data in proportion " + str(proportion))
	Xs = Xs_t[0:int(proportion*len(Xs_t))]
	
	print("Writing result to output file " + outputFile)
	f = open(outputFile, 'w')
	
	for x in Xs:
		for i in range(0, len(x)):
			f.write(str(x[i]))
			if i < len(x)-1:
				f.write(',')
			else:
				f.write('\n')
	
	f.close()
main()