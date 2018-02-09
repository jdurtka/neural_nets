# CSCI 447 Machine Learning
# Project #2 - Neural Networks
# Sampling function that generates data set for neural network training and testing
# Author: Hongchuan Wang
# In collaboration with: Mitch Baker, James Durtka


import numpy as np
import math
from itertools import product
import random
import csv

### RosenBrock function
def rosenBrock(X):

	Y = np.zeros((X.shape[0],1))

	for i in range(0,X.shape[1]-1):

		# first term is the (1- Xi)^2
		first_term = np.square((1-X[:,i,None]))
		# second term
		second_term = 100 * np.square(X[:,i+1,None] - np.square(X[:,i,None]))

		Y += np.add(first_term,second_term)
	
	return Y


#######   Function that writes the samples into csv file  ##########
def csv_write(X,target_Y, data_set_index,sampling_method):
	
	out = sampling_method + str(data_set_index) + ".csv"
	with open(out,"w") as f:
		writer = csv.writer(f)
		writer.writerows(np.append(X,target_Y,axis=1))



def grid_sampling(num_point_per_dim, upperbound,lowerbound,n_dim):
	total_num_samples = int(math.pow(num_point_per_dim,n_dim))
	interval = (upperbound - lowerbound) / num_point_per_dim
	# a small value is added to upperbound so the results includes upperbounds
	delta = 0.0005
	x = np.arange(lowerbound,upperbound + delta,interval)

	if n_dim == 2:
		iterator = list(product(x,x))
	elif n_dim == 3:
		iterator = list(product(x,x,x))
	elif n_dim == 4:
		iterator = list(product(x,x,x,x))
	elif n_dim == 5:
		iterator = list(product(x,x,x,x,x))
	elif n_dim == 6:
		iterator = list(product(x,x,x,x,x,x))
	
	sample_X = np.asarray(iterator)
	target_Y = rosenBrock(sample_X)

	return sample_X,target_Y

def random_sampling(num_point_per_dim, upperbound,lowerbound,n_dim):

	total_num_samples = int(math.pow(num_point_per_dim,n_dim));
	X = np.random.uniform(lowerbound,upperbound,size=(total_num_samples,n_dim))
	Y = rosenBrock(X)

	return X,Y


for n in [2, 3, 4, 5, 6]:
	print("Producing data for case n=" + str(n))
	##  Test case #1 for random_sampling function
	X1,target_Y1 = random_sampling(10,1.5,-1.5,n)
	csv_write(X1,target_Y1, 1, str(n)+"d_random_sampling")

	## Test case for random_sampling function
	X2,target_Y2 = grid_sampling(10,1.5,-1.5, n)
	csv_write(X2,target_Y2,1,str(n)+"d_grid_sampling")



