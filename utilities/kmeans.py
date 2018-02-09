# CSCI 447 Machine Learning
# Project #2 - Neural Networks
#
# Auxiliary program: k-Means Clustering

# Author: James Durtka
# In collaboration with: Mitch Baker, Wang Hongchuan
import numpy as np
import time

import matplotlib.pyplot as plt

def l2_norm(x1, x2):
	return np.sqrt(np.dot(np.abs(x2-x1), (np.abs(x2-x1))))

def l1_norm(x1, x2):
	return np.sum(np.abs(x2-x1))

class KMeansClusterer(object):
	def __init__(self, k, features=1, seed=0, scale=1):
		if seed == None:
			seed = int(time.time())
		np.random.seed(seed)
		self.features = features
		
		self.k = k
		
		self.means = []
		#initialize means randomly at first
		for i in range(0,k):
			self.means.append(scale*2*(np.random.rand(features)-0.5))
			
		self.seed = seed
		pass
		
	def plotClusters(self, X):
		Xcats = self.encode(X)
		for i, vector in enumerate(X):
			#print("V " + str(vector))
			(x, y) = vector.tolist()
			plt.plot(x, y, marker=str(Xcats[i]+1))
		plt.show()
		
	def pointsPerCat(self, X):
		Xcats = self.encode(X)
		
		tots = []
		for i in range(0,self.k):
			tots.append(0)
		for c in Xcats:
			tots[c] += 1
		return tots

	#assumption: X contains a set of example vectors
	def fit (self, X, metric=l2_norm, iterations=10, debug=False, showIntermediate=False):
		if showIntermediate:
		#if True:
			print("INITIAL RANDOM MEANS")
			for mean in self.means:
				print(mean)
			print("")
		
		#for as many iterations
		for q in range(0,iterations):
			#setup some storage arrays
			xsums = []
			xtots = []
			#for each of the k means
			for i in range(0,self.k):
				xsums.append([])
				xtots.append(0)
		
			Xcats = self.encode(X,metric=metric,debug=debug)
			
			#for each category, update its mean based on the points
			for i, xc in enumerate(Xcats):
				if xsums[xc] == []:
					xsums[xc] = np.array(X[i])
				else:
					xsums[xc] += np.array(X[i])
				xtots[xc] += 1
			#produce the new means
			for i in range(0, self.k):
				#if some mean was assigned nothing, set it to the mean of all the others
				if xtots[i] == 0:
					fixsum = []
					fixtot = 0
					for j, s in enumerate(xsums):
						if s != []:
							if fixsum == []:
								fixsum = s
							else:
								fixsum = fixsum+s
							fixtot = fixtot+xtots[j]
				else:
					self.means[i] = xsums[i]/xtots[i]
					
			if showIntermediate:
				print("ITERATION:")
				for mean in self.means:
					print(mean)
				print("")
			
		if showIntermediate:
			print("FINAL CENTROIDS:")
			for mean in self.means:
				print(mean)
			print("")
		return self.means
		
	def encode(self, X, metric=l2_norm, debug=False):
		Xcats = []
		for i in range(0,len(X)):
			Xcats.append(0)
		
		#for each datapoint
		for i, x in enumerate(X):
			bestcat = 0
			bestmean = metric(x,self.means[0])
			#find best category
			for cat, mean in enumerate(self.means):
				#based on the metric provided
				if (metric(x,mean)) < bestmean:
					bestcat = cat
					bestmean = metric(x,mean)
			Xcats[i] = bestcat
		
		if debug:
			print(Xcats)
		
		#return the encoded datapoints
		return Xcats
							
					
		
	#compute the RMSE of the classifier on the given dataset with the current set of means
	def rmse(self, X):
		Xmeans = []
		Xcats = self.encode(X)
		for i in range(0, len(Xcats)):
			Xmeans.append(self.means[Xcats[i]])
			
		err = []
		for x, xm in zip(X, Xmeans):
			diff = np.array(X)-np.array(Xmeans)
			err.append(l2_norm(x,xm))
		
		return(np.sqrt(np.dot(np.abs(err),np.abs(err))))
		
	
#Generates n*k d-dimensional points with k means generated in the range +/- scale and individual points differing
#from the means by +/- rscale
def kmeans_test(k, seed=0, scale=1, d=2, n=30, rscale=0.1, debug=False):
	np.random.seed(seed)
	X = []
	for i in range(0,k):
		thismean = scale*2*(np.random.rand(d)-0.5)
		if debug:
			print("Mean #" + str(i) + ": " + str(thismean))
		for j in range(0,n):
			X.append(thismean + rscale*2*(np.random.rand(d)-0.5))
	return X

def main():
	#Testing k means
	#Pick a k and generate n datapoints with k means
	k = 3
	d = 2
	scale = 100
	rscale = 0.3
	seed = None

	clusterer = KMeansClusterer(k, features=d, seed=1, scale=scale)

	#Note that it is necessary to do this, since sometimes the algorithm fails due to none of the points
	#lying within one of the randomly initialized means in the first iteration.
	#When this happens, None will be returned and we need to try again.
	#generate 30 points per mean
	X = kmeans_test(k=k, seed=None, scale=scale, d=d, n=30, rscale=rscale, debug=True)

	means = clusterer.fit(X=X, metric=l2_norm, iterations=10, debug=False, showIntermediate=True)
		
	print("RMSE: " + str(clusterer.rmse(X)))
	maxes = clusterer.findMaxes(X=X, metric=l2_norm, debug=False)
	print('d_max by cluster: ' + str(maxes))
	print('Points assigned to each centroid:')
	print(clusterer.pointsPerCat(X))
	
	
	clusterer.plotClusters(X)


if __name__ == '__main__':
	main()