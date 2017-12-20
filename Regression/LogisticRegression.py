import math
import numpy as np

class LogisticRegression:
	### log(p(x)/log(1-p(x)) = B0 + B1*x
	'''
		Input:
			X = [[x00,x01,x02,x03],
				 [x10,x11,x12,x13],
				  ....
				 [xn0,xn1,xn2,xn3]]
			Y = [1,
				 0,
				 1,
				 0]
				 
			Transform X to add biases
			
			X = [[1,x00,x01,x02,x03...x0m],
				 [1,x10,x11,x12,x13...x1m],
				  ....
				 [1,xn0,xn1,xn2,xn3...xnm]]
				 	 	  
	'''
	def __init__(self, X, Y, alpha = 0.001, nepochs = 10000):
		X = np.c_[np.mat(np.ones(X.shape[0])).transpose(),X]
		self.X = X
		self.Y = Y
		self.alpha = alpha
		self.nepochs = nepochs
		self.weights = self.train()		
		'''
			Weights: [b,
					  w0,
					  w1,
					  ...
					  wm]
			
		'''
	def sigmoid(self, Z):
		return 1.0/(1 + np.exp(-Z))
		
	def predict(self, test_x, weights = None):
		if(weights is None):
			weights = self.weights
		if(test_x.shape[1]!=self.X.shape[1]):
			test_x = np.c_[np.mat(np.ones(test_x.shape[0])).transpose(),test_x]
		Y_hat = np.matmul(test_x,weights)
		return self.sigmoid(Y_hat)
	
	def costFunction(self, weights):
		##Cross Entropy or Log Loss
		##Can be used to find the error with particular weight 
		'''
			Cost = y*(log(a)) + (1-y)*(log(1-a))
		'''		
		A = self.predict(self.X, weights)
		N = self.X.shape[0]
		cost = (self.Y*(np.log(A)) + (1-self.Y)*(np.log(1-A)))/float(N)
		return cost	
		
	def naiveGradientDescent(self, weights):
		'''
			d(C)/d(W[j]) = avg(X[:][j] * (A-Y)) Multiply all coefficients of Wj with Error ans sum it up
		'''
		A = self.predict(self.X, weights)
		N = self.X.shape[0]
		partialDerivative = np.matmul(self.X.T,(A-self.Y))/N
		return partialDerivative
		
	def train(self):
		weights = np.zeros((self.X.shape[1],1)) ##X = (n,m) W = (m,1)	
		total_epochs = self.nepochs
		for epoch in xrange(total_epochs):
			weights+= (-1)*self.alpha*self.naiveGradientDescent(weights)	##Small increments in NEGATIVE direction of gradient
		return weights	
		
	def threshold(self,prob):
		if prob >= 0.5: 
			return 1
		return 0
	
	def classify(self, predictions):
		thresh = np.vectorize(self.threshold)
		return thresh(predictions)	
	
	def accuracy(self, test_x, test_y):
		pred_prob = self.predict(test_x)
		pred_labels = self.classify(pred_prob) 
		diff = pred_labels - test_y
		return 1.0 - (float(np.count_nonzero(diff)) / len(diff))
			
