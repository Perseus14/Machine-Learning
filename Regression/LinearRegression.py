import math
import numpy as np

class LinearRegression:
	class SimpleRegression:
		def __init__(self, X, Y):	#Input X and Y: A list of numbers
			self.X = X
			self.Y = Y
			self.predictions = []
			self.b0, self.b1 = self.coefficients()		##Fitting the model

		def mean(self, values):
			return sum(values)/float(len(values))
	
		def variance(self, values, mean):
			return sum([(x-mean)**2 for x in values])
	
		def covariance(self, mean_x, mean_y):
			covar = 0.0
			for i in range(len(self.X)):
				covar += (self.X[i] - mean_x) * (self.Y[i] - mean_y)
			return covar

		def coefficients(self):
			## B1 = (S(x*y) - S(x)*S(y))	/(S(x*x) - S(x)*S(x)) = covar(x,y)/var(x)		S = Summation
			## B0 = (S(y) - b1*S(x))/n	= mean(y) - b1*mean(x)							(Following partial differentiation of least squared error)
			x_mean, y_mean = self.mean(self.X), self.mean(self.Y)
			b1 = self.covariance(x_mean, y_mean) / self.variance(self.X, x_mean)
			b0 = y_mean - b1 * x_mean
			return [b0, b1]	

		def predict(self, test_x): #Input: X values of the test dataset, Output: Y Predictions
			pred_y = [] 
			for x in test_x:
				yhat = self.b0 + self.b1*x
				pred_y.append(yhat)
			self.predictions = pred_y	
			return pred_y
	
		def accuracy(self, test_y):	#Input: Y values of the test dataset, Output: RMSE
			sum_squared_err = 0		
			for id_y in xrange(len(test_y)):
				sum_squared_err+=(test_y[id_y] - self.predictions[id_y])**2
			mean_squared_err = 	sum_squared_err/float(len(test_y))	
			root_mean_squared_err = math.sqrt(mean_squared_err)
			return root_mean_squared_err
	
	class PolynomialRegression:
		def __init__(self, X, Y):	
			'''
			Input: Y = a coloumn list of numbers [1,
											  2,
											  3], 
			 X = a matrix, each row is one datapoint [[1,x01,x02,x03...],
														[1,x11,x12,x12...],
														[1,x21,x22,x23...]]
			'''		
			X = np.c_[np.mat(np.ones(X.shape[0])).transpose(),X] ##Adding a coloumn of ones to X as coeff for B0
			self.X = X 
			self.Y = Y
			self.predictions = []
			self.coeff_b = self.coefficients()	
		
		def coefficients(self):
			X_t = self.X.transpose()
			return np.matmul(np.matmul(np.linalg.inv(np.matmul(X_t,self.X)),X_t),self.Y) #B = ((XT)*X)^(-1)*(XT)*y

		def predict(self, test_x):	#Input: A matrix X 	
			test_x = np.c_[np.mat(np.ones(test_x.shape[0])).transpose(),test_x]	##Adding a coloumn of ones to X as coeff for B0
			pred_y = np.matmul(test_x, self.coeff_b) 
			self.predictions = pred_y
			return pred_y

		def accuracy(self, test_y): #Input: A list Y ([1,2,3...])
			err = self.predictions - test_y
			sum_squared_err = 0
			for val in err:
				sum_squared_err += val**2				
			mean_squared_err = 	sum_squared_err/float(test_y.shape[0])
			root_mean_squared_err = math.sqrt(mean_squared_err)
			return root_mean_squared_err				
