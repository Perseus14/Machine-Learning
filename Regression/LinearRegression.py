import math

class LinearRegression:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
		self.predictions = []

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
		b0, b1 = self.coefficients()
		for x in test_x:
			yhat = b0 + b1*x
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
