from LinearRegression import LinearRegression
import numpy as np

X = [1,2,4,3,5]
Y = [1,3,3,2,5]

LR = LinearRegression.SimpleRegression(X[:3],Y[:3])		##	Fit model on first 3 datapoints
pred = LR.predict(X[3:])				##  Predict for the last 2 datapoints
rmse = LR.accuracy(Y[3:])				##	Calc. accuracy for the predictions

print "Predictions:",pred
print "RMSE: %f"%rmse


X = np.mat('0.44, 0.68; 0.99 0.23')
Y = np.mat('109.85, 155.72').transpose()
LR = LinearRegression.PolynomialRegression(X,Y)		##	Fit model 
pred = LR.predict(np.mat('0.49 0.18'))				##  Predict
#rmse = LR.accuracy(Y)				##	Calc. accuracy for the predictions

print "Predictions:",pred
#print "RMSE: %f"%rmse


