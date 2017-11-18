from LinearRegression import LinearRegression

X = [1,2,4,3,5]
Y = [1,3,3,2,5]

LR = LinearRegression(X[:3],Y[:3])		##	Fit model on first 3 datapoints
pred = LR.predict(X[3:])				##  Predict for the last 2 datapoints
rmse = LR.accuracy(Y[3:])				##	Calc. accuracy for the predictions

print "Predictions:",pred
print "Accuracy: %f"%rmse
