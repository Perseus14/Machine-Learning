from LogisticRegression import LogisticRegression
import numpy as np
from sklearn import linear_model, datasets

iris = datasets.load_iris()

X = iris.data[:70, :2]
Y = iris.target[:70]

test_X = iris.data[70:100, :2]
test_Y = iris.target[70:100]

LR_Y = np.reshape(Y, (Y.shape[0],1))

LR = LogisticRegression(X,LR_Y, alpha = 0.01, nepochs = 10000)
pred = LR.predict(test_X)
label_pred = LR.classify(pred)

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
sk_pred = logreg.predict(test_X)

sk_pred = np.reshape(sk_pred, (sk_pred.shape[0],1))

print LR.accuracy(test_X,sk_pred)
