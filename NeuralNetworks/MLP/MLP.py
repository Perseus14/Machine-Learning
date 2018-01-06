import numpy as np


def sigmoid(A):
	return 1.0/(1 + np.exp(-A))
		 
			 
def softmax(A):
	exp_val = np.exp(A)
	prob = exp_val/np.sum(exp_val)
	return exp_val, prob
		
class Layer:
	class Weights:
		def __init__(self, n_hidden_1, n_hidden_2, mu = 0, sigma = 0.01):
			self.n_hidden_1 = n_hidden_1
			self.n_hidden_2 = n_hidden_2
			self.mu = mu
			self.sigma = sigma
			self.weights = np.random.normal(self.mu,self.sigma,(self.n_hidden_1, self.n_hidden_2)) #Initalise a Random Gaussian matrix of size (n1, n2)

	class Bias:
		def __init__(self, n_hidden, mu = 0, sigma = 0.01):
			self.n_hidden = n_hidden
			self.mu = mu
			self.sigma = sigma
			self.bias = np.random.normal(self.mu,self.sigma,(self.n_hidden, 1))			#Initalise a Random Gaussian matrix of size (n, 1)

	def __init__(self, n_hidden_1, n_hidden_2, act_fn = sigmoid):
		self.n_hidden_1 = n_hidden_1
		self.n_hidden_2 = n_hidden_2													
		self.weights = Layer.Weights(self.n_hidden_1, self.n_hidden_2).weights			# Matrix(n1,n2)
		self.bias = Layer.Bias(self.n_hidden_2).bias									# Coloumn matrix (n2,1)
		self.output = []
		self.activation_fn = act_fn
		self.activation_output = []
		
	def evaluate(self, X):																# X - Coloumn matrix (n1,1)
		self.output = np.matmul(X.T, self.weights).T + self.bias	 				# [(1, n1) * (n1,n2)]^T + (n2,1)
		return self.output

	def activation(self, A):
		self.activation_output = self.activation_fn(A)
		return self.activation_output

				 
#TODO Backpropagation, Gradient Descent, Update Weights

inp = np.array([[1],
				[2],
				[3],
				[4],
				[5]])
n_input = 5
n_hidden_1 = 2
n_hidden_2 = 2
n_output = 2
nepochs = 100
learning_rate = 0.001
##Initialise Layers

layer1 = Layer(n_input, n_hidden_1)
layer2 = Layer(n_hidden_1,n_hidden_2)
layer3 = Layer(n_hidden_2,n_output)

def forwardPropagation():
	layer1_output = layer1.evaluate(inp) 
	layer1_act = layer1.activation(layer1_output)
	
	layer2_output = layer2.evaluate(layer1_act) 
	layer2_act = layer2.activation(layer2_output)
		 
	layer3_output = layer3.evaluate(layer2_act)	 
	score,prob = softmax(layer3_output)

def backPropagation():
	

def gradientDescent():
	for epoch in xrange(nepochs):
		forwardPropagation()	
		backPropagation()	
	
	print layer1_act,"\n Layer 2: \n", layer2_act, "\n Score: \n",score,"\n Prob: \n", prob
