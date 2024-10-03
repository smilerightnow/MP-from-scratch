import numpy as np
from collections import Counter

def one_hot(Y):
	one_hot_Y = np.zeros((Y.size, Y.max() + 1))
	one_hot_Y[np.arange(Y.size), Y] = 1
	one_hot_Y = one_hot_Y.T
	return one_hot_Y

def distance_between_points(p1, p2):
	# p1 and p2 are arrays.
	# AKA: Multidimensional space: √((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 + ...)
	return np.sqrt(np.sum((p1-p2)**2))

def mean_sequared_error(y_true, y_predicted):
	return np.mean((y_true-y_predicted)**2)

def mean_abs_error(y_true, y_predicted):
	return np.mean(np.abs(y_true-y_predicted))

def accuracy_sum(y_true, y_predicted): ## for binary classification
	return np.sum(y_true == y_predicted) / len(y_true)

def sigmoid(x): ## for binary classification
	return 1 / (1 + np.exp(-x))

def softmax(x): ## for multi class classification
	return np.exp(x)/np.sum(np.exp(x))

def tanh(x):
	return np.tanh(x)

def relu(x):
	return np.maximum(x, 0)

def d_tanh(x):
	return 1 - np.power(np.tanh(x), 2)

def d_relu(x):
	return np.array(x>0, dtype=np.float32)


class KNN:
	def __init__(self, X_train, y_train, k=3):
		self.k=k
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X):
		predicted_labels = []

		for x in X:
			# compute distances
			distances = [distance_between_points(x, x_train) for x_train in self.X_train]
			# get k nearest samples, labels
			k_indices = [distances.index(d) for d in sorted(distances)[:self.k]]
			k_nearest_labels = [self.y_train[i] for i in k_indices]
			# get most common class label
			most_common = Counter(k_nearest_labels).most_common(1)[0][0]

			predicted_labels.append(most_common)

		return np.array(predicted_labels)


class LinearRegression:
	def __init__(self, X_train, y_train, learning_rate=0.01, n_iters=1000):
		self.learning_rate = learning_rate
		self.n_iters = n_iters

		self.X_train = X_train
		self.y_train = y_train

		n_samples, n_features = self.X_train.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		self.errors = []

		for i in range(self.n_iters):
			y_predicted = np.dot(self.X_train, self.weights) + self.bias # ŷ = Xw + b

			dw = (1/n_samples) * np.dot(self.X_train.T, (y_predicted - self.y_train))
			db = (1/n_samples) * np.sum(y_predicted - self.y_train)

			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

			# if i%10==0:
				# error_rate = mean_abs_error(self.y_train, y_predicted)
				# self.errors.append(error_rate)
				# print(error_rate)

	def predict(self, X):
		y_predicted = np.dot(X, self.weights) + self.bias # ŷ = Xw + b
		return y_predicted


class LogisticRegression: ## binary classifications. output: 1 or 0
	def __init__(self, X_train, y_train, learning_rate=0.001, n_iters=1000):
		self.learning_rate = learning_rate
		self.n_iters = n_iters

		self.X_train = X_train
		self.y_train = y_train

		n_samples, n_features = self.X_train.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		self.errors = []

		for i in range(self.n_iters):
			y_predicted_linear = np.dot(self.X_train, self.weights) + self.bias # ŷ = Xw + b
			y_predicted = sigmoid(y_predicted_linear)

			dw = (1/n_samples) * np.dot(self.X_train.T, (y_predicted - self.y_train))
			db = (1/n_samples) * np.sum(y_predicted - self.y_train)

			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

			# if i%10==0:
				# y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
				# error_rate = 1 - accuracy_sum(self.y_train, y_predicted_cls)
				# self.errors.append(error_rate)
				# print(error_rate)

	def predict(self, X):
		y_predicted_linear = np.dot(X, self.weights) + self.bias # ŷ = Xw + b
		y_predicted = sigmoid(y_predicted_linear)
		y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]

		return y_predicted_cls


class NN:
	def __init__(self, X_train, y_train, layer_dims, activation, out_activation, learning_rate=0.001, n_iters=1000):
		# layer_dims = [X_train.shape[1], 100,...., 200, y_train.shape[0]]

		self.learning_rate = learning_rate
		self.n_iters = n_iters

		self.X_train = X_train.T
		self.y_train = one_hot(y_train)

		self.activation = activation
		self.out_activation = out_activation

		n_samples, n_features = self.X_train.shape
		self.errors = []

		self.output_size = self.y_train.shape[0]

		self.params = {}

		for i in range(1, len(layer_dims)):
			self.params["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
			self.params["b"+str(i)] = np.zeros((layer_dims[i], 1))

		for i in range(self.n_iters):
			AL, forward_cache = self.forward_propagation(self.X_train)
			cost = self.cost(AL)
			grads = self.backward_propagation(AL, forward_cache)
			self.update_parameters(grads)

			if i%10 == 0:
				print(cost)

	def predict(self, X):

		m = X.shape[1]
		y_pred, caches = self.forward_propagation(X)

		if self.output_size == 1:
			y_pred = np.array(y_pred > 0.5, dtype = 'float')
		else:
			y_pred = np.argmax(y_pred, 0)

		return y_pred

	def forward_propagation(self, X):
		forward_cache = {}

		forward_cache["A0"] = X ## input layer

		L = len(self.params)//2

		for i in range(1, L): ## hidden layers
			l = str(i)
			forward_cache["Z"+l] = self.params["W"+l].dot(forward_cache["A"+str(i-1)]) + self.params["b"+l] ## Zi = Wi * Ai-1 + bi

			if self.activation == "relu":
				forward_cache["A"+l] = relu(forward_cache["Z"+l])
			if self.activation == "tanh":
				forward_cache["A"+l] = tanh(forward_cache["Z"+l])

		## output layer
		forward_cache["Z"+str(L)] = self.params["W"+str(L)].dot(forward_cache["A"+str(L-1)]) + self.params["b"+str(L)] ## Zi = Wi * Ai-1 + bi

		if self.out_activation == "sigmoid":
			forward_cache["A"+str(L)] = sigmoid(forward_cache["Z"+str(L)])
		else:
			forward_cache["A"+str(L)] = softmax(forward_cache["Z"+str(L)])

		return forward_cache["A"+str(L)], forward_cache

	def cost(self, AL):
		m = self.output_size

		if self.output_size == 1: ## binary classification
			cost = (1./m) * (-np.dot(self.y_train,np.log(AL).T) - np.dot(1-self.y_train, np.log(1-AL).T))
		else:
			cost = -(1./m) * np.sum(self.y_train * np.log(AL))

		cost = np.squeeze(cost) ## flattening the list
		return cost

	def backward_propagation(self, AL, forward_cache):
		grads = {}
		L = len(self.params)//2
		m = AL.shape[1]

		grads["dZ"+str(L)] = AL - self.y_train
		grads["dW"+str(L)] = (1./m) * np.dot(grads["dZ"+str(L)], forward_cache["A"+str(L-1)].T)
		grads["db"+str(L)] = (1./m) * np.sum(grads["dZ"+str(L)], axis=1, keepdims=True)

		for i in reversed(range(1, L)):
			if self.activation == "relu":
				grads["dZ"+str(i)] = np.dot(self.params["W"+str(i+1)].T, grads["dZ"+str(i+1)])*d_relu(forward_cache["A"+str(i)])
			if self.activation == "tanh":
				grads["dZ"+str(i)] = np.dot(self.params["W"+str(i+1)].T, grads["dZ"+str(i+1)])*d_tanh(forward_cache["A"+str(i)])
			grads["dW"+str(i)] = (1./m) * np.dot(grads["dZ"+str(i)], forward_cache["A"+str(i-1)].T)
			grads["db"+str(i)] = (1./m) * np.sum(grads["dZ"+str(i)], axis=1, keepdims=True)

		return grads

	def update_parameters(self, grads):
		L = len(self.params)//2
		for i in range(1, L+1):
			self.params["W"+str(i)] = self.params["W"+str(i)] - self.learning_rate*grads["dW"+str(i)]
			self.params["b"+str(i)] = self.params["b"+str(i)] - self.learning_rate*grads["db"+str(i)]