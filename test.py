from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from ML import *

#### KNN ####

# cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# knn = KNN(X_train=X_train, y_train=y_train)
# predictions = knn.predict(X_test)

# acc = np.sum(predictions == y_test) / len(y_test)
# print(acc)
# ---------------------

#### LINEAR REGRESSION ####
# X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# # print(X)
# regressor = LinearRegression(X_train, y_train)
# predicted = regressor.predict(X_test)

# plt.figure()
# plt.scatter(X_test, predicted, color='b', marker='o', s=30)
# plt.plot(X_test, predicted, color='r')
# plt.show()

# # -----------------------

#### LOGISTIC REGRESSION ####
# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
# # X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1234)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# # print(bc.target)

# regressor = LogisticRegression(X_train, y_train, 0.0001)
# predicted = regressor.predict(X_test)

# acc = accuracy_sum(y_test, predicted) * 100
# print(acc)
# # -----------------------

#### NN ####

# numbers = datasets.load_digits()
# X, Y = numbers.data, numbers.target
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# X_test, Y_test = X_test.T, one_hot(Y_test)


# n = NN(X_train, Y_train, [X_train.shape[1], 64, 10], "relu", "sigmoid", 0.0075, 2500)

# import random
# for j in range(15):
    # i=random.randint(0,X_train.shape[0])
    # img=numbers.images[i].reshape((64,1)).T
    # img=img.T
    # predicted_digit = n.predict(img)

    # print('Predicted digit is : '+str(predicted_digit))
    # print('True digit is: '+ str(Y[i]))

#### ? ####