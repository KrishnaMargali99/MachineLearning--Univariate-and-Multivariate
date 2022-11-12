import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Name: Goparapu Krishna Margali, kg4060@rit.edu
CSCI 635: Introduction to Machine Learning
Problem 2: Polynomial Regression &

@author/lecturer - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself
#       (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p1_fit'  # will add a unique sub-string to output of this program
degree = 15 # p, order of model
# degree=[degree]
beta = 0.001  # regularization coefficient
alpha = 0.01  # step size coefficient
eps = 0  # controls convergence criterion
n_epoch = 1000
# number of epochs (full passes through the dataset)


def regress(X, theta):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    return theta[0] + np.dot(X, theta[1].T)


def gaussian_log_likelihood(mu, y):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    return np.sum(np.square(mu - y))


def computeCost(X, y, theta, beta):  ## loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
    # WRITEME: write your code here to complete the routine
    mu = regress(X, theta)
    log_loss = (gaussian_log_likelihood(mu, y) + beta*(np.sum((theta[1]**2))) / (2 * X.shape[0])
    return log_loss


def computeGrad(X, y, theta, beta):
    ############################################################################
    # WRITEME: write your code here to complete the routine (
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)

    db = (1 / X.shape[0]) * np.sum((regress(X, theta) - y))
    dw = (np.dot(X.T, (regress(X, theta) - y)).T + beta * theta[1]) / X.shape[0]
    return db, dw


def X_transform(X, degree):
    features= X
    for i in range(2,degree+1):
        X = np.append(X, features ** i,axis=1)
    return X


path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
X = X_transform(X, degree)
############################################################################

# convert to numpy arrays and initialize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta, beta)
halt = eps  # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
while i < n_epoch and halt >=eps:
    db, dw = computeGrad(X, y, theta, beta)
    b = b - (alpha * db)
    w = w - (alpha * dw)
    theta = (b, w)
    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    L = computeCost(X, y, theta, beta)
    ############################################################################
    print(" {0} L = {1}".format(i, L))
    i += 1

# print parameter values found after the search
print("w = ", w)
print("b = ", b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test,
                        axis=1)  # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
X_feat = X_transform(X_feat, degree)
############################################################################
plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X) - kludge, np.amax(X) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")


############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################
plt.savefig("C:\\Users\\Owner\\Desktop\\hw2_release\\hw2_release\\code\\out\\Polynomial_b[3].png")
plt.show()

