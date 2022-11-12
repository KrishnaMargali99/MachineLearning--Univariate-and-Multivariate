import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Name: Goparapu Krishna Margali, kg4060@rit.edu
CSCI 635: Introduction to Machine Learning
Problem 1: Univariate Regression

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

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.01  # step size coefficient
eps = 0.00000  # controls convergence criterion
n_epoch = 10000  # number of epochs (full passes through the dataset)


# begin simulation

def regress(X, theta):
    ############################################################################
    # WRITEME: write your code here to complete the routine

    return theta[0] + (theta[1] * X)
    ############################################################################


def gaussian_log_likelihood(mu, y):
    ############################################################################
    # WRITEME: write your code here to complete the sub-routine
    loss = np.sum((mu - y) ** 2)
    return loss
    ############################################################################


def computeCost(X, y, theta):  # loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
    # WRITEME: write your code here to complete the routine
    m = 1 / 2 * (len(y))
    cost = m * gaussian_log_likelihood(regress(X, theta), y)
    return cost
    ############################################################################


def computeGrad(X, y, theta):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    b_temp = (np.subtract(regress(X, theta), np.asarray(y)))
    db = theta[0] - (alpha / len(X)) * (np.subtract(regress(X, theta), np.asarray(y))).sum()
    dw = theta[1] - (alpha / len(X)) * np.matmul(np.transpose(b_temp), X).sum()

    return db, dw


############################################################################

path = os.getcwd() + '/data/prob1.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# display some information about the dataset itself here
############################################################################
# WRITEME: write your code here to print out information/statistics about the
#          data-set "data" using Pandas (consult the Pandas documentation to learn how)
# WRITEME: write your code here to create a simple scatterplot of the dataset
#          itself and print/save to disk the result
############################################################################

stats_data = pd.DataFrame(data)
print(stats_data.describe())

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

plt.scatter(X, y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.savefig("C:\\Users\\Owner\\Desktop\\hw2_release\\hw2_release\\code\\out\\scatter_plot.png")
plt.show()
# convert to numpy arrays and initialize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
halt = 0  # halting variable (you can use these to terminate the loop if you have converged)
i = 0
cost = []  # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
while (i < n_epoch and halt == 0):
    updated_b, updated_w = computeGrad(X, y, theta)
    b = updated_b
    w = updated_w
    ############################################################################
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent &
    #          record anything else desired for later
    ############################################################################

    # (note: don't forget to override the theta variable...)
    theta = (b, w)
    L = computeCost(X, y, theta)  # track our loss after performing a single step

    print(" {0} L = {1}".format(i, L))
    cost=cost+[L]
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

kludge = 0.25  # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
plt.savefig("C:\\Users\\Owner\\Desktop\\hw2_release\\hw2_release\\code\\out\\univariate_regression_plot.png")
############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################

############################################################################
# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch
############################################################################

plt.show()  # convenience command to force plots to pop up on desktop

x_axis=range(len(cost))
plt.plot(x_axis, cost, 'r--')
plt.legend(['Training loss', 'Test Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("C:\\Users\\Owner\\Desktop\\hw2_release\\hw2_release\\code\\out\\loss_plot.png")
plt.show()

