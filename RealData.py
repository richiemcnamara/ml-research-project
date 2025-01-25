import numpy as np                  #NumPy
import matplotlib.pyplot as plt     #plotting
from numpy.linalg import inv, det, norm   #inverse of a matrix, norm of a vector
import pandas as pd                  #package for working with spreadsheet data
from sklearn.model_selection import train_test_split # for splitting data into training and validation sets
from sklearn.metrics import mean_squared_error       # for computing mse
import itertools                     # for making higher degree regression matrices
from io import BytesIO #for getting files from the web
import requests        #also for getting files from the web
from BH_Simple import BH_Simple
from BH_Med import BH_Med
from BH_Complex import BH_Complex
from bootstrap import Bootstrap
import assess_pi


def standardize(X):
    return X/(X.max(axis = 0))

def column_products(X, d):
    '''produce a matrix whose columns are all degree d products of the columns of X
    it will become a part of the regression metric for multivariate polynomial regression'''
    Z = [np.product(np.array(x),axis = 0) for x in itertools.combinations_with_replacement(X.T,d)]
    return np.array(Z).squeeze().T

r = requests.get('http://web.bowdoin.edu/~tpietrah/colab/housing/housing_data_X.npy', stream = True)
X = np.load(BytesIO(r.raw.read()))

r = requests.get('http://web.bowdoin.edu/~tpietrah/colab/housing/housing_data_y.npy', stream = True)
y = np.load(BytesIO(r.raw.read()))


def real_experiment(LEARNER, bags, savefig=False):

    X1, X2, y1, y2 = train_test_split(X, y, test_size = 0.3, random_state= 42)

    b = Bootstrap(LEARNER, {'input_shape': X1.shape[1], 'output_shape': y1.shape[0]}, bags=bags)

    b.train(X1, y1)

    conf = 95
    lowers, uppers, points = b.test(X1, confidence=conf)

    plt.figure(1, (15,6))
    plt.title(f"{bags} Resamples with {conf}% Confidence (Training Data)")
    plt.scatter(points, y1, label='Actual', color='blue')
    plt.scatter(points, points, label='Pred', color='red')
    plt.scatter(points, lowers, label="Lower Predictions", color = 'orange')
    plt.scatter(points, uppers, label="Upper Predictions", color = 'orange')
    plt.xlabel("Predictions")
    plt.legend()
    if savefig:
        plt.savefig(f"./BH_figures/BH_{LEARNER}_train.png", dpi=300)
    plt.show()

    avg_w, max_w, min_w, std_w = assess_pi.width(lowers, uppers)

    train_mse = mean_squared_error(y1, points)

    lowers, uppers, points = b.test(X2, confidence=conf)

    plt.figure(2, (15,6))
    plt.title(f"{bags} Resamples with {conf}% Confidence (Testing Data)")
    plt.scatter(points, y2, label='Actual', color='blue')
    plt.scatter(points, points, label='Pred', color='red')
    plt.scatter(points, lowers, label="Lower Predictions", color = 'orange')
    plt.scatter(points, uppers, label="Upper Predictions", color = 'orange')
    plt.xlabel("Predictions")
    plt.legend()
    if savefig:
        plt.savefig(f"./BH_figures/BH_{LEARNER}_test.png", dpi=300)
    plt.show()

    coverage = assess_pi.coverage(y2, lowers, uppers)
    test_avg_w, test_max_w, test_min_w, test_std_w = assess_pi.width(lowers, uppers)

    print("Training Widths (avg, max, min, std)", avg_w, max_w, min_w, std_w)
    print("Testing Widths (avg, max, min, std)", test_avg_w, test_max_w, test_min_w, test_std_w)
    print("Coverage %:", coverage)
    print("TRAIN Pred vs. Actual MSE:", train_mse)
    print("TEST Pred vs. Actual MSE:", mean_squared_error(y2, points))


## Run one at a time to get the stats

# Simple NN
real_experiment(BH_Simple, 1000, savefig=True)

# Med NN
real_experiment(BH_Med, 1000, savefig=True)

# Complex NN
real_experiment(BH_Complex, 1000, savefig=True)
