import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simple_NN_Learner import Simple_NN_Learner
from Med_NN_Learner import Med_NN_Learner
from Complex_NN_Learner import Complex_NN_Learner
from bootstrap import Bootstrap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import assess_pi

def synthetic_experiment(LEARNER, bags, linear='Linear', savefig=False):
    
    # Synthetic Data with Known Error Function
    n = 200   #number of points in our data set
    if linear == 'Linear':
        error = 1
    else:
        error = 0.3

    if linear == 'Linear':

        x1 = np.sort(np.random.uniform(0,3,(n, 1)), axis=0) #choose n random values of x, sort then so they are in increasing order.
        y1 = 5*x1       
        noise = np.random.uniform(-error, error, size=(n, 1))        
        b1 =  y1 + noise                       

        plt.figure(1, figsize = (15,6)) 
        plt.title(f"{n} Synthetic Data Points (Uniform Error between -{error} and {error})")
        plt.plot(x1, y1, label="Known Function")            
        plt.plot(x1,b1,'.',markersize=15,label='noisy data')
        plt.legend()
        if savefig:
            plt.savefig(f"./uniform_figures/{linear}_data_{LEARNER}.png", dpi=300)
        plt.show()

    else:
        # Put Non-Linear Function here
        x1 = np.sort(np.random.uniform(0,3,(n, 1)), axis=0) #choose n random values of x, sort then so they are in increasing order.
        y1 = np.square(x1)       
        noise = np.random.uniform(-error, error, size=(n, 1))        
        b1 =  y1 + noise                       

        plt.figure(1, figsize = (15,6)) 
        plt.title(f"{n} Synthetic Data Points (Uniform Error between -{error} and {error})")
        plt.plot(x1, y1, label="Known Function")            
        plt.plot(x1,b1,'.',markersize=15,label='noisy data')
        plt.legend()
        if savefig:
            plt.savefig(f"./uniform_figures/{linear}_data_{LEARNER}.png", dpi=300)
        plt.show()

    # Training / Testing Data
    X_train, X_test, Y_train, Y_test = train_test_split(x1, b1, test_size=0.4, shuffle=True)

    ## Bootstrap Prediction Intervals

    b = Bootstrap(LEARNER, {'input_shape': 1, 'output_shape': 1}, bags=bags)

    b.train(X_train, Y_train)

    # Training Data
    conf = 95
    lowers, uppers, points = b.test(X_train, confidence=conf)

    if linear == 'Linear':

        plt.figure(2, (15,6))
        plt.title(f"{bags} Resamples with {conf}% Confidence (Training Data)")
        plt.scatter(X_train, Y_train, label='Data')
        plt.scatter(X_train, 5*X_train - error, label="Known Error Range", color='gray', alpha=0.5)
        plt.scatter(X_train, 5*X_train + error, color='gray', alpha=0.5)
        plt.scatter(X_train, points, label='Point Predictions', color='red')
        plt.scatter(X_train, lowers, label="Lower Predictions", color = 'orange')
        plt.scatter(X_train, uppers, label='Upper Predictions', color = 'orange')
        plt.legend()
        if savefig:
            plt.savefig(f"./uniform_figures/{linear}_train_{LEARNER}.png", dpi=300)
        plt.show()

    else:
        # Non-Linear Plot
        plt.figure(2, (15,6))
        plt.title(f"{bags} Resamples with {conf}% Confidence (Training Data)")
        plt.scatter(X_train, Y_train, label='Data')
        plt.scatter(X_train, np.square(X_train) - error, label="Known Error Range", color='gray', alpha=0.5)
        plt.scatter(X_train, np.square(X_train) + error, color='gray', alpha=0.5)
        plt.scatter(X_train, points, label='Point Predictions', color='red')
        plt.scatter(X_train, lowers, label="Lower Predictions", color = 'orange')
        plt.scatter(X_train, uppers, label='Upper Predictions', color = 'orange')
        plt.legend()
        if savefig:
            plt.savefig(f"./uniform_figures/{linear}_train_{LEARNER}.png", dpi=300)
        plt.show()

    avg_w, max_w, min_w, std_w = assess_pi.width(lowers, uppers)

    train_mse = mean_squared_error(Y_train, points)

    
    # Testing Data
    lowers, uppers, points = b.test(X_test, confidence=95)

    if linear == 'Linear':

        plt.figure(3, (15,6))
        plt.title(f"{bags} Resamples with {conf}% Confidence (Test Data)")
        plt.scatter(X_test, Y_test, label='Data')
        plt.scatter(X_test, 5*X_test - error, label="Known Error Range", color='gray', alpha=0.5)
        plt.scatter(X_test, 5*X_test + error, color='gray', alpha=0.5)
        plt.scatter(X_test, points, label='Point Predictions', color='red')
        plt.scatter(X_test, lowers, label="Lower Predictions", color = 'orange')
        plt.scatter(X_test, uppers, label='Upper Predictions', color = 'orange')
        plt.legend()
        if savefig:
            plt.savefig(f"./uniform_figures/{linear}_test_{LEARNER}.png", dpi=300)
        plt.show()

    else:
        # Nonlinear plot
        plt.figure(3, (15,6))
        plt.title(f"{bags} Resamples with {conf}% Confidence (Test Data)")
        plt.scatter(X_test, Y_test, label='Data')
        plt.scatter(X_test, np.square(X_test) - error, label="Known Error Range", color='gray', alpha=0.5)
        plt.scatter(X_test, np.square(X_test) + error, color='gray', alpha=0.5)
        plt.scatter(X_test, points, label='Point Predictions', color='red')
        plt.scatter(X_test, lowers, label="Lower Predictions", color = 'orange')
        plt.scatter(X_test, uppers, label='Upper Predictions', color = 'orange')
        plt.legend()
        if savefig:
            plt.savefig(f"./uniform_figures/{linear}_test_{LEARNER}.png", dpi=300)
        plt.show()

    coverage = assess_pi.coverage(Y_test, lowers, uppers)
    test_avg_w, test_max_w, test_min_w, test_std_w = assess_pi.width(lowers, uppers)

    print("Training Widths (avg, max, min, std)", avg_w, max_w, min_w, std_w)
    print("Testing Widths (avg, max, min, std)", test_avg_w, test_max_w, test_min_w, test_std_w)
    print("Coverage %:", coverage)
    print("TRAIN Pred vs. Actual MSE:", train_mse)
    print("TEST Pred vs. Actual MSE:", mean_squared_error(Y_test, points))


## Run one at a time to get the stats

# Linear:
#Simple
synthetic_experiment(Simple_NN_Learner, 1000, 'Linear', savefig=True)

# Medium
synthetic_experiment(Med_NN_Learner, 1000, 'Linear', savefig=True)

# Complex
synthetic_experiment(Complex_NN_Learner, 1000, 'Linear', savefig=True)

# Non-linear:
#Simple
synthetic_experiment(Simple_NN_Learner, 1000, 'NonLinear', savefig=True)

# Medium
synthetic_experiment(Med_NN_Learner, 1000, 'NonLinear', savefig=True)

# Complex
synthetic_experiment(Complex_NN_Learner, 1000, 'NonLinear', savefig=True)
