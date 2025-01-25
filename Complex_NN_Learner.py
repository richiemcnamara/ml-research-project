import matplotlib.pyplot as plt     #plotting
import numpy as np                  #NumPy          
import pandas as pd    
from tensorflow import keras        #various NN packages
from keras import layers
from keras.layers import Dense
from keras.callbacks import EarlyStopping

class Complex_NN_Learner:

    def __init__(self, input_shape, output_shape):

        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),              
                layers.Dense(10, activation='relu'),
                layers.Dense(5, activation='relu'),
                layers.Dense(output_shape),                   
            ]
        )

    def train(self, x, y):

        # Compile model
        self.model.compile(loss="mean_squared_error", optimizer="sgd")

        # Train model
        history = self.model.fit(x, y, batch_size=32, epochs=100)


    def test(self, x):
        predictions = self.model.predict(x)
        return predictions
    
    def __repr__(self):
        return "Complex"
