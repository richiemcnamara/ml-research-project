from tensorflow import keras        #various NN packages
from keras.layers import Dense


# Boston Housing
class BH_Med:

    def __init__(self, input_shape, output_shape):

        self.model = keras.Sequential()
            
        self.model.add(Dense(16, input_shape=(12, ), activation='relu', name='dense_1'))
        self.model.add(Dense(8, activation='relu', name='dense_2'))
        self.model.add(Dense(1, activation='linear', name='dense_output'))             
            
        

    def train(self, x, y):

        # Compile model
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        history = self.model.fit(x, y, batch_size=32, epochs=100)


    def test(self, x):
        predictions = self.model.predict(x)
        return predictions
    
    def __repr__(self):
        return 'Med'