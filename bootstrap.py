import numpy as np
import pandas as pd
import scipy.stats as st


class Bootstrap:

    def __init__(self, model, kwargs, bags = 20):
        # set up your object
        self.models = []
        for i in range(bags):
            self.models.append(model(**kwargs))


    def train(self, x, y):
        n = x.shape[0]

        model_num = 1

        for model in self.models:
            sampled_rows_indexes = np.random.choice(n, n, replace=True)

            sampled_x = x[sampled_rows_indexes]
            sampled_y = y[sampled_rows_indexes]

            print(f"Training Model: {model_num}")
            model_num += 1

            model.train(sampled_x, sampled_y)


    def test(self, x, confidence=95):
        predictions = np.empty(shape=(x.shape[0], len(self.models)))        

        for i in range(len(self.models)):
            model = self.models[i]
            predictions[:,i] = model.test(x).reshape((x.shape[0],))

        alpha = 1 - confidence/100
        z = st.norm.ppf(1 - alpha/2)

        point_estimates = predictions.mean(axis=1)
        lowers = point_estimates - (z * predictions.std(axis=1))
        uppers = point_estimates + (z * predictions.std(axis=1))

        return lowers, uppers, point_estimates
        