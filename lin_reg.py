import numpy as np
import pandas as pd
# %pylab inline

class LinearRegression:
    def __init__(self, lr=0.1, max_iter=1000000):
        self.k = np.random.random(2)
        self.b = np.random.random()
        self.lr = lr # learning rate

    def predict(self, x):
        return x * self.k + self.b

    def fit(self, data, answers):
        N = data.shape[0]
        step = 1
        it = 0

        while step > 1e-4:
            delta = self.k * data + self.b - answers
            mse = 1 / N * delta.T.dot(delta)
            dk = 2 / N * data.dot(delta.T)
            db = 2 / N * np.sum(delta.T)

            self.k = self.k - self.lr * dk[0, 0] # coz matrix
            self.b = self.b - self.lr * db

            # Count step to prevent overfitting
            step = np.sqrt(dk * dk + db * db) * self.lr
            it += 1

lr = LinearRegression()

plot([lr.predict(i) for i in range(-10, 10)])

X = y = np.arange(-10, 10).reshape((-1, 1))
y = (np.arange(-10, 10) + np.random.normal(scale=5, size=20)).reshape((-1, 1))