import numpy as np
class LinearRegression:
    def __init__(self):
        self.n_epochs = 1000
        self.weights = None
        self.bias = None
        self.lr = 0.001
    def fit(self,x,y):
        (no_inputs, no_x) = x.shape
        self.weights = np.random.uniform(0,10,(no_x, 1))
        self.bias = float(np.random.randint(0,10))
        for i in range(self.n_epochs):
            y_pred = x @ self.weights + self.bias
            dm = (-2/no_inputs) * (x.T @ (y-y_pred))
            dc = (-2/no_inputs) * (y-y_pred).sum()
            self.weights = self.weights - self.lr*dm
            self.bias = self.bias - self.lr*dc
    def predict(self, x):
        y_pred = x @ self.weights + self.bias
        return y_pred