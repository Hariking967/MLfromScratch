import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self):
        self.n_epochs = 1200
        self.weights = None
        self.bias = None
        self.lr = 0.01
    def fit(self,x,y):
        (no_inputs, no_x) = x.shape
        self.weights = np.random.uniform(10,100,(no_x, 1))
        self.bias = float(np.random.randint(0,100))
        for i in range(self.n_epochs):
            y_lin_pred = x @ self.weights + self.bias
            y_pred = sigmoid(y_lin_pred)
            dm = (1/no_inputs) * (x.T @ (y_pred-y))
            dc = (1/no_inputs) * (y_pred-y).sum()
            self.weights = self.weights - self.lr*dm
            self.bias = self.bias - self.lr*dc
    def predict(self, x):
        y_lin_pred = x @ self.weights + self.bias
        y_pred = sigmoid(y_lin_pred)
        return (y_pred > 0.5).astype(int).flatten()