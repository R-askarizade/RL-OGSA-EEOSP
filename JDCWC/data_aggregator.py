import math
import numpy as np


class USEKLMSAggregator:
    def __init__(self, eta=0.1, alpha=1.0, c=0.5, sigma=1.0):
        self.eta = eta
        self.alpha = alpha
        self.c = c
        self.sigma = sigma
        self.dictionary = []  # list of x vectors
        self.weights = []     # coefficient vector Ω

    def sigmoid_kernel(self, x1, x2):
        return math.tanh(self.alpha * np.dot(x1, x2) + self.c)

    def exp_kernel(self, x1, x2):
        return math.exp(-np.linalg.norm(np.array(x1) - np.array(x2)) ** 2 / (2 * self.sigma ** 2))

    def use_kernel(self, x1, x2):
        # Eq. (16)
        return (self.sigmoid_kernel(x1, x2) + self.exp_kernel(x1, x2)) / 2.0

    def predict(self, x):
        if not self.dictionary:
            return 0.0
        k_vals = [self.use_kernel(x, x_dict) for x_dict in self.dictionary]
        return sum(w * k for w, k in zip(self.weights, k_vals))

    def train(self, x, y):
        y_pred = self.predict(x)
        e_n = y - y_pred  # Eq. (20)
        # Update weights: Ω_n = Ω_{n-1} + η e_n Φ(x_n)
        # In dual form: weights updated via kernel expansion
        self.dictionary.append(x)
        new_weights = []
        for i in range(len(self.weights)):
            # Implicitly: w_i += η e_n k(x, x_i)
            # But paper uses weight vector update, we store dual form
            new_weights.append(self.weights[i])
        new_weights.append(self.eta * e_n)
        self.weights = new_weights

    def aggregate(self, data_list):
        if not data_list:
            return 0.0
        x_dummy = np.array([1.0])  # 1D input
        for d in data_list:
            self.train(x_dummy, d)
        return self.predict(x_dummy)
