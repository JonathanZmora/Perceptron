import pandas as pd
import numpy as np


class Perceptron:

    max_iter = 1000

    def __init__(self):
        self._weights = None
        self._train_score = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X['b'] = np.ones(X.shape[0])
        self._weights = np.zeros(X.shape[1])
        best_score = self.score(X, y)
        best_weights = self._weights.copy()
        X = X.to_numpy()
        for _ in range(self.max_iter):
            errors = False
            for i, x in enumerate(X):
                product = np.dot(x, self._weights)
                product_sign = 1 if product >= 0 else -1
                if not product_sign == y[i]:
                    self._weights += x * y[i]
                    errors = True
                    current_score = self.score(X, y)
                    if current_score > best_score:
                        best_score = current_score
                        best_weights = self._weights.copy()

            if not errors:
                break

        self._weights = best_weights
        self._train_score = best_score

    def predict(self, X):
        X['b'] = np.ones(X.shape[0])
        products = np.dot(X, self._weights)
        predictions = np.where(products >= 0, 1, -1)

        return predictions

    def score(self, X, y: pd.Series):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    @property
    def weights(self):
        return self._weights

    @property
    def train_score(self):
        return self._train_score






