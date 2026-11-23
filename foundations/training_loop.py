import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        for _ in range(epochs):
            y_hat = X @ w + b
            err = y_hat - y
            dw = (2 / n) * (X.T @ err)
            db = (2 / n) * np.sum(err)
            w = w - lr * dw
            b = b - lr * db

        return np.round(w, 5), round(float(b), 5)