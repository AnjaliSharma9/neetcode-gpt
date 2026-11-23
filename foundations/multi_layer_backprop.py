import numpy as np
from typing import List, Dict


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> Dict:
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y = np.array(y_true, dtype=float)

        # forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2
        n = len(y)
        loss = np.mean((z2 - y) ** 2)

        # backward pass
        dz2 = 2 * (z2 - y) / n
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        da1 = W2.T @ dz2
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        def r(a):
            return np.round(a, 4).tolist()

        return {
            'loss': round(float(loss), 4),
            'dW1': r(dW1),
            'db1': r(db1),
            'dW2': r(dW2),
            'db2': r(db2),
        }