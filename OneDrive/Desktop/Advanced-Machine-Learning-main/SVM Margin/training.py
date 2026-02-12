import numpy as np
from losses import margin

def train_svm(X, y, lr=0.01, epochs=500):
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(epochs):
        m = margin(w, b, X, y)
        mask = m < 1

       
        if mask.any():
            dw = -np.mean(y[mask, None] * X[mask], axis=0)
            db = -np.mean(y[mask])
        else:
            dw = 0
            db = 0
            print("Margin satisfied — converged")
            break

        w -= lr * dw
        b -= lr * db

    return w, b