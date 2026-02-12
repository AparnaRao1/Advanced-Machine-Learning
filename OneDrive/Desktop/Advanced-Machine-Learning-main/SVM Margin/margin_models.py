import numpy as np

def compute_margin(w, b, X, y):
    return y * (np.dot(X, w) + b)

def hard_margin_loss(margin):
    return np.sum(np.maximum(0, 1 - margin))

def soft_margin_loss(margin, C=1.0):
    return C * np.sum(np.maximum(0, 1 - margin))