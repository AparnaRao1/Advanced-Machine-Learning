import numpy as np

def margin(w, b, X, y):
    return y * (X @ w + b)

def hinge(m):
    return np.maximum(0, 1 - m)

def total_hinge(w, b, X, y):
    return np.mean(hinge(margin(w,b,X,y)))