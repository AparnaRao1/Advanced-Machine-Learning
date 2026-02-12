import numpy as np

def generate_separable(n=100):
    np.random.seed(0)
    X1 = np.random.randn(n,2) + [2,2]
    X2 = np.random.randn(n,2) + [-2,-2]
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y

def generate_overlap(n=100):
    np.random.seed(1)
    X1 = np.random.randn(n,2)
    X2 = np.random.randn(n,2) + [1,1]
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n), -np.ones(n)])
    return X, y