import numpy as np

def bernoulli_log_likelihood(p, data):
    return np.sum(data*np.log(p) + (1-data)*np.log(1-p))

def normal_log_likelihood(mu, sigma, data):
    n = len(data)
    return -n*np.log(sigma*np.sqrt(2*np.pi)) - np.sum((data - mu)**2)/(2*sigma**2)