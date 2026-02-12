import numpy as np
from likelihood import bernoulli_log_likelihood, normal_log_likelihood

def mle_bernoulli(data):
    return data.mean()

def mle_normal(data):
    mu_mle = np.mean(data)
    sigma_mle = np.sqrt(np.mean((data - mu_mle)**2))
    return mu_mle, sigma_mle