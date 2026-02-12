import numpy as np
import matplotlib.pyplot as plt
from likelihood import bernoulli_log_likelihood, normal_log_likelihood

def plot_bernoulli_ll(data):
    p_values = np.linspace(0.01, 0.99, 100)
    ll = [bernoulli_log_likelihood(p, data) for p in p_values]
    plt.plot(p_values, ll)
    plt.xlabel("p")
    plt.ylabel("Log Likelihood")
    plt.show()

def plot_normal_ll_mu(data, sigma_mle):
    mu_mle = np.mean(data)
    mu_vals = np.linspace(mu_mle-3, mu_mle+3, 100)
    ll = [normal_log_likelihood(mu, sigma_mle, data) for mu in mu_vals]
    plt.plot(mu_vals, ll)
    plt.show()