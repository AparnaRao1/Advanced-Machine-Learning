from data_generation import generate_bernoulli, generate_normal
from mle_estimation import mle_bernoulli, mle_normal
from Visualization import plot_bernoulli_ll, plot_normal_ll_mu

# Bernoulli example
data_b = generate_bernoulli()
p_mle = mle_bernoulli(data_b)
print("Bernoulli MLE:", p_mle)
plot_bernoulli_ll(data_b)

# Normal example
data_n = generate_normal()
mu_mle, sigma_mle = mle_normal(data_n)
print("Normal MLE:", mu_mle, sigma_mle)
plot_normal_ll_mu(data_n, sigma_mle)