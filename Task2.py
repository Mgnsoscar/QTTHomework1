import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def poisson_probability(k, dt, tau):
    """
    Calculate the probability of observing k events in time window dt
    with a rate parameter tau.
    """
    lambda_ = dt / tau
    return np.exp(-lambda_) * (lambda_**k) / factorial(k)

def plot_poisson_distribution(dt, tau, max_k):
    """
    Plot the Poisson distribution for observing up to max_k events
    in time window dt with a rate parameter tau.
    """
    k_values = np.arange(max_k + 1)
    probabilities = [poisson_probability(k, dt, tau) for k in k_values]

    plt.bar(k_values, probabilities, color='skyblue')
    plt.title('Poisson Distribution for Time Window dt={}'.format(dt))
    plt.xlabel('Number of Events (N)')
    plt.ylabel('Probability')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

# Example usage
dt = 5  # Time window
tau = 5  # Rate parameter
max_k = 10  # Maximum number of events to plot
plot_poisson_distribution(dt, tau, max_k)
