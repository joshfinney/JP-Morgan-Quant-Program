import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Constants
N_BUCKETS = 10  # Number of buckets

# Load the dataset


def load_data(filename):
    """Load CSV data."""
    return pd.read_csv(filename)

# Plotting functions


def plot_fico_distribution(data, mse_centers=None, log_likelihood_boundaries=None):
    """Visualize the FICO scores along with the bucket boundaries."""
    plt.figure(figsize=(10, 5))
    sns.histplot(data['fico_score'], bins=30, kde=False, stat="probability")

    if mse_centers is not None:
        for i, center in enumerate(mse_centers):
            linestyle = '--'
            color = 'red'
            label = 'MSE centers' if i == 0 else ""
            plt.axvline(center, color=color, linestyle=linestyle, label=label)

    if log_likelihood_boundaries is not None:
        for i, boundary in enumerate(log_likelihood_boundaries):
            linestyle = '-.'
            color = 'blue'
            label = 'Log Likelihood boundaries' if i == 0 else ""
            plt.axvline(boundary, color=color,
                        linestyle=linestyle, label=label)

    plt.title('Distribution of FICO scores with bucket boundaries')
    plt.xlabel('FICO Score')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


# Mean squared error quantization using KMeans clustering
def quantize_mse(data, n_clusters):
    """Cluster data into buckets to minimize mean squared error."""
    kmeans = KMeans(n_clusters=n_clusters)
    data['label'] = kmeans.fit_predict(data[['fico_score']])
    data['cluster_center'] = data['label'].apply(
        lambda x: kmeans.cluster_centers_[x])
    return data, kmeans.cluster_centers_.flatten()

# Log-likelihood optimization
def log_likelihood(buckets, scores, defaults):
    """Compute the negative log likelihood of the given bucket boundaries."""
    total_log_likelihood = 0
    for i in range(len(buckets) - 1):
        indices = (scores >= buckets[i]) & (scores < buckets[i + 1])
        ni = np.sum(indices)
        ki = np.sum(defaults[indices])
        pi = ki / ni if ni != 0 else 0
        total_log_likelihood += ki * np.log(pi) + (ni - ki) * np.log(1 - pi)
    return -total_log_likelihood


def optimize_log_likelihood(data, n_buckets):
    """Determine optimal bucket boundaries by maximizing log likelihood."""
    scores = data['fico_score'].values
    defaults = data['default'].values
    initial_buckets = np.linspace(
        scores.min(), scores.max(), n_buckets + 1)[1:-1]
    result = minimize(log_likelihood, initial_buckets, args=(scores, defaults))
    boundaries = [scores.min()] + list(result.x) + [scores.max()]
    return sorted(boundaries)


# Main
if __name__ == "__main__":
    data = load_data("Task 3 and 4_Loan_Data.csv")

    # Debugging: First few rows of the data
    print(data.head())

    # Using MSE and log-likelihood approaches for quantization
    data_mse, mse_centers = quantize_mse(data, N_BUCKETS)
    log_likelihood_boundaries = optimize_log_likelihood(data, N_BUCKETS)

    # Debugging: Print the results
    print("\nCenters using MSE:", sorted(mse_centers))
    print("Boundaries using Log Likelihood:", log_likelihood_boundaries)

    # Visualize the FICO score distribution and the bucket boundaries
    plot_fico_distribution(data, mse_centers, log_likelihood_boundaries)