# Author: Albert Thomas
# Example of the estimators tuned on a Gaussian Mixture

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

from anomaly_tuning.estimators import (AverageKLPE, MaxKLPE, OCSVM,
                                       KernelSmoothing, IsolationForest)
from anomaly_tuning import anomaly_tuning
from anomaly_tuning.utils import GaussianMixture
from anomaly_tuning.utils import compute_ensemble_score_samples

# Matplotlib configuration
font = {'weight': 'normal',
        'size': 15}
matplotlib.rc('font', **font)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Generating Gaussian mixture sample

N_JOBS = 1  # set to -1 to use all your CPUs

n_samples = 1000
n_features = 2
weight_1 = 0.5
weight_2 = 0.5
mean_1 = np.ones(n_features) * 2.5
mean_2 = np.ones(n_features) * 7.5
cov_1 = np.identity(n_features)
cov_2 = np.identity(n_features)
weights = np.array([weight_1, weight_2])
means = np.array([mean_1, mean_2])
covars = np.array([cov_1, cov_2])

gm = GaussianMixture(weights, means, covars, random_state=42)
X = gm.sample(n_samples)

alpha_set = 0.95
# Estimation of the density level set corresponding to the true MV set
n_quantile = 1000000
Xq = gm.sample(n_quantile)
density_q = gm.density(Xq)
tau = np.percentile(density_q, 100 * (1 - alpha_set))

# Plot grid
X_range = np.zeros((n_features, 2))
X_range[:, 0] = np.min(X, axis=0)
X_range[:, 1] = np.max(X, axis=0)

h = 0.1  # step size of the mesh
x_min, x_max = X_range[0, 0] - 0.5, X_range[0, 1] + 0.5
y_min, y_max = X_range[1, 0] - 0.5, X_range[1, 1] + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
Z_true = gm.density(grid)
Z_true = Z_true.reshape(xx.shape)


algorithms = [AverageKLPE, MaxKLPE, OCSVM, IsolationForest, KernelSmoothing]
algo_param = {
    'AverageKLPE': {'k': np.arange(1, min(50, int(0.8 * n_samples)), 2),
                    'novelty': [True]},
    'MaxKLPE': {'k': np.arange(1, min(50, int(0.8 * n_samples)), 2),
                'novelty': [True]},
    'OCSVM': {'sigma': np.linspace(0.01, 5., 10)},
    'IsolationForest': {'max_samples': np.linspace(0.1, 1., 10)},
    'KernelSmoothing': {'bandwidth': np.linspace(0.01, 5., 10)},
}

# Tuning step
n_estimators = 10
cv = ShuffleSplit(n_splits=n_estimators, test_size=0.2, random_state=42)

for algo in algorithms:

    name_algo = algo.__name__
    print('--------------', name_algo, ' -------------')
    parameters = algo_param[name_algo]

    models, _ = anomaly_tuning(X, base_estimator=algo,
                               parameters=parameters,
                               random_state=42,
                               cv=cv, n_jobs=N_JOBS)

    Z = compute_ensemble_score_samples(models, grid, n_jobs=N_JOBS)
    Z_data = compute_ensemble_score_samples(models, X, n_jobs=N_JOBS)

    off_data = np.percentile(Z_data, 100 * (1 - alpha_set))

    X_outliers = X[Z_data - off_data < 0, :]

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.title(name_algo)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    c_0 = plt.contour(xx, yy, Z, levels=[off_data], linewidths=2,
                      colors='green')
    plt.clabel(c_0, inline=1, fontsize=15, fmt={off_data: str(alpha_set)})
    plt.contour(xx, yy, Z_true, levels=[tau], linewidths=2, colors='red')
    plt.axis('tight')
    plt.scatter(X[:, 0], X[:, 1], s=4., color='black')
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], s=4., color='red')

plt.show()
