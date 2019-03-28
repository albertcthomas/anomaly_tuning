# Approximation of the scoring function of the OneClassSVM with a
# constant piecewise function obtained with a RegressionTree
# Computation of volumes of level sets of the scoring function

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.stats import kendalltau
from scipy.stats.mstats import mquantiles

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from anomaly_tuning.utils import GaussianMixture
from anomaly_tuning.estimators import OCSVM

# Matplotlib configuration
font = {'weight': 'normal',
        'size': 15}
matplotlib.rc('font', **font)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'


class RegressionTree(DecisionTreeRegressor):
    """Subclass of DecisionTreeRegressor estimator adding two more methods:
    Retrieving leafs with values above a given offset and getting
    hyperrectangles corresponding to leafs
    """

    def leafs_above_offfset(self, X, offset):
        """Retrieve leafs of the tree with values greater than offset. Returns
        the indexes of such leaves"""
        tree = self.tree_

        leaf_id = self.apply(X)  # leaf id of each sample
        leaf_id_sorted = np.unique(leaf_id)  # leaf indexes

        node_values = tree.value.reshape(tree.value.shape[0])
        leaf_values = node_values[leaf_id_sorted]
        # Tree leafs with values above offset
        leaf_id_sorted_offset = leaf_id_sorted[leaf_values >= offset]

        return leaf_id_sorted_offset

    def leafs_hyperrectangle(self, X, leaf_indexes):
        """Each leaf of a tree defines an hyperrectangle. Returns the
        hyperrectangle of the given leafs.
        """

        tree = self.tree_
        leaf_id = self.apply(X)  # leaf id of each sample

        n_samples, n_features = X.shape
        X_range = np.zeros((n_features, 2))
        X_range[:, 0] = np.min(X, axis=0)
        X_range[:, 1] = np.max(X, axis=0)

        n_leaf_indexes = len(leaf_indexes)
        leaf_range = np.zeros((n_leaf_indexes, n_features, 2))
        for i in range(n_leaf_indexes):
            leaf_range[i, :] = X_range

        # Get the samples that belong to the leafs
        ind_samples = np.arange(n_samples)
        samples_idx = ind_samples[np.where(np.in1d(leaf_id,
                                                   leaf_indexes))]

        node_indicator = self.decision_path(X)
        feature = tree.feature
        threshold = tree.threshold

        for i in samples_idx:
            node_index = node_indicator.indices[node_indicator.indptr[i]:
                                                node_indicator.indptr[i + 1]]
            for node_id in node_index:
                if leaf_id[i] == node_id:
                    continue

                range_id = np.where(leaf_indexes == leaf_id[i])[0][0]

                feature_i = feature[node_id]
                threshold_i = threshold[node_id]
                if (X[i, feature_i] <= threshold_i):
                    leaf_range[range_id, feature_i, 1] = threshold_i
                else:
                    leaf_range[range_id, feature_i, 0] = threshold_i

        return leaf_range




    vol_tot_leaf_n = vol_tot_leaf / np.sum(vol_tot_leaf)

    # Computing volume with Monte Carlo
    rng = np.random.RandomState(random_state)
    # number of samples to draw / leaf
    n_sim_leaf = (vol_tot_leaf_n * n_sim).astype(int) + 1
    U = np.zeros((np.sum(n_sim_leaf), n_features))

    vol = 0.
    for i in range(n_leaf_set):
        leaf_i_range = leaf_range[i]
        vol_leaf = vol_tot_leaf[i]
        n_sim_l = n_sim_leaf[i]
        left_idx = np.sum(n_sim_leaf[:i])
        right_idx = left_idx + n_sim_l
        leaf_slice = slice(left_idx, right_idx)
        for l in range(n_features):
            U[leaf_slice, l] = rng.uniform(leaf_i_range[l, 0],
                                           leaf_i_range[l, 1],
                                           n_sim_l)
        leaf_U = clf.score_samples(U[leaf_slice, :])
        U_inliers_leaf = (leaf_U - offset >= 0)

        vol += vol_leaf * np.mean(U_inliers_leaf)

    return vol_tot_leaf_set, vol, U


if __name__ == "__main__":
    # Gaussian mixture
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

    gmm = GaussianMixture(weights, means, covars,
                          random_state=2)

    X = gmm.sample(n_samples=n_samples)

    def density_gm(x):
        return (weight_1 * multivariate_normal.pdf(x, mean_1, cov_1) +
                weight_2 * multivariate_normal.pdf(x, mean_2, cov_2))

    # Estimation of the density level set corresponding to the MV set
    n_quantile = 1000000
    Xq = gmm.sample(n_samples=n_quantile)
    density_q = density_gm(Xq)
    tau = mquantiles(density_q, 1 - 0.95)

    X_train, X_test = train_test_split(X, test_size=0.2)

    sigmas = np.linspace(0.01, 20, 20)
    # sigmas = np.array([1.3])
    vol_sigmas = np.zeros(len(sigmas))
    vol_hyp_array = np.zeros(len(sigmas))

    for s, sigma in enumerate(sigmas):
        clf = OCSVM(sigma=sigma)
        clf = clf.fit(X_train)

        Z_data = clf.score_samples(X)
        min_data = np.min(Z_data)
        max_data = np.max(Z_data)

        Z_test = clf.score_samples(X_test)

        X_range = np.zeros((n_features, 2))
        X_range[:, 0] = np.min(X, axis=0)
        X_range[:, 1] = np.max(X, axis=0)

        b_0 = mquantiles(Z_test, 1 - 0.95)[0]
        b_0 = b_0 - 1e-12  # for robustness of the code

        # Printing proba of the estimated MV sets
        print(np.mean(Z_data >= b_0))

        X_outliers = X[Z_data < b_0, :]

        # Decision Tree regression on scoring function
        # def ranking_loss(truth, pred):
        #     return kendalltau(truth, pred)[0]

        # rank_score = make_scorer(ranking_loss, greater_is_better=True)

        # max_depths = {'max_depth': np.arange(10, 11, 1)}
        # reg = GridSearchCV(RegressionTree(), max_depths, cv=5,
        #                    scoring=rank_score, n_jobs=-1).fit(X, Z_data)
        # print(reg.best_params_)
        # reg = reg.best_estimator_
        reg = RegressionTree()
        reg = reg.fit(X, Z_data)

        Z_test_tree = reg.predict(X_test)
        Z_data_tree = reg.predict(X)
        b_1 = mquantiles(Z_test_tree, 1 - 0.95)[0]
        b_1 = b_1 - 1e-12  # for robustness of the code

        # Volume estimated with regression tree
        n_sim = 10000
        vol_leafs, vol, U = volume_from_reg_tree(reg, b_0, X, clf, n_sim=n_sim)
        print(np.mean(Z_data_tree >= b_0))

        # Volume estimated with uniform sampling in the enclosing hypercube
        rng = np.random.RandomState(42)
        vol_tot_cube = np.prod(X_range[:, 1] - X_range[:, 0])
        U_hyp = np.zeros((n_sim, n_features))
        for l in range(n_features):
            U_hyp[:, l] = rng.uniform(X_range[l, 0], X_range[l, 1], n_sim)

        clf_U = clf.score_samples(U_hyp)
        U_inliers = (clf_U - b_0 >= 0)
        vol_hyp = vol_tot_cube * np.mean(U_inliers)

        print('Volume hypercube:', vol_hyp)
        print('Volume tree:', vol)
        print('Volume leafs', vol_leafs)

        vol_sigmas[s] = vol_leafs
        vol_hyp_array[s] = vol_hyp

    plt.figure()
    plt.plot(sigmas, np.log(vol_sigmas), label='Volume leafs > threshold')
    plt.plot(sigmas, np.log(vol_hyp_array),
             label='Volume from uniform sampling')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('log(Volume)')
    plt.legend(loc=0)
    plt.show()

    if n_features == 2:  # Plotting the results

        h = 0.1  # step size of the mesh
        x_min, x_max = X_range[0, 0] - 0.5, X_range[0, 1] + 0.5
        y_min, y_max = X_range[1, 0] - 0.5, X_range[1, 1] + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.score_samples(grid)
        Z_reg = reg.predict(grid)
        Z_reg = Z_reg.reshape(xx.shape)

        Z = Z.reshape(xx.shape)
        Z_true = density_gm(grid)
        Z_true = Z_true.reshape(xx.shape)

        plt.figure()
        plt.subplot(1, 2, 1)
        c_0 = plt.contour(xx, yy, Z, levels=[b_0], linewidths=2,
                          colors='green')
        plt.clabel(c_0, inline=1, fontsize=15, fmt={b_0: '0.95'})
        plt.contour(xx, yy, Z_true, levels=tau, linewidths=2, colors='red')
        plt.scatter(U_hyp[:, 0], U_hyp[:, 1], s=1., color='black')
        plt.title('Estimated volume: %.2f' % (vol_hyp))
        plt.axis('tight')
        plt.subplot(1, 2, 2)
        plt.contour(xx, yy, Z_reg, levels=[b_0], linewidths=2, colors='blue')
        c_0 = plt.contour(xx, yy, Z, levels=[b_0], linewidths=2,
                          colors='green')
        plt.clabel(c_0, inline=1, fontsize=15, fmt={b_0: '0.95'})
        plt.contour(xx, yy, Z_true, levels=tau, linewidths=2, colors='red')
        plt.scatter(U[:, 0], U[:, 1], s=1., color='black')
        plt.title('Estimated volume: %.2f' % (vol))
        plt.axis('tight')
        plt.show()
