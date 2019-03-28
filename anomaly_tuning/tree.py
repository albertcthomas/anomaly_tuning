# Author: Albert Thomas
# License: BSD (3-clause)

# Extending scikit-learn regression tree to compute volumes of level sets.

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted


class RegressionTree(DecisionTreeRegressor):
    """Adding volume computation to scikit-learn DecisionTreeRegressor."""

    def volume_leafs(self, X_range, offset):
        """Compute volume of leafs with values greater than offset.

        Parameters
        ----------
        X_range : array, shape (n_features, 2)
            Range of the training data. X_range[j] contains the min and max
            of the training data along feature j.

        offset : float
            Threshold value.

        Returns
        -------
        volume : float
            Volume of the leafs with values greater than offset.
        """

        check_is_fitted(self, 'tree_')

        tree = self.tree_
        n_nodes = tree.node_count
        node_values = tree.value.reshape(n_nodes)
        children_left = tree.children_left
        children_right = tree.children_right
        thresholds = tree.threshold
        features = tree.feature

        is_leaf = (children_left == children_right)

        n_features = X_range.shape[0]
        node_ranges = np.zeros((n_nodes, n_features, 2))

        stack = [0]  # 0 is the root node id
        node_ranges[0] = X_range
        while len(stack) > 0:
            node_id = stack.pop()
            children_left_id = children_left[node_id]
            children_right_id = children_right[node_id]
            node_feature = features[node_id]
            node_treshold = thresholds[node_id]

            # as node_id is in the stack its range X_range[node_id]
            # has already been set either initially (root node)
            # previously in the while loop
            node_range = node_ranges[node_id]

            # range of children is inferred from parent range (node_range) and
            # information on split
            range_children_left = node_range.copy()
            range_children_left[node_feature, 1] = node_treshold
            range_children_right = node_range.copy()
            range_children_right[node_feature, 0] = node_treshold

            node_ranges[children_right_id] = range_children_right
            node_ranges[children_left_id] = range_children_left

            if not is_leaf[children_left_id]:
                stack.append(children_left_id)
            if not is_leaf[children_right_id]:
                stack.append(children_right_id)

        # getting leafs with values greater than offset and their ranges
        leaf_greater_offset = np.logical_and(is_leaf, node_values >= offset)
        leaf_ranges = node_ranges[leaf_greater_offset]

        # volume computation
        n_leafs = len(leaf_ranges)
        vol_leaf = np.zeros(n_leafs)
        for i in range(n_leafs):
            leaf_i_range = leaf_ranges[i]
            vol_leaf[i] = np.prod(leaf_i_range[:, 1] - leaf_i_range[:, 0])

        volume = np.sum(vol_leaf)

        return volume
