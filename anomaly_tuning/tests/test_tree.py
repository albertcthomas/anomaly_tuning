import pytest

import numpy as np

from anomaly_tuning.tree import RegressionTree

# create a simple tree so that we know the exact volumes of the leafs
TREE_LEAF = -1
INTERNAL = -1  # we just need the leaf values for this test
node_count = 9
children_left = np.array(
    [1, 3, 5, TREE_LEAF, TREE_LEAF, TREE_LEAF, 7, TREE_LEAF, TREE_LEAF]
)
children_right = np.array(
    [2, 4, 6, TREE_LEAF, TREE_LEAF, TREE_LEAF, 8, TREE_LEAF, TREE_LEAF]
)
feature = np.array(
    [0, 0, 1, TREE_LEAF, TREE_LEAF, TREE_LEAF, 0, TREE_LEAF, TREE_LEAF]
)
threshold = np.array(
    [0.5, 0.2, 0.8, TREE_LEAF, TREE_LEAF, TREE_LEAF, 0.9, TREE_LEAF, TREE_LEAF]
)
value = np.array([INTERNAL, INTERNAL, INTERNAL, 0, 1, 4, INTERNAL, 2, 3])
value = value.reshape(-1, 1, 1)

# The range is the unit cube
X_range = np.array([[0, 1], [0, 1]])


class SimpleTree:
    """Simple tree class for the sake of testing.

    The only attributes are the ones needed for the tests.
    """

    def __init__(self,
                 node_count,
                 children_left,
                 children_right,
                 feature,
                 threshold,
                 value):
        self.node_count = node_count
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.value = value


simple_tree = SimpleTree(node_count, children_left, children_right, feature,
                         threshold, value)
reg_tree = RegressionTree()
reg_tree.tree_ = simple_tree


def test_volume_leafs():
    # only a subset of leafs
    offset = 2
    volume = reg_tree.volume_leafs(X_range, offset)
    assert volume == 0.5

    # only a subset of leafs
    offset = 2.5
    volume = reg_tree.volume_leafs(X_range, offset)
    assert volume == pytest.approx(0.42)

    # all the leafs
    offset = -0.5
    volume = reg_tree.volume_leafs(X_range, offset)
    assert volume == 1

    # none of the leafs
    offset = 10
    volume = reg_tree.volume_leafs(X_range, offset)
    assert volume == 0
