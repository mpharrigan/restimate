__author__ = 'harrigan'

from sklearn import cluster
import numpy as np


class TreeTraverser(object):
    def __init__(self, depth, children, n_leaves):

        self.labels = -1 * np.ones(n_leaves, dtype=int)
        self.maxdepth = depth
        self.n_leaves = n_leaves
        self.children = children

        self._traverse_tree(-1, 0, 0)

    def _traverse_tree(self, node_i, label, depth):

        if 2 ** (self.maxdepth - 1) >= self.n_leaves:
            # Special case for each leaf in its own cluster
            self.labels = np.arange(self.n_leaves)
            return

        if depth < self.maxdepth:
            label = node_i

        wc = self.children[node_i]
        tn = wc - self.n_leaves
        if np.all(tn >= 0):
            self._traverse_tree(tn[0], label, depth + 1)
            self._traverse_tree(tn[1], label, depth + 1)
        else:
            self.labels[wc[0]] = label
            self.labels[wc[1]] = label


class WardTree(object):
    def __init__(self):
        ward = cluster.Ward(n_clusters=1, compute_full_tree=True)
        self.ward = ward
        self.n_levels = 0

    def fit(self, pos):
        self.ward.fit(pos)
        self.n_levels = int(np.log2(len(pos))) + 1

    def labels_at_depth(self, depth):
        tt = TreeTraverser(depth, self.ward.children_, self.ward.n_leaves_)
        return tt.labels

    def complete_labels(self):
        levels = range(1, self.n_levels + 1)
        complete_labels = np.zeros((self.ward.n_leaves_, self.n_levels), dtype=int)
        for i, level in enumerate(levels):
            labels = self.labels_at_depth(level)
            complete_labels[:, i] = np.asarray(labels).T

        return complete_labels

def get_heir_assignments_from_state_indices(states):
    #TODO: Implement this function to turn a list of state indices
    # into a 2d array. Row: time, col: assignment at specified level
    # Maybe make this yield, and instead just act as a translator
    pass

def get_count_matrix_from_assignments(assignments, lag_time, n_states):
    pass
