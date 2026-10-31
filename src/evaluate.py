"""Normalized Robinson-Foulds distance (spec 4.6)."""
from dendropy.calculate import treecompare


def normalized_rf(true_tree, est_tree, n_cells):
    """0 = identical unrooted topology, 1 = no shared non-trivial splits.

    Scores a clone of the true tree: encode_bipartitions() collapses the degree-2
    root in place. clone(depth=1) shares the namespace; copy.deepcopy would not.
    """
    t = true_tree.clone(depth=1)
    t.is_rooted = False
    est_tree.is_rooted = False
    t.encode_bipartitions()
    est_tree.encode_bipartitions()
    rf = treecompare.symmetric_difference(t, est_tree)
    return rf / (2 * (n_cells - 3))
