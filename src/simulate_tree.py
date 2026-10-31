"""Random binary tree with exponential branch lengths (spec 4.1)."""
import dendropy
import numpy as np


def make_namespace(n_cells):
    """Namespace in label order c0..c{n-1}. Row i of every matrix is namespace[i]."""
    return dendropy.TaxonNamespace([f"c{i}" for i in range(n_cells)])


def simulate_tree(n_cells, rng, taxon_namespace=None, mean_branch_length=1.0):
    """Yule-like tree: repeatedly split a uniformly chosen leaf.

    Labels are attached to leaves in a random order, because leaf-splitting would
    otherwise give every true cherry consecutive labels, and lowest-index NJ
    tie-breaking would then resolve ties toward the true tree.
    """
    if n_cells < 4:
        raise ValueError("need at least 4 cells for a non-trivial unrooted tree")
    tns = taxon_namespace if taxon_namespace is not None else make_namespace(n_cells)
    if len(tns) != n_cells:
        raise ValueError("namespace size does not match n_cells")
    tree = dendropy.Tree(taxon_namespace=tns)
    leaves = [tree.seed_node]
    while len(leaves) < n_cells:
        node = leaves.pop(int(rng.integers(len(leaves))))
        leaves.append(node.new_child())
        leaves.append(node.new_child())
    for leaf, k in zip(leaves, rng.permutation(n_cells)):
        leaf.taxon = tns[int(k)]
    for node in tree.preorder_node_iter():
        if node is not tree.seed_node:
            node.edge.length = float(rng.exponential(mean_branch_length))
    tree.is_rooted = True
    return tree


def path_length_matrix(tree):
    """Patristic distances between leaves, rows in namespace order."""
    tns = tree.taxon_namespace
    pdm = tree.phylogenetic_distance_matrix()
    n = len(tns)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = pdm.patristic_distance(tns[i], tns[j])
    return D
