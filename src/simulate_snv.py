"""Infinite-sites SNVs on the tree, plus dropout / false-positive noise (spec 4.2)."""
import numpy as np


def leaf_rows(tree):
    """Map each leaf node to its row: the index of its taxon in the namespace."""
    index = {t: i for i, t in enumerate(tree.taxon_namespace)}
    return {leaf: index[leaf.taxon] for leaf in tree.leaf_node_iter()}


def simulate_snvs(tree, mu_snv, rng):
    """Clean SNV data under infinite sites.

    Each edge gets k ~ Poisson(mu_snv * length) new sites. Returns (M, genotypes):
    M is cells x sites (uint8, rows in namespace order) and genotypes maps every
    node, internal ones included, to its clean 0/1 vector over the same sites.
    """
    edge_sites = {}
    n_sites = 0
    for node in tree.preorder_node_iter():
        k = 0 if node is tree.seed_node else int(rng.poisson(mu_snv * node.edge.length))
        edge_sites[node] = np.arange(n_sites, n_sites + k)
        n_sites += k
    genotypes = {}
    for node in tree.preorder_node_iter():
        if node is tree.seed_node:
            g = np.zeros(n_sites, dtype=np.uint8)
        else:
            g = genotypes[node.parent_node].copy()
        g[edge_sites[node]] = 1
        genotypes[node] = g
    rows = leaf_rows(tree)
    M = np.zeros((len(rows), n_sites), dtype=np.uint8)
    for leaf, i in rows.items():
        M[i] = genotypes[leaf]
    return M, genotypes


def add_snv_noise(M, alpha, beta, rng):
    """True 1 -> 0 with prob. alpha (dropout); true 0 -> 1 with prob. beta (false positive).

    One uniform per entry, so for a fixed rng state a higher alpha drops a superset
    of the entries a lower alpha drops (coupled noise across the grid).
    """
    u = rng.random(M.shape)
    out = M.copy()
    out[(M == 1) & (u < alpha)] = 0
    out[(M == 0) & (u < beta)] = 1
    return out


def drop_empty_sites(M):
    """Remove all-zero columns (sites no cell is observed to carry)."""
    return M[:, M.any(axis=0)]
