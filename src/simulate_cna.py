"""Segmental copy-number gains/losses on the tree, plus per-bin jitter (spec 4.3)."""
import numpy as np

from src.simulate_snv import leaf_rows


def simulate_cnas(tree, n_bins, mu_cna, seg_min, seg_max, cn_max, rng, root_cn=2):
    """Clean copy-number profiles.

    Each edge gets k ~ Poisson(mu_cna * length) events. An event picks a segment of
    length uniform in [seg_min, seg_max], start uniform in [0, n_bins - length], and
    adds +1 or -1 (equal odds), clipped to [0, cn_max]. Clipping lets a bin at 0 come
    back, a known simplification. Children inherit the parent's profile first.
    Returns (C, profiles): C is cells x bins (ints, rows in namespace order).
    """
    if not 1 <= seg_min <= seg_max <= n_bins:
        raise ValueError("need 1 <= seg_min <= seg_max <= n_bins")
    profiles = {}
    for node in tree.preorder_node_iter():
        if node is tree.seed_node:
            profiles[node] = np.full(n_bins, root_cn, dtype=np.int16)
            continue
        cn = profiles[node.parent_node].copy()
        for _ in range(int(rng.poisson(mu_cna * node.edge.length))):
            length = int(rng.integers(seg_min, seg_max + 1))
            start = int(rng.integers(0, n_bins - length + 1))
            delta = 1 if rng.random() < 0.5 else -1
            cn[start:start + length] = np.clip(cn[start:start + length] + delta, 0, cn_max)
        profiles[node] = cn
    rows = leaf_rows(tree)
    C = np.zeros((len(rows), n_bins), dtype=np.int16)
    for leaf, i in rows.items():
        C[i] = profiles[leaf]
    return C, profiles


def add_cna_noise(C, sigma, rng):
    """Add Normal(0, sigma) to every bin. Kept as floats: rounding would hide low noise."""
    return C.astype(float) + rng.normal(0.0, sigma, size=C.shape)
