"""Pairwise distances, normalization and the weighted combination (spec 4.4)."""
import numpy as np


def snv_distance(M):
    """Pairwise Hamming distance / number of sites. Zero sites -> all zeros (no signal)."""
    n, n_sites = M.shape
    if n_sites == 0:
        return np.zeros((n, n))
    X = M.astype(np.int64)
    ones_vs_zeros = X @ (1 - X).T
    return (ones_vs_zeros + ones_vs_zeros.T) / n_sites


def cna_distance(C):
    """Pairwise L1 distance / number of bins."""
    C = np.asarray(C, dtype=float)
    n, n_bins = C.shape
    D = np.empty((n, n))
    for i in range(n):
        D[i] = np.abs(C - C[i]).sum(axis=1)
    return D / n_bins


def off_diagonal_mean(D):
    n = len(D)
    return (D.sum() - np.trace(D)) / (n * (n - 1))


def has_signal(D):
    """Check BEFORE normalizing: an empty source would turn into NaN, not zeros."""
    m = off_diagonal_mean(D)
    return bool(np.isfinite(m) and m > 0)


def normalize(D):
    """Scale to mean off-diagonal = 1 so both sources mean the same thing before mixing."""
    if not has_signal(D):
        raise ValueError("no signal; check has_signal() and skip this source")
    return D / off_diagonal_mean(D)


def combine(D_snv, D_cna, w):
    """w * D_snv + (1 - w) * D_cna. The endpoints use one matrix only, so an empty
    (None) source on the other side can never leak in as 0 * NaN."""
    if w == 1:
        return D_snv.copy()
    if w == 0:
        return D_cna.copy()
    return w * D_snv + (1 - w) * D_cna
