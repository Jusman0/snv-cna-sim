"""Distance matrix -> tree via deterministic NumPy neighbor joining (spec 4.5).

DendroPy's own nj_tree() breaks ties by memory address, so the same seed can give
different trees in different processes. This NJ joins the lowest-index pair among
(near-)tied minima, so a given matrix always gives the same tree.
"""
import io

import dendropy
import numpy as np

TIE_RTOL = 1e-9  # Q entries within TIE_RTOL * max|Q| of the minimum count as tied


def _num(x):
    return repr(float(x))  # plain float text; numpy 2 reprs read "np.float64(...)"


def nj_newick(D, labels):
    """Neighbor joining on D (rows in `labels` order). Returns an unrooted Newick string."""
    D = np.array(D, dtype=float)
    n = len(labels)
    if D.shape != (n, n):
        raise ValueError(f"matrix is {D.shape}, expected {(n, n)}")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance matrix contains NaN/inf; skip this source instead")
    if n < 3:
        raise ValueError("need at least 3 taxa")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    subtrees = list(labels)
    active = list(range(n))  # slot indices; a merged node reuses the lower slot

    while len(active) > 3:
        m = len(active)
        sub = D[np.ix_(active, active)]
        R = sub.sum(axis=1)
        Q = (m - 2) * sub - R[:, None] - R[None, :]
        np.fill_diagonal(Q, np.inf)
        q_min = Q.min()
        tol = TIE_RTOL * np.abs(Q[np.isfinite(Q)]).max()
        tied = np.argwhere(np.triu(Q <= q_min + tol, k=1))
        a, b = tied[0]  # argwhere is row-major, so this is the lowest-index pair
        i, j = active[a], active[b]
        d_ij = sub[a, b]
        len_i = 0.5 * d_ij + (R[a] - R[b]) / (2 * (m - 2))
        len_j = d_ij - len_i
        others = [k for k in active if k != i and k != j]
        new_row = 0.5 * (D[i, others] + D[j, others] - d_ij)
        D[i, others] = new_row
        D[others, i] = new_row
        subtrees[i] = f"({subtrees[i]}:{_num(len_i)},{subtrees[j]}:{_num(len_j)})"
        active.remove(j)

    a, b, c = active
    len_a = 0.5 * (D[a, b] + D[a, c] - D[b, c])
    len_b = 0.5 * (D[a, b] + D[b, c] - D[a, c])
    len_c = 0.5 * (D[a, c] + D[b, c] - D[a, b])
    return (f"({subtrees[a]}:{_num(len_a)},{subtrees[b]}:{_num(len_b)},"
            f"{subtrees[c]}:{_num(len_c)});")


def nj_from_matrix(D, taxon_namespace):
    """Deterministic NJ tree on the shared namespace; row i of D is taxon_namespace[i]."""
    labels = [t.label for t in taxon_namespace]
    n_before = len(taxon_namespace)
    tree = dendropy.Tree.get(data=nj_newick(D, labels), schema="newick",
                             taxon_namespace=taxon_namespace, rooting="force-unrooted")
    if len(taxon_namespace) != n_before:
        raise RuntimeError("Newick labels did not match the namespace")
    return tree


def nj_dendropy(D, taxon_namespace):
    """DendroPy's own NJ (spec 4.5 pattern). Non-deterministic on ties; cross-checks only."""
    labels = [t.label for t in taxon_namespace]
    csv = "x," + ",".join(labels) + "\n" + "\n".join(
        labels[i] + "," + ",".join(str(D[i, j]) for j in range(len(labels)))
        for i in range(len(labels)))
    pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
        src=io.StringIO(csv), taxon_namespace=taxon_namespace,
        is_first_row_column_names=True, is_first_column_row_names=True)
    tree = pdm.nj_tree()
    tree.is_rooted = False
    return tree
