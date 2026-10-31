"""Stage tests from SPEC.md Section 7. Run with: pytest"""
import numpy as np
import pytest

from src.evaluate import normalized_rf
from src.reconstruct import nj_dendropy, nj_from_matrix, nj_newick
from src.simulate_cna import simulate_cnas
from src.simulate_snv import leaf_rows, simulate_snvs
from src.simulate_tree import make_namespace, path_length_matrix, simulate_tree

N = 40
CNA = dict(n_bins=5000, seg_min=5, seg_max=60, cn_max=8)


def rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- Stage 1
def test_tree_shape():
    tree = simulate_tree(N, rng(1))
    tns = tree.taxon_namespace
    assert [t.label for t in tns] == [f"c{i}" for i in range(N)]
    leaves = tree.leaf_nodes()
    assert sorted(leaf.taxon.label for leaf in leaves) == sorted(f"c{i}" for i in range(N))
    for node in tree.preorder_internal_node_iter():
        assert len(node.child_nodes()) == 2
    for node in tree.preorder_node_iter():
        if node is not tree.seed_node:
            assert node.edge.length > 0


def test_labels_are_shuffled():
    # Leaf-splitting order alone would put cherries on consecutive labels.
    tree = simulate_tree(N, rng(2))
    cherries = [n for n in tree.preorder_internal_node_iter()
                if all(c.is_leaf() for c in n.child_nodes())]
    consecutive = 0
    for node in cherries:
        a, b = (int(c.taxon.label[1:]) for c in node.child_nodes())
        consecutive += abs(a - b) == 1
    assert consecutive < len(cherries)


@pytest.mark.parametrize("seed", range(50))
def test_nj_recovers_true_tree_from_path_lengths(seed):
    tree = simulate_tree(N, rng(seed))
    est = nj_from_matrix(path_length_matrix(tree), tree.taxon_namespace)
    assert normalized_rf(tree, est, N) == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_numpy_nj_matches_dendropy_on_tie_free_matrices(seed):
    r = rng(1000 + seed)
    A = r.random((N, N))
    D = A + A.T
    np.fill_diagonal(D, 0.0)
    tns = make_namespace(N)
    assert normalized_rf(nj_dendropy(D, tns), nj_from_matrix(D, tns), N) == 0.0


def test_nj_is_deterministic_on_ties():
    D = np.ones((N, N))  # every pair tied
    np.fill_diagonal(D, 0.0)
    labels = [f"c{i}" for i in range(N)]
    assert nj_newick(D, labels) == nj_newick(D.copy(), labels)


def test_nj_rejects_nan():
    D = np.full((N, N), np.nan)
    with pytest.raises(ValueError):
        nj_newick(D, [f"c{i}" for i in range(N)])


def test_rf_bounds_and_true_tree_untouched():
    tree = simulate_tree(N, rng(3))
    n_nodes = len(tree.nodes())
    assert normalized_rf(tree, tree.clone(depth=1), N) == 0.0
    other = simulate_tree(N, rng(4), taxon_namespace=tree.taxon_namespace)
    assert 0.0 < normalized_rf(tree, other, N) <= 1.0
    assert len(tree.nodes()) == n_nodes  # scoring must not collapse the true tree's root


# ---------------------------------------------------------------- Stage 2
@pytest.mark.parametrize("seed", range(10))
def test_snv_superset_of_parent_and_cherries_share_ancestor(seed):
    tree = simulate_tree(N, rng(seed))
    M, geno = simulate_snvs(tree, 3.0, rng(100 + seed))
    for node in tree.preorder_node_iter():
        if node is not tree.seed_node:
            parent = geno[node.parent_node]
            assert np.all(geno[node][parent == 1] == 1)
    rows = leaf_rows(tree)
    n_cherries = 0
    for node in tree.preorder_internal_node_iter():
        kids = node.child_nodes()
        if all(k.is_leaf() for k in kids):
            a, b = (M[rows[k]] for k in kids)
            assert np.array_equal(a & b, geno[node])
            n_cherries += 1
    assert n_cherries > 0


def test_snv_rows_follow_namespace_and_every_site_is_carried():
    tree = simulate_tree(N, rng(11))
    M, geno = simulate_snvs(tree, 3.0, rng(12))
    for leaf, i in leaf_rows(tree).items():
        assert leaf.taxon is tree.taxon_namespace[i]
        assert np.array_equal(M[i], geno[leaf])
    assert M.shape[1] > 0 and M.any(axis=0).all()


def test_snv_zero_rate_gives_no_sites():
    M, _ = simulate_snvs(simulate_tree(N, rng(13)), 0.0, rng(14))
    assert M.shape == (N, 0)


# ---------------------------------------------------------------- Stage 3
@pytest.mark.parametrize("mu", [8.0, 40.0])
def test_cna_values_within_bounds(mu):
    tree = simulate_tree(N, rng(21))
    C, profiles = simulate_cnas(tree, mu_cna=mu, rng=rng(22), **CNA)
    for cn in [C, *profiles.values()]:
        assert cn.min() >= 0 and cn.max() <= CNA["cn_max"]
    assert (C != 2).any()


def test_cna_zero_rate_is_all_twos():
    C, _ = simulate_cnas(simulate_tree(N, rng(23)), mu_cna=0.0, rng=rng(24), **CNA)
    assert C.shape == (N, CNA["n_bins"]) and np.all(C == 2)


def test_cna_rows_follow_namespace():
    tree = simulate_tree(N, rng(25))
    C, profiles = simulate_cnas(tree, mu_cna=8.0, rng=rng(26), **CNA)
    for leaf, i in leaf_rows(tree).items():
        assert np.array_equal(C[i], profiles[leaf])
