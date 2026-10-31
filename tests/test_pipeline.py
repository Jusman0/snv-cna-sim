"""Stage tests from SPEC.md Section 7. Run with: pytest"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.stats import cell_summary, holdout_best_w, verdict
from src.distances import cna_distance, combine, has_signal, normalize, snv_distance
from src.evaluate import normalized_rf
from src.reconstruct import nj_dendropy, nj_from_matrix, nj_newick
from src.run_sweep import calibration_table, load_config, run_grid
from src.simulate_cna import add_cna_noise, simulate_cnas
from src.simulate_snv import add_snv_noise, drop_empty_sites, leaf_rows, simulate_snvs
from src.simulate_tree import make_namespace, path_length_matrix, simulate_tree

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "src" / "run_sweep.py"

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


# ---------------------------------------------------------------- Stage 4
def test_zero_noise_is_identity():
    M, _ = simulate_snvs(simulate_tree(N, rng(31)), 3.0, rng(32))
    assert np.array_equal(add_snv_noise(M, 0.0, 0.0, rng(33)), M)
    C, _ = simulate_cnas(simulate_tree(N, rng(34)), mu_cna=8.0, rng=rng(35), **CNA)
    assert np.array_equal(add_cna_noise(C, 0.0, rng(36)), C.astype(float))


def test_full_dropout_removes_every_one():
    M, _ = simulate_snvs(simulate_tree(N, rng(37)), 3.0, rng(38))
    noisy = add_snv_noise(M, 1.0, 0.0, rng(39))
    assert noisy.sum() == 0
    assert drop_empty_sites(noisy).shape == (N, 0)


def test_full_dropout_triggers_skip_rule_not_nan():
    cfg = load_config()
    df, skipped = run_grid(cfg, replicates=[0], alphas=[1.0], sigmas=[0.0], beta=0.0)
    assert (df["n_snv_sites"] == 0).all()
    assert df.loc[df["w"] > 0, "nrf"].isna().all()  # every weight that needs SNVs is skipped
    cna_only = df.loc[df["w"] == 0, "nrf"]
    assert cna_only.notna().all() and (cna_only < 1).all()  # CNA-only is not poisoned
    assert len(skipped) == len(cfg["weights"]) - 1
    assert len(df) == len(cfg["weights"])  # skipped rows stay in the table


def test_empirical_noise_rates():
    r = rng(40)
    M = (r.random((2000, 500)) < 0.4).astype(np.uint8)
    alpha, beta = 0.3, 0.1
    noisy = add_snv_noise(M, alpha, beta, rng(41))
    assert abs((noisy[M == 1] == 0).mean() - alpha) < 0.02
    assert abs((noisy[M == 0] == 1).mean() - beta) < 0.02
    C = np.full((200, 5000), 2, dtype=np.int16)
    assert abs((add_cna_noise(C, 0.6, rng(42)) - C).std() - 0.6) < 0.02


# ---------------------------------------------------------------- Stage 5
def test_distances_and_normalize():
    M = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    assert np.allclose(snv_distance(M), np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]]) / 4)
    C = np.array([[2, 2, 2], [3, 2, 1], [2, 2, 0]])
    assert np.allclose(cna_distance(C), np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]]) / 3)
    D = normalize(snv_distance(M))
    assert np.isclose(D[~np.eye(3, dtype=bool)].mean(), 1.0)
    assert not has_signal(snv_distance(np.zeros((3, 0), dtype=np.uint8)))
    with pytest.raises(ValueError):
        normalize(np.zeros((3, 3)))
    assert np.array_equal(combine(D, None, 1.0), D) and np.array_equal(combine(None, D, 0.0), D)


def test_calibration_passes_at_zero_noise():
    cfg = load_config()
    tab = calibration_table(cfg)
    bar = cfg["calibration"]["max_mean_nrf"]
    assert tab.loc[1.0, "mean"] <= bar, "SNV-only fails calibration"
    assert tab.loc[0.0, "mean"] <= bar, "CNA-only fails calibration"


def test_random_distances_give_nrf_near_one():
    tns = make_namespace(N)
    true = simulate_tree(N, rng(50), taxon_namespace=tns)
    vals = []
    for k in range(30):
        A = rng(500 + k).random((N, N))
        D = A + A.T
        np.fill_diagonal(D, 0.0)
        vals.append(normalized_rf(true, nj_from_matrix(D, tns), N))
    assert np.mean(vals) >= 0.95


def _probe(tmp_path, k, dummy, nj):
    out = tmp_path / f"probe_{nj}_{k}.csv"
    subprocess.run([sys.executable, str(SWEEP), "--probe", str(out), "--dummy", str(dummy),
                    "--nj", nj], check=True, cwd=str(tmp_path))
    return out.read_bytes()


def test_rows_identical_across_processes(tmp_path):
    runs = [_probe(tmp_path, k, dummy, "numpy") for k, dummy in enumerate([0, 13_337, 271_828])]
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------- Stage 6
def test_small_grid_end_to_end(tmp_path):
    out = tmp_path / "small.csv"
    subprocess.run([sys.executable, str(SWEEP), "--small", "--out", str(out)], check=True,
                   cwd=str(tmp_path))
    df = pd.read_csv(out)
    cfg = load_config()
    assert list(df.columns) == ["alpha", "beta", "sigma", "w", "replicate", "seed",
                                "n_cells", "n_snv_sites", "n_bins", "nrf"]
    assert len(df) == 2 * 2 * 3 * len(cfg["weights"])
    assert df["nrf"].between(0, 1).all()
    assert sorted(df["seed"].unique()) == [cfg["base_seed"] + r for r in range(3)]
    # Pairing: SNV-only ignores sigma and CNA-only ignores alpha, so those rows must agree.
    for (rep, a), g in df[df["w"] == 1.0].groupby(["replicate", "alpha"]):
        assert g["nrf"].nunique() == 1
    for (rep, s), g in df[df["w"] == 0.0].groupby(["replicate", "sigma"]):
        assert g["nrf"].nunique() == 1


# ---------------------------------------------------------------- Stage 7 (analysis)
def _synthetic_results(effect):
    """Fake results.csv: combined = CNA-only + effect, exactly, in every replicate."""
    r = rng(70)
    rows = []
    for alpha in (0.0, 0.5):
        for sigma in (0.0, 1.0):
            for rep in range(20):
                base = r.uniform(0.2, 0.4)
                for w, v in ((0.0, base), (0.5, base + effect), (1.0, base + 0.3)):
                    rows.append(dict(alpha=alpha, sigma=sigma, w=w, replicate=rep, nrf=v))
    return pd.DataFrame(rows)


def test_analysis_verdicts_on_synthetic_results():
    assert verdict(0.0, 0.0) == "no clear difference"
    assert verdict(0.01, 0.0) == "combined worse"  # identical nonzero paired differences
    assert verdict(-0.05, 0.01) == "combined better" and verdict(0.01, 0.01) == "no clear difference"
    for effect, expected in ((0.05, "combined worse"), (-0.05, "combined better"),
                             (0.0, "no clear difference")):
        s = cell_summary(_synthetic_results(effect), 0.5)
        assert (s["better_single"] == "CNA-only").all()
        assert np.allclose(s["harm_vs_better_single"], effect)
        assert (s["verdict_vs_better_single"] == expected).all()
    h = holdout_best_w(_synthetic_results(-0.05), (0, 9), (10, 19))
    assert (h["best_w"] == 0.5).all() and (h["n_report"] == 10).all()
    assert np.allclose(h["report_diff"], -0.05)


def test_harness_detects_dendropy_nondeterminism(tmp_path):
    # Positive control: the same harness must be able to see DendroPy's tie-order drift.
    dummies = [0, 13_337, 271_828, 99_991, 500_009, 7, 1_000_003, 31_415]
    runs = [_probe(tmp_path, k, d, "dendropy") for k, d in enumerate(dummies)]
    assert len(set(runs)) > 1, "DendroPy route looked deterministic; the harness proves nothing"
