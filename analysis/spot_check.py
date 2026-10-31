"""Stage 7 spot-check: recompute 3 random rows of results.csv along an independent path.

Distances are recomputed with plain Python loops, and RF is recomputed from leaf-set
splits written here (not dendropy.treecompare). Writes results/spot_check.txt.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.reconstruct import nj_from_matrix  # noqa: E402
from src.run_sweep import load_config, simulate_clean  # noqa: E402
from src.simulate_cna import add_cna_noise  # noqa: E402
from src.simulate_snv import add_snv_noise, drop_empty_sites  # noqa: E402


def loop_distance(X, metric):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = [abs(float(a) - float(b)) for a, b in zip(X[i], X[j])]
            D[i, j] = sum(1 for d in diff if d != 0) if metric == "hamming" else sum(diff)
    return D / X.shape[1]


def mean_normalize(D):
    off = ~np.eye(len(D), dtype=bool)
    return D / D[off].mean()


def splits(tree):
    """Non-trivial splits as the leaf set on the side without c0."""
    everyone = frozenset(t.label for t in tree.taxon_namespace)
    out = set()
    for node in tree.postorder_node_iter():
        if node.parent_node is None:
            continue
        below = frozenset(leaf.taxon.label for leaf in node.leaf_iter())
        if 1 < len(below) < len(everyone) - 1:
            out.add(below if "c0" not in below else everyone - below)
    return out


def main(n_rows=3, pick_seed=7):
    cfg = load_config()
    df = pd.read_csv(ROOT / "results" / "results.csv")
    picks = df.sample(n=n_rows, random_state=pick_seed)
    lines = [f"Spot-check of {n_rows} rows (picked with random_state={pick_seed})", ""]
    all_ok = True
    for _, r in picks.iterrows():
        tree, M_clean, C_clean, ss = simulate_clean(cfg, int(r["seed"]))
        M = drop_empty_sites(add_snv_noise(M_clean, r["alpha"], r["beta"],
                                           np.random.default_rng(ss["snv_noise"])))
        C = add_cna_noise(C_clean, r["sigma"], np.random.default_rng(ss["cna_noise"]))
        D_snv = mean_normalize(loop_distance(M, "hamming"))
        D_cna = mean_normalize(loop_distance(C, "l1"))
        w = r["w"]
        D = D_snv if w == 1 else D_cna if w == 0 else w * D_snv + (1 - w) * D_cna
        est = nj_from_matrix(D, tree.taxon_namespace)
        a, b = splits(tree), splits(est)
        n = cfg["n_cells"]
        assert len(a) == n - 3 and len(b) == n - 3
        nrf = len(a ^ b) / (2 * (n - 3))
        ok = np.isclose(nrf, r["nrf"]) and M.shape[1] == r["n_snv_sites"]
        all_ok &= bool(ok)
        lines.append(f"alpha={r['alpha']:g} sigma={r['sigma']:g} w={w:g} replicate={int(r['replicate'])} "
                     f"seed={int(r['seed'])}: csv nrf={r['nrf']:.6f} recomputed={nrf:.6f}; "
                     f"csv sites={int(r['n_snv_sites'])} recomputed={M.shape[1]}; "
                     f"shared splits {len(a & b)}/{n - 3} -> {'MATCH' if ok else 'MISMATCH'}")
    lines += ["", "ALL MATCH" if all_ok else "MISMATCH FOUND"]
    (ROOT / "results" / "spot_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
