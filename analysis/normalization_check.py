"""Post-hoc diagnostic (added after the sweep, after review): does CNA jitter shift the mix?

Jitter adds a near-constant offset to every pairwise CNA L1 distance. NJ ignores a
constant offset, but normalize() divides by the mean distance, offset included. If
D_cna(sigma) ~ a + b * D_cna(0), then after normalizing, the clean CNA signal enters
the w-mix with coefficient b / m(sigma) instead of 1 / m(0). Relative to the SNV
matrix, CNA's effective share of a w = 0.5 mix is r / (1 + r) with r = b * m(0) / m(sigma)
(0.5 at sigma = 0). Uses the sweep's own replicates and noise streams; changes no result.
Writes results/normalization_check.txt.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.stats import se  # noqa: E402
from src.distances import cna_distance  # noqa: E402
from src.run_sweep import load_config, simulate_clean  # noqa: E402
from src.simulate_cna import add_cna_noise  # noqa: E402


def main():
    cfg = load_config()
    off = ~np.eye(cfg["n_cells"], dtype=bool)
    rows = []
    for rep in range(cfg["n_replicates"]):
        _, _, C, ss = simulate_clean(cfg, cfg["base_seed"] + rep)
        d0 = cna_distance(C)[off]
        for sigma in cfg["cna"]["sigma_grid"]:
            d = cna_distance(add_cna_noise(C, sigma, np.random.default_rng(ss["cna_noise"])))[off]
            slope, intercept = np.polyfit(d0, d, 1)
            r = slope * d0.mean() / d.mean()
            rows.append(dict(sigma=sigma, inflation=d.mean() / d0.mean(), slope=slope,
                             intercept=intercept, r2=np.corrcoef(d0, d)[0, 1] ** 2,
                             cna_share=r / (1 + r)))
    df = pd.DataFrame(rows)
    out = df.groupby("sigma").agg(
        mean_inflation=("inflation", "mean"), slope=("slope", "mean"),
        intercept=("intercept", "mean"), r2_min=("r2", "min"),
        cna_share_w05=("cna_share", "mean"), cna_share_se=("cna_share", se))
    text = ["Post-hoc normalization diagnostic (20 replicates, sweep noise streams)",
            "inflation = mean CNA distance at sigma / at sigma 0; slope, intercept, r2 from",
            "regressing noisy on clean pairwise distances; cna_share_w05 = CNA's effective",
            "share of the w = 0.5 mix after normalization (0.5 = intended).", "",
            out.to_string(float_format=lambda x: f"{x:.4f}")]
    (ROOT / "results" / "normalization_check.txt").write_text("\n".join(text) + "\n",
                                                               encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
