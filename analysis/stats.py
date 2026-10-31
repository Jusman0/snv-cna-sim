"""Summary statistics from results.csv (spec 4.8 and rules 6.2-6.3).

Fixed before the full sweep was run. All method differences use paired,
per-replicate differences; |mean| < 2 SE is reported as "no clear difference".
"""
import numpy as np
import pandas as pd

SNV_ONLY, CNA_ONLY = 1.0, 0.0
CLEAR = 2.0  # differences need |mean| >= CLEAR * SE to count


def se(x):
    x = np.asarray(x, dtype=float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def verdict(mean, s):
    if not np.isfinite(mean) or not np.isfinite(s):
        return "n/a"
    if mean == 0 or abs(mean) < CLEAR * s:
        return "no clear difference"
    return "combined worse" if mean > 0 else "combined better"


def wide(df):
    """One row per (alpha, sigma, replicate), one column per weight."""
    return df.set_index(["alpha", "sigma", "replicate", "w"])["nrf"].unstack("w")


def paired(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return d.mean(), se(d)


def cell_summary(df, w_comb):
    """Per (alpha, sigma): means, SEs, winner of the three methods, and paired contrasts."""
    rows = []
    for (alpha, sigma), g in wide(df).groupby(level=["alpha", "sigma"]):
        snv, cna, comb = g[SNV_ONLY], g[CNA_ONLY], g[w_comb]
        row = {"alpha": alpha, "sigma": sigma, "n": len(g)}
        for w in g.columns:
            row[f"mean_w{w}"] = g[w].mean()
            row[f"se_w{w}"] = se(g[w])
        methods = {"SNV-only": snv, "CNA-only": cna, "combined": comb}
        order = sorted(methods, key=lambda m: methods[m].mean())
        row["winner"], row["runner_up"] = order[0], order[1]
        row["winner_mean"] = methods[order[0]].mean()
        d, s = paired(methods[order[0]], methods[order[1]])
        row["winner_margin"], row["winner_margin_se"] = d, s
        row["winner_clear"] = verdict(d, s) != "no clear difference"
        better = "SNV-only" if snv.mean() <= cna.mean() else "CNA-only"
        row["better_single"] = better
        for name, ref in (("vs_better_single", methods[better]), ("vs_snv", snv), ("vs_cna", cna)):
            d, s = paired(comb, ref)
            row[f"harm_{name}"], row[f"harm_{name}_se"] = d, s
            row[f"verdict_{name}"] = verdict(d, s)
        rows.append(row)
    return pd.DataFrame(rows)


def holdout_best_w(df, select, report):
    """Rule 6.2: choose the best w (and the better single source) on `select`
    replicates, then score both on the untouched `report` replicates."""
    sel = df[df["replicate"].between(*select)]
    rep = df[df["replicate"].between(*report)]
    w_sel, w_rep = wide(sel), wide(rep)
    rows = []
    for (alpha, sigma), g in w_sel.groupby(level=["alpha", "sigma"]):
        means = g.mean()
        best_w = float(means.idxmin())
        single = SNV_ONLY if means[SNV_ONLY] <= means[CNA_ONLY] else CNA_ONLY
        h = w_rep.xs((alpha, sigma), level=["alpha", "sigma"])
        d, s = paired(h[best_w], h[single])
        rows.append({"alpha": alpha, "sigma": sigma, "best_w": best_w,
                     "single_w": single, "report_mean_best_w": h[best_w].mean(),
                     "report_se_best_w": se(h[best_w]), "report_mean_single": h[single].mean(),
                     "report_diff": d, "report_diff_se": s, "verdict": verdict(d, s),
                     "n_report": len(h)})
    return pd.DataFrame(rows)
