"""Figures and tables from results/results.csv (spec 4.8).

Usage: python analysis/plots.py   -> results/figures/*.png, results/cell_summary.csv,
                                     results/holdout_best_w.csv, results/tables.md
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analysis.stats import CNA_ONLY, SNV_ONLY, cell_summary, holdout_best_w, se, wide  # noqa: E402
from src.run_sweep import load_config  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Reference palette (dataviz skill): first three categorical slots, chart chrome, diverging pair.
METHOD_COLOR = {"SNV-only": "#2a78d6", "CNA-only": "#eb6834", "combined": "#1baf7a"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("harm", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
})


def text_on(color):
    """Ink with the better contrast on a given fill."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.30 else "#ffffff"


def _grid_axes(ax, alphas, sigmas):
    ax.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
    ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_xlabel("CNA jitter σ")
    ax.set_ylabel("SNV dropout α")
    ax.set_xlim(-0.5, len(sigmas) - 0.5)
    ax.set_ylim(-0.5, len(alphas) - 0.5)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)


def _cell(ax, x, y, color):
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color,
                               edgecolor=SURFACE, linewidth=2))


def win_map(summary, cfg, path):
    alphas, sigmas = cfg["snv"]["alpha_grid"], cfg["cna"]["sigma_grid"]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for _, r in summary.iterrows():
        x, y = sigmas.index(r["sigma"]), alphas.index(r["alpha"])
        color = METHOD_COLOR[r["winner"]]
        _cell(ax, x, y, color)
        tie = "" if r["winner_clear"] else "\n≈ " + r["runner_up"]
        ax.text(x, y, f"{r['winner']}\n{r['winner_mean']:.3f}{tie}", ha="center",
                va="center", fontsize=7.5, color=text_on(color))
    _grid_axes(ax, alphas, sigmas)
    w = cfg["analysis"]["combined_w"]
    ax.set_title(f"Lowest mean nRF among SNV-only, CNA-only and combined (w = {w:g})\n"
                 f"{int(summary['n'].iloc[0])} paired replicates per cell; "
                 "“≈ X” = within 2 paired SE of runner-up X", fontsize=9.5, loc="left")
    ax.legend(handles=[Patch(color=c, label=m) for m, c in METHOD_COLOR.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def harm_map(summary, cfg, path):
    alphas, sigmas = cfg["snv"]["alpha_grid"], cfg["cna"]["sigma_grid"]
    vals = summary["harm_vs_better_single"]
    lim = max(float(np.abs(vals).max()), 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    for _, r in summary.iterrows():
        x, y = sigmas.index(r["sigma"]), alphas.index(r["alpha"])
        color = DIVERGING(norm(r["harm_vs_better_single"]))
        _cell(ax, x, y, color)
        clear = r["verdict_vs_better_single"] != "no clear difference"
        label = (f"{r['harm_vs_better_single']:+.3f}\n±{r['harm_vs_better_single_se']:.3f}"
                 + ("" if clear else "\nno clear diff."))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color=text_on(color),
                fontweight="bold" if clear else "normal")
    _grid_axes(ax, alphas, sigmas)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("nRF(combined) − nRF(better single source)\n> 0: combining made the tree worse",
                   color=INK_2)
    cbar.outline.set_visible(False)
    w = cfg["analysis"]["combined_w"]
    ax.set_title(f"Harm from combining (w = {w:g}) vs the better single source per cell\n"
                 "mean of paired per-replicate differences ± 1 SE; bold = |mean| ≥ 2 SE",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def line_plot(df, cfg, path):
    sigma = cfg["analysis"]["line_plot_sigma"]
    w_comb = cfg["analysis"]["combined_w"]
    sub = wide(df).xs(sigma, level="sigma")
    alphas = cfg["snv"]["alpha_grid"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    series = (("CNA-only", CNA_ONLY, "s"), ("combined", w_comb, "o"), ("SNV-only", SNV_ONLY, "^"))
    for name, w, marker in series:
        g = sub[w].groupby(level="alpha")
        m = g.mean().reindex(alphas)
        s = g.apply(se).reindex(alphas)
        color = METHOD_COLOR[name]
        ax.fill_between(alphas, m - s, m + s, color=color, alpha=0.18, linewidth=0)
        ax.plot(alphas, m, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{name} (w = {w:g})")
        ax.annotate(name, (alphas[-1], m.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("SNV dropout α")
    ax.set_ylabel("mean normalized RF (lower is better)")
    ax.set_xticks(alphas)
    ax.set_ylim(bottom=0)
    ax.set_xlim(alphas[0] - 0.02, alphas[-1] + 0.1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    n = int(sub.groupby(level="alpha").size().iloc[0])
    ax.set_title(f"Mean nRF vs SNV dropout at CNA jitter σ = {sigma:g} "
                 f"(bands ±1 SE, {n} replicates)", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tables_md(summary, holdout, cfg):
    def fmt(m, s):
        return f"{m:+.3f} ± {s:.3f}"

    out = ["# Result tables (generated by analysis/plots.py)", "",
           "nRF: 0 = true tree recovered, 1 = no shared splits. Differences are means of paired "
           "per-replicate differences ± 1 SE; |mean| < 2 SE = no clear difference.", "",
           f"## Combined (w = {cfg['analysis']['combined_w']:g}) minus each single source", "",
           "| α | σ | SNV-only | CNA-only | combined | comb − SNV | comb − CNA | comb − better single | verdict |",
           "|---|---|---|---|---|---|---|---|---|"]
    w = cfg["analysis"]["combined_w"]
    for _, r in summary.iterrows():
        out.append(
            f"| {r['alpha']:g} | {r['sigma']:g} | {r[f'mean_w{SNV_ONLY}']:.3f} ± {r[f'se_w{SNV_ONLY}']:.3f}"
            f" | {r[f'mean_w{CNA_ONLY}']:.3f} ± {r[f'se_w{CNA_ONLY}']:.3f}"
            f" | {r[f'mean_w{w}']:.3f} ± {r[f'se_w{w}']:.3f}"
            f" | {fmt(r['harm_vs_snv'], r['harm_vs_snv_se'])}"
            f" | {fmt(r['harm_vs_cna'], r['harm_vs_cna_se'])}"
            f" | {fmt(r['harm_vs_better_single'], r['harm_vs_better_single_se'])}"
            f" | {r['verdict_vs_better_single']} |")
    sel, rep = cfg["analysis"]["select_replicates"], cfg["analysis"]["report_replicates"]
    out += ["", f"## Best w chosen on replicates {sel[0]}–{sel[1]}, scored on {rep[0]}–{rep[1]} (rule 6.2)",
            "", "The better single source is also chosen on the selection replicates.", "",
            "| α | σ | best w | its nRF (report reps) | single source | its nRF | best w − single | verdict |",
            "|---|---|---|---|---|---|---|---|"]
    for _, r in holdout.iterrows():
        single = "SNV-only" if r["single_w"] == SNV_ONLY else "CNA-only"
        out.append(f"| {r['alpha']:g} | {r['sigma']:g} | {r['best_w']:g}"
                   f" | {r['report_mean_best_w']:.3f} ± {r['report_se_best_w']:.3f}"
                   f" | {single} | {r['report_mean_single']:.3f}"
                   f" | {fmt(r['report_diff'], r['report_diff_se'])} | {r['verdict']} |")
    return "\n".join(out) + "\n"


def main(results_csv=RESULTS / "results.csv"):
    cfg = load_config()
    df = pd.read_csv(results_csv)
    if df["nrf"].isna().any():
        raise SystemExit(f"{int(df['nrf'].isna().sum())} skipped rows; inspect run_log.txt first")
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = cell_summary(df, cfg["analysis"]["combined_w"])
    holdout = holdout_best_w(df, cfg["analysis"]["select_replicates"],
                             cfg["analysis"]["report_replicates"])
    summary.to_csv(RESULTS / "cell_summary.csv", index=False)
    holdout.to_csv(RESULTS / "holdout_best_w.csv", index=False)
    (RESULTS / "tables.md").write_text(tables_md(summary, holdout, cfg), encoding="utf-8")
    win_map(summary, cfg, FIGURES / "win_map.png")
    harm_map(summary, cfg, FIGURES / "harm_map.png")
    line_plot(df, cfg, FIGURES / f"line_sigma{cfg['analysis']['line_plot_sigma']:g}.png")
    print(f"wrote figures to {FIGURES} and tables to {RESULTS}")


if __name__ == "__main__":
    main()
