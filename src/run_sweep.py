"""Grid x replicates -> results/results.csv (spec 4.7).

Usage (any working directory):
  python src/run_sweep.py                 full sweep -> results/results.csv + results/run_log.txt
  python src/run_sweep.py --calibrate     zero-noise check (spec 5) -> results/calibration.txt
  python src/run_sweep.py --small --out F 2x2 grid, 3 replicates (Stage 6 test)
  python src/run_sweep.py --probe F --dummy N [--nj dendropy]
                                          zero-noise rows for the determinism test
"""
import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.distances import cna_distance, combine, has_signal, normalize, snv_distance  # noqa: E402
from src.evaluate import normalized_rf  # noqa: E402
from src.reconstruct import nj_dendropy, nj_from_matrix  # noqa: E402
from src.simulate_cna import add_cna_noise, simulate_cnas  # noqa: E402
from src.simulate_snv import add_snv_noise, drop_empty_sites, simulate_snvs  # noqa: E402
from src.simulate_tree import simulate_tree  # noqa: E402

COLUMNS = ["alpha", "beta", "sigma", "w", "replicate", "seed",
           "n_cells", "n_snv_sites", "n_bins", "nrf"]
STREAMS = ("tree", "snv", "cna", "snv_noise", "cna_noise")
NJ_ROUTES = {"numpy": nj_from_matrix, "dendropy": nj_dendropy}
RESULTS = ROOT / "results"


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def streams(seed):
    """Independent RNG streams for one replicate (seed = base_seed + replicate)."""
    return dict(zip(STREAMS, np.random.SeedSequence(seed).spawn(len(STREAMS))))


def simulate_clean(cfg, seed):
    """True tree and clean data, simulated once per replicate."""
    ss = streams(seed)
    tree = simulate_tree(cfg["n_cells"], np.random.default_rng(ss["tree"]),
                         mean_branch_length=cfg.get("mean_branch_length", 1.0))
    M, _ = simulate_snvs(tree, cfg["snv"]["mu_snv"], np.random.default_rng(ss["snv"]))
    c = cfg["cna"]
    C, _ = simulate_cnas(tree, c["n_bins"], c["mu_cna"], c["seg_min"], c["seg_max"],
                         c["cn_max"], np.random.default_rng(ss["cna"]))
    return tree, M, C, ss


def run_replicate(cfg, replicate, alphas, sigmas, beta, weights, nj_route="numpy"):
    """Every (alpha, sigma, w) for one replicate, all on the same tree and clean data.

    The noise streams restart for each grid value, so every alpha sees the same
    uniforms and every sigma the same standard normals (common random numbers),
    and every w within a cell sees one identical noise draw (paired comparison).
    """
    seed = cfg["base_seed"] + replicate
    tree, M_clean, C_clean, ss = simulate_clean(cfg, seed)
    n_cells, n_bins = cfg["n_cells"], cfg["cna"]["n_bins"]
    nj = NJ_ROUTES[nj_route]

    snv = {}
    for alpha in alphas:
        M = drop_empty_sites(add_snv_noise(M_clean, alpha, beta,
                                           np.random.default_rng(ss["snv_noise"])))
        D = snv_distance(M)
        snv[alpha] = (M.shape[1], normalize(D) if has_signal(D) else None)
    cna = {}
    for sigma in sigmas:
        D = cna_distance(add_cna_noise(C_clean, sigma, np.random.default_rng(ss["cna_noise"])))
        cna[sigma] = normalize(D) if has_signal(D) else None

    rows, skipped = [], []
    for alpha in alphas:
        n_sites, D_snv = snv[alpha]
        for sigma in sigmas:
            D_cna = cna[sigma]
            for w in weights:
                row = dict(alpha=alpha, beta=beta, sigma=sigma, w=w, replicate=replicate,
                           seed=seed, n_cells=n_cells, n_snv_sites=n_sites, n_bins=n_bins,
                           nrf=np.nan)
                missing = [name for name, D, needed in (("snv", D_snv, w > 0),
                                                        ("cna", D_cna, w < 1))
                           if needed and D is None]
                if missing:
                    skipped.append({**row, "reason": "no signal in " + "+".join(missing)})
                else:
                    est = nj(combine(D_snv, D_cna, w), tree.taxon_namespace)
                    row["nrf"] = normalized_rf(tree, est, n_cells)
                rows.append(row)
    return rows, skipped


def run_grid(cfg, replicates=None, alphas=None, sigmas=None, beta=None, weights=None,
             nj_route="numpy"):
    replicates = range(cfg["n_replicates"]) if replicates is None else replicates
    alphas = cfg["snv"]["alpha_grid"] if alphas is None else alphas
    sigmas = cfg["cna"]["sigma_grid"] if sigmas is None else sigmas
    beta = cfg["snv"]["beta"] if beta is None else beta
    weights = cfg["weights"] if weights is None else weights
    rows, skipped = [], []
    for r in replicates:
        rr, ss = run_replicate(cfg, r, alphas, sigmas, beta, weights, nj_route)
        rows += rr
        skipped += ss
    return pd.DataFrame(rows, columns=COLUMNS), skipped


def calibration_table(cfg, nj_route="numpy"):
    """Zero noise (alpha = beta = sigma = 0): mean +/- SE nRF per weight."""
    df, _ = run_grid(cfg, alphas=[0.0], sigmas=[0.0], beta=0.0, nj_route=nj_route)
    g = df.groupby("w")["nrf"]
    return pd.DataFrame({"mean": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.count()),
                         "min": g.min(), "max": g.max(), "n": g.count()})


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _pip_freeze():
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return out.stdout.strip()


def cmd_calibrate(cfg):
    t0 = time.perf_counter()
    tab = calibration_table(cfg)
    bar = cfg["calibration"]["max_mean_nrf"]
    ok = {name: tab.loc[w, "mean"] <= bar for name, w in (("SNV-only", 1.0), ("CNA-only", 0.0))}
    lines = [f"Calibration at zero noise (alpha = beta = sigma = 0), {cfg['n_replicates']} replicates, "
             f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             f"mu_snv={cfg['snv']['mu_snv']} n_bins={cfg['cna']['n_bins']} "
             f"mu_cna={cfg['cna']['mu_cna']} seg=[{cfg['cna']['seg_min']},{cfg['cna']['seg_max']}]",
             tab.to_string(float_format=lambda x: f"{x:.4f}"),
             *(f"{name}: mean {tab.loc[w, 'mean']:.4f} -> {'PASS' if ok[name] else 'FAIL'} (bar {bar})"
               for name, w in (("SNV-only", 1.0), ("CNA-only", 0.0))),
             f"runtime {time.perf_counter() - t0:.1f} s"]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return all(ok.values())


def cmd_sweep(cfg, out):
    t0 = time.perf_counter()
    df, skipped = run_grid(cfg)
    runtime = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sites = df.groupby("alpha")["n_snv_sites"].agg(["min", "median", "max"])
    calib = RESULTS / "calibration.txt"
    log = [f"run finished {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
           f"git commit {_git('rev-parse', 'HEAD')}"
           f" ({'dirty' if _git('status', '--porcelain', '--untracked-files=no') else 'clean'})",
           f"python {platform.python_version()} on {platform.platform()}",
           f"rows {len(df)}; nrf NaN rows {int(df['nrf'].isna().sum())}; runtime {runtime:.1f} s",
           "", "== skipped rows (no signal) ==",
           *([str(s) for s in skipped] or ["none"]),
           "", "== SNV sites after noise, by alpha ==", sites.to_string(),
           "", "== config.yaml (frozen) ==", (ROOT / "config.yaml").read_text(encoding="utf-8"),
           "== calibration ==", calib.read_text(encoding="utf-8") if calib.exists() else "not run",
           "== pip freeze ==", _pip_freeze()]
    (out.parent / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(df)} rows) in {runtime:.1f} s")


def cmd_probe(cfg, out, dummy, nj_route):
    """Zero-noise rows for all replicates and weights, after shifting the heap."""
    junk = [bytearray(1 + (k * 7919) % 257) for k in range(dummy)]  # noqa: F841
    df, _ = run_grid(cfg, alphas=[0.0], sigmas=[0.0], beta=0.0, nj_route=nj_route)
    df.to_csv(out, index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--out", type=Path, default=RESULTS / "results.csv")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--small", action="store_true")
    p.add_argument("--probe", type=Path)
    p.add_argument("--dummy", type=int, default=0)
    p.add_argument("--nj", choices=sorted(NJ_ROUTES), default="numpy")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.calibrate:
        return 0 if cmd_calibrate(cfg) else 1
    if a.probe:
        cmd_probe(cfg, a.probe, a.dummy, a.nj)
        return 0
    if a.small:
        df, _ = run_grid(cfg, replicates=range(3), alphas=cfg["snv"]["alpha_grid"][:2],
                         sigmas=cfg["cna"]["sigma_grid"][:2])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        return 0
    cmd_sweep(cfg, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
