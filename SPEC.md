# Project Spec: When Do Noisy SNVs Help Tumor Tree Reconstruction?

**Owner:** Juice
**Hard deadline:** results usable in an application essay by **Nov 1, 2026** (application due Nov 2)
**Scope:** small, honest simulation study. Not a publication. Done beats perfect.

---

## 0. Instructions for the agent (read first)

- This project is **simulation only. It needs no external data.** (Checked 2026-10-30: the author's machine holds no single-cell, genomic or phylogenetic data, so there is nothing to reuse anyway.) Work **only inside this repo**; do not read, import or copy any file from outside it. The machine also holds an unrelated trading-research project and secret-bearing files. The local, git-ignored `LOCAL_BUILD_NOTES.md` lists what to stay away from.
- **The repo already exists.** This directory is the git repo, with remote `https://github.com/Jusman0/snv-cna-sim` (**public**). Do not create another repo. Commit and push after every stage in Section 7 and after every spec change, each as its own commit. Every push is public at once, so keys, `.env` files, trading code and `LOCAL_BUILD_NOTES.md` must never be committed (`.gitignore` covers them).
- **Machine-specific setup** (which Python and venv, commit identity, hook quirks) is in `LOCAL_BUILD_NOTES.md`, which exists only on the author's PC. On any other machine, create a fresh Python 3.13 virtual environment and `pip install -r requirements.txt`.
- Build in the order in Section 7. After each stage, run the listed tests before moving on.
- Report the **actual numbers you get**. Never round toward a nicer story. If a result is weak or inconclusive, say so.
- Keep it simple: NumPy + DendroPy + pandas + matplotlib. No deep learning and no external phylogenetics binaries.

---

## 1. Research question

An NSF-funded UConn project (award IIS-2212511, *Algorithms for scalable inference and phylodynamic analysis of tumor haplotypes using low-coverage single cell sequencing data*; PI Ion Măndoiu, co-PI Mukul Bansal; 2022–2026) develops scalable methods to infer tumor phylogenies from low-coverage single-cell DNA data, reconstructing clone profiles that include both **copy-number alterations (CNAs)** and **single-nucleotide variants (SNVs)**. Its award abstract does not itself pose a "combined vs. better single source" comparison; that comparison is the question this study adds.

Closest prior work: model-based joint SNV+CNA methods that benchmark against SNV-only and CNA-only baselines, e.g. ScisTreeCNA (Zhang & Wu, UConn; bioRxiv preprint 2025, doi:10.1101/2025.11.21.689819), Pharming (Weber, Hart, Ochoa & El-Kebir; *J Comput Biol* 33(1), 2026, doi:10.1177/15578666251374694) and COMPASS (Sollier et al.; *Nat Commun* 2023, doi:10.1038/s41467-023-40378-8). Those are likelihood models. This study isolates the *naive* version (mixing two distance matrices), where adding a noisier source can plausibly hurt.

**This study asks a related question at toy scale:**

> Using a simple distance-based pipeline, under what SNV and CNA noise levels does combining SNV and CNA distances beat SNV-only and CNA-only reconstruction, and when does adding the noisier source make the tree *worse*?

Ground truth is known because the tree is simulated. Real tumors never give you that, which is the whole point of simulation-based evaluation.

---

## 2. Pipeline overview

```
config.yaml
    │
    ▼
simulate_tree ──► true tree (n cells, branch lengths)
    │
    ├──► simulate_snvs ──► clean SNV matrix (cells × sites, 0/1)
    │         └──► add_snv_noise (dropout α, false-pos β) ──► noisy SNV matrix
    │
    └──► simulate_cnas ──► clean CN matrix (cells × bins, ints)
              └──► add_cna_noise (jitter σ) ──► noisy CN matrix
    │
    ▼
distances: D_snv, D_cna  (each normalized to mean off-diagonal = 1)
           D_w = w·D_snv + (1−w)·D_cna     for w ∈ {0, .25, .5, .75, 1}
    │
    ▼
nj_tree(D_w) ──► reconstructed tree
    │
    ▼
normalized RF distance vs. true tree ──► results.csv ──► plots
```

`w = 0` is CNA-only, `w = 1` is SNV-only, and the values in between are combinations. One sweep over `w` therefore covers all three methods.

---

## 3. Directory layout

```
snv-cna-sim/
├── SPEC.md                # this file
├── .gitignore             # already present; ignores secrets and LOCAL_BUILD_NOTES.md
├── .gitattributes         # already present; normalizes line endings
├── LOCAL_BUILD_NOTES.md   # author's PC only, git-ignored, never committed
├── README.md              # what, why, how to run, results summary, limitations
├── config.yaml            # all parameters; nothing hard-coded
├── requirements.txt       # pinned, verified on Python 3.13.5: numpy==2.5.4 pandas==3.1.0
│                          #   matplotlib==3.11.2 dendropy==5.1.3 pyyaml==6.0.3 pytest==9.2.0
├── src/
│   ├── simulate_tree.py   # random binary tree with branch lengths
│   ├── simulate_snv.py    # infinite-sites mutations + noise
│   ├── simulate_cna.py    # segmental gains/losses + noise
│   ├── distances.py       # Hamming, L1, normalization, weighted combo
│   ├── reconstruct.py     # matrix → deterministic NumPy NJ → DendroPy tree
│   ├── evaluate.py        # normalized Robinson–Foulds
│   └── run_sweep.py       # grid × replicates → results/results.csv
├── analysis/
│   └── plots.py           # figures from results.csv
├── results/               # results.csv, figures/*.png, run_log.txt
└── tests/
    └── test_pipeline.py
```

---

## 4. Module specifications

### 4.1 `simulate_tree.py`
- `simulate_tree(n_cells, rng) -> dendropy.Tree`
- Random binary tree over `n_cells` leaves named `c0..c{n-1}`. Simplest valid method: start with a root and repeatedly split a uniformly chosen leaf (a Yule-like process). Then assign each edge a length drawn from `Exponential(mean=1.0)`.
- Every tree in a replicate must share **one `dendropy.TaxonNamespace`**, or RF comparison will fail. Create it first, in label order `c0..c{n-1}`; row and column `i` of every data and distance matrix is `taxon_namespace[i]`.
- Attach the labels to the leaves in a random order. Splitting leaves otherwise gives every true cherry consecutive labels. A paired check on 100 trees found no detectable bias from this, but shuffling is free insurance.
- Never `copy.deepcopy` a tree: the copy gets a new namespace and RF fails with `TaxonNamespaceIdentityError`. Use `tree.clone(depth=1)`, which shares the namespace.

### 4.2 `simulate_snv.py`
- **Clean data (infinite-sites model):** for each edge, draw `k ~ Poisson(mu_snv × edge_length)` new mutations. Each mutation is a new site (column). A cell has a 1 at a site if the mutation lies on the path from the root to that cell.
- **Noise:** `add_snv_noise(M, alpha, beta, rng)`
  - true 1 → observed 0 with probability `alpha` (allelic dropout / false negative)
  - true 0 → observed 1 with probability `beta` (false positive)
- Keep the clean genotypes of internal nodes as well as leaves; the Stage 2 superset test needs each node's parent.
- Drop all-zero columns after noise. Log the site count. If no sites remain (possible only when β = 0, e.g. the α = 1 test), the SNV source has no signal: apply the skip rule in 4.4.

### 4.3 `simulate_cna.py`
- Genome of `n_bins` bins. Root copy number = 2 everywhere.
- **Clean data:** for each edge, draw `k ~ Poisson(mu_cna × edge_length)` events. Each event picks a random contiguous segment (length uniform in `[seg_min, seg_max]` bins, start uniform in `[0, n_bins − length]`) and applies +1 or −1 with equal probability, clipped to `[0, cn_max]`. Clipping means a bin at 0 can come back to 1, which real biology does not allow; accept this simplification. Children inherit the parent's profile, then apply their own events.
- **Noise:** `add_cna_noise(C, sigma, rng)` adds `Normal(0, sigma)` to every bin. **Keep the result as floats; do not round.** Rounding hides low noise and makes the curve stair-stepped.

### 4.4 `distances.py`
- `snv_distance(M)`: pairwise Hamming distance divided by the number of sites. (The division cancels in `normalize()`; it only keeps logged values readable.)
- `cna_distance(C)`: pairwise L1 distance divided by the number of bins.
- `normalize(D)`: divide by the mean of the off-diagonal entries, so both sources are on the same scale before mixing. **This step is essential.** Without it, `w` means nothing.
- `combine(D_snv, D_cna, w) = w·D_snv + (1−w)·D_cna`
- Edge case: check **before** normalizing. A source with zero sites, or an off-diagonal mean of 0, gives NaN after normalizing, not zeros, so an "all zeros" check never fires. This was verified: Hamming ÷ 0 sites is all-NaN, and DendroPy's NJ accepts NaN silently (nRF 0.973). Because 0·NaN = NaN, `combine(NaN, D_cna, w=0)` also poisons CNA-only. In that case log a warning, skip only the weights that need the empty source (every w > 0 for an empty SNV source, every w < 1 for an empty CNA source), and list the skipped rows in `run_log.txt`.

### 4.5 `reconstruct.py`
`nj_from_matrix(D, taxon_namespace) -> dendropy.Tree`

**Use a deterministic NJ.** DendroPy 5.1.3's `nj_tree()` runs with the pattern below unchanged, but it breaks ties by memory address. As a result, the same seed can give different trees in different processes. This was verified: 2 of 20 replicates changed between runs (for example nRF 0.1351 vs 0.1622). Rescaling a matrix by a factor that changes nothing in theory (error ≤ 4.4e-16) flipped a tree by RF = 8. Integer Hamming distances create many exact ties, and every results.csv row must be reproducible from its seed. So:

- implement NJ in NumPy (about 40 lines), breaking ties by lowest index (`np.argmin` on the Q matrix);
- write the result as Newick with the `c` labels;
- parse it into the shared namespace:

```python
est_tree = dendropy.Tree.get(data=newick, schema="newick",
                             taxon_namespace=taxon_namespace, rooting="force-unrooted")
```

Validate it in two ways: with the Stage 1 recovery test, and by checking it matches DendroPy's `nj_tree()` (RF = 0) on tie-free random float matrices. If time runs short, fall back to DendroPy's NJ and state the non-reproducibility in the README limitations.

The DendroPy route below runs as written on DendroPy 5.1.3; use it for that cross-check. Row and column `i` must be `taxon_namespace[i]`. If the namespace order differs from the matrix order (for example, a namespace filled by reading a Newick file), a perfect matrix silently scores nRF 0.865.

```python
import io, dendropy
labels = [t.label for t in taxon_namespace]
csv = "x," + ",".join(labels) + "\n" + "\n".join(
    labels[i] + "," + ",".join(str(D[i, j]) for j in range(len(labels)))
    for i in range(len(labels)))
pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
    src=io.StringIO(csv), taxon_namespace=taxon_namespace,
    is_first_row_column_names=True, is_first_column_row_names=True)
tree = pdm.nj_tree()
tree.is_rooted = False
```

### 4.6 `evaluate.py`
`normalized_rf(true_tree, est_tree, n_cells) -> float`

```python
from dendropy.calculate import treecompare
t = true_tree.clone(depth=1)      # encode_bipartitions() edits the tree in place; never deepcopy
t.is_rooted = False; est_tree.is_rooted = False
t.encode_bipartitions(); est_tree.encode_bipartitions()
rf = treecompare.symmetric_difference(t, est_tree)
return rf / (2 * (n_cells - 3))   # 0 = identical, 1 = no shared splits
```

Verified behavior (DendroPy 5.1.3):
- **Root collapse.** With `is_rooted = False`, `encode_bipartitions()` silently collapses the simulated tree's degree-2 root. That leaves n − 3 = 37 non-trivial splits for 40 cells, which is correct. The collapse modifies the tree in place, which is why the snippet scores a clone: code that walks the true tree afterwards would otherwise see different nodes.
- **Rooting flag.** If the true tree is left `is_rooted = True`, a topologically perfect estimate scores nRF 0.149, with no error raised. An in-memory tree defaults to `is_rooted = None`, which is treated as unrooted, but set `False` explicitly anyway. The old comment about `rooting="force-unrooted"` applies only when reading a tree from a file.

**Gotcha (verified):** if the true tree has no branch lengths, its path distances are all 0, and NJ returns the same wrong tree on every run (nRF 0.973). Branch lengths must exist.

### 4.7 `run_sweep.py`
- Loop over: SNV dropout `alpha` × CNA jitter `sigma` × replicate seed.
- **Within each replicate**, simulate the tree and the clean data once, apply noise, then evaluate **every** `w` on that same noisy data. This pairs the comparison, so every method sees an identical tree and identical noise.
- Seeds: `seed = base_seed + replicate_index`, logged per row. All 36 (α, σ) cells of a replicate therefore share the same seed. They share the same 20 trees and clean data, and only the noise level differs. That is intended (common random numbers); say so in the README.
- Output `results/results.csv` with columns:
  `alpha, beta, sigma, w, replicate, seed, n_cells, n_snv_sites, n_bins, nrf`

### 4.8 `analysis/plots.py`
1. **Win-map heatmap.** x = `sigma`, y = `alpha`, color = which of {SNV-only, CNA-only, combined w=0.5} has the lowest mean nRF. Annotate each cell with that mean.
2. **Harm heatmap.** Color = `nRF(combined w=0.5) − min(nRF SNV-only, nRF CNA-only)`. Positive means combining made the tree *worse*. Use a diverging colormap centered at 0. This is the key figure.
3. **Line plot.** At `sigma` = 0.4, fixed before the sweep (a six-value grid has no single middle value), plot mean nRF vs. `alpha` for w ∈ {0, 0.5, 1}, with ±1 standard-error bands.

---

## 5. Default config (`config.yaml`)

```yaml
base_seed: 2027
n_replicates: 20
n_cells: 40

snv:
  mu_snv: 3.0          # mutations per unit branch length
  beta: 0.01           # false-positive rate (fixed)
  alpha_grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

cna:
  n_bins: 5000         # was 500: with 500 bins CNA-only cannot pass calibration (see below)
  mu_cna: 8.0          # CNA events per unit branch length (was 1.0)
  seg_min: 5
  seg_max: 60
  cn_max: 8
  sigma_grid: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

weights: [0.0, 0.25, 0.5, 0.75, 1.0]
```

Run size: 6 × 6 × 20 = 720 replicates × 5 weights = 3,600 trees. This should take minutes on a laptop. A throwaway implementation measured 67 ms per replicate at 500 bins, about 48 s for the grid. If it takes longer, profile before optimizing anything.

**Calibration step:** before the full sweep, check at zero noise (α = β = σ = 0) that the **mean** nRF over the 20 replicates is ≤ 0.2 for both SNV-only and CNA-only. A throwaway implementation of Sections 4.1–4.6 measured the following (2026-10-30; mean ± SE over 20 replicates):

| Setting | SNV-only nRF | CNA-only nRF |
|---|---|---|
| Original defaults (`mu_snv` 3, `n_bins` 500, `mu_cna` 1) | 0.169 ± 0.018 (pass) | 0.553 ± 0.024 (fail) |
| `n_bins` 500, `mu_cna` 2 / 4 / 8 | — | 0.468 / 0.465 / 0.472 (fail: plateau) |
| `n_bins` 5000, `mu_cna` 4 | — | 0.192 ± 0.018 (borderline) |
| `n_bins` 5000, `mu_cna` 8 (**new default**) | — | 0.155 ± 0.012 (pass) |
| `mu_snv` 5 / 10 | 0.115 / 0.055 | — |

The 5000-bin rows used different random draws from the others.

- **Why raising the rate alone fails.** With 500 bins, overlapping 5–60-bin segments saturate the genome, so L1 distance grows much more slowly than tree distance. The log-log slope against path length is 0.52 at `mu_cna` 8.
- **What to do if a source fails.** Raise `n_bins` (or shorten segments) as well as the rate. Write every change in `run_log.txt`, and re-check with the project's own code, since these numbers come from a throwaway implementation.
- **Do not change rates after looking at the noise-sweep results.**

**Settle before the sweep:** with 10× more bins, per-bin jitter σ averages over many more bins, so the σ grid `[0, 1.0]` may barely move CNA-only. Decide the σ grid now, from this reasoning, not after seeing sweep results.

---

## 6. Rigor rules (non-negotiable)

1. **Paired comparisons.** All weights are evaluated on the same tree and the same noise draw (Section 4.7).
2. **No tuning on reported data.** If we report a "best w," choose it on replicate indices 0–9 (seeds 2027–2036) and report its nRF on replicates 10–19 (seeds 2037–2046). Picking the best of 5 weights on the same data inflates results, which is the same leakage problem as in ML.
3. **Report uncertainty.** Every mean is shown with its standard error, and differences smaller than about 2 SE are described as "no clear difference." Because the design is paired, the SE of a difference between methods (including the harm heatmap) is the SE of the per-replicate paired differences, not one computed from two separate SEs.
4. **Freeze the config.** Copy the final `config.yaml`, the git commit hash and `pip freeze` output into `results/run_log.txt`.

---

## 7. Build order and tests

| Stage | Build | Must pass before moving on |
|---|---|---|
| 1 | `simulate_tree`, `reconstruct`, `evaluate` | **NJ recovery test:** NJ on the true tree's own path-length distances returns nRF = 0 exactly. A throwaway check passed on 50/50 seeds. The NumPy NJ matches DendroPy's `nj_tree()` (RF = 0) on tie-free random float matrices. |
| 2 | `simulate_snv` (clean) | Each node's mutation set is a superset of its parent's (uses the internal-node genotypes from 4.2); sibling cells share exactly their common ancestor's mutations. |
| 3 | `simulate_cna` (clean) | All values are in `[0, cn_max]`; with `mu_cna = 0`, every cell is all 2s. |
| 4 | noise functions | With α=β=0 / σ=0 output equals input. With α=1, every 1 becomes 0; with β = 0 that leaves zero sites, so assert that the 4.4 skip rule fires instead of a NaN reaching NJ. Empirical flip rates are within ±0.02 of the requested rates on a large matrix. |
| 5 | `distances` + full single run | Zero noise passes the calibration step (Section 5). **Fully random** symmetric distance matrices give mean nRF ≥ 0.95 over ≥ 20 draws (measured 0.992 ± 0.002 at n = 40). **Determinism:** the same replicate run twice, in separate processes, gives identical rows. |
| 6 | `run_sweep` | Small grid (2×2, 3 replicates) runs end to end and the CSV has the expected row count. |
| 7 | full sweep + `plots` | Figures render; spot-check 3 rows by hand. |

All tests go in `tests/test_pipeline.py` and run with `pytest`.

---

## 8. Deliverables

1. Public repo `snv-cna-sim/` with all code, tests, and `results/`.
2. Three figures (Section 4.8).
3. A **`results/SUMMARY.md`**, under 250 words, written plainly:
   - Two or three headline findings with **exact numbers** and SEs. Example of the shape (not the content): *"With CNA jitter σ=0.4, combining (w=0.5) beat CNA-only until SNV dropout reached α≈__; above that, it was worse by __ ± __ nRF."*
   - Where combining helped, where it hurt, and where there was no clear difference.
   - The limitations (Section 9), stated briefly.
4. **One sentence suitable for an application essay**, which must be fully supported by the numbers in SUMMARY.md. No extrapolating.

---

## 9. Known limitations (put these in the README; do not try to fix them before the deadline)

- **Naive combination.** A weighted sum of distances is not how model-based methods work. COMPASS fits a joint likelihood over SNV and CNA read counts that includes dropout and doublets. SCARLET (Satas et al., *Cell Systems* 2020) models SNV read-count error and takes a copy-number tree as given, allowing an SNV to be lost only where that tree shows a deletion. This study tests the *naive* approach only.
- **No SNV–CNA interaction.** In real tumors, a deletion of the copy carrying an SNV can erase it. Here the two processes are independent.
- **Simplified noise.** No doublets, no missing data, no breakpoint-boundary errors in CNAs.
- **Distance-based only.** NJ is one reconstruction method. The Bansal lab's DICE (Weiner & Bansal, *Life Sci Alliance* 2025) is also distance-based but CNA-only: it builds trees from copy-number distances with FastME, typically balanced minimum evolution (NJ is an option). That makes this a reasonable but limited baseline.
- **Small scale.** 40 cells, versus hundreds to thousands in current single-cell DNA datasets.

**Stretch goals, only if everything above is done by Oct 31:** (a) SNV loss when the covering bin hits copy number 0; (b) replace the hand-rolled CNA simulator with the lab's CNAsim (Weiner & Bansal, *Bioinformatics* 2023). CNAsim simulates copy number only (profiles, read counts or reads), not SNVs, so it would replace only the CNA half; its error model adds boundary errors and jitter; (c) add a missing-data rate for SNVs.
