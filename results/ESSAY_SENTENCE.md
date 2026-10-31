# Essay sentence

> In a 40-cell tumor-phylogeny simulation I ran with AI coding assistance, a naive 50/50 average of normalized SNV and copy-number distance matrices rebuilt trees more accurately than either data type alone when no SNVs dropped out (normalized Robinson–Foulds error 0.109 vs 0.161 for copy number alone, without copy-number noise), but once 20% or more of SNVs dropped out it was worse than copy number alone in 23 of 24 noise settings (20 simulated trees, reused across settings).

Shorter variant:

> In a tumor-phylogeny simulation I ran with AI coding assistance, naively averaging SNV and copy-number distances beat either source alone when SNV calls were clean, but was worse than copy number alone in 23 of 24 settings once 20% or more of SNVs dropped out.

**Wording note.** "I ran with AI coding assistance" is deliberate. The code and commits were written with an AI coding assistant, and the commit history shows it. Check the application's AI policy before changing that wording; "I built" alone would overstate what you did.

**Support** (all in `SUMMARY.md`, `tables.md` and `normalization_check.txt`):
- **"More accurately than either alone" with no SNV dropout.** Finding 1. At α=0, combining beat the better single source at all six σ, by 0.028–0.051. It also beat each source separately at ≥ 3 SE in every cell.
- **0.109 vs 0.161.** At α=0, σ=0: combined 0.109±0.012 vs CNA-only 0.161±0.011, a paired difference of −0.051±0.010. That ties for the largest gain across σ.
- **"Worse … in 23 of 24".** Finding 2. The settings are α ∈ {0.2, 0.3, 0.4, 0.5} × 6 σ. Combining was worse in 23 of them, each by ≥ 2.87 paired SE. The 24th (α=0.2, σ=0) showed no clear difference. All 24 settings use the same 20 trees, so they are not independent tests.
- **Not claimed: "equal weight".** After mean-normalization, copy-number jitter shrinks copy number's effective share of the "50/50" mix (0.27 at σ=0.4, 0.11 at σ=1.0; Finding 3). Part of the harm at higher σ comes from that tilt. The σ=0 column, where no tilt occurs, still shows harm for α≥0.3.
