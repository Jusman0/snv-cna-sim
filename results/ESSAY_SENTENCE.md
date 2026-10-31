# Essay sentence

> In a 40-cell simulation I built, naively averaging SNV and copy-number distance matrices rebuilt simulated tumor trees more accurately than either data type alone when no SNVs dropped out (normalized Robinson–Foulds error 0.109 vs 0.161 for copy number alone, with no copy-number noise), but once 20% or more of true SNVs dropped out, the same average was worse than copy number alone in 23 of 24 noise settings.

Support (all in `SUMMARY.md` and `tables.md`):
- **"More accurately than either alone" with no SNV dropout.** Finding 1: at α = 0, combining beat the better single source at all six σ.
- **0.109 vs 0.161.** α = 0, σ = 0: combined 0.109 ± 0.012 vs CNA-only 0.161 ± 0.011, a paired difference of −0.051 ± 0.010.
- **"Worse … in 23 of 24".** Finding 2: α ∈ {0.2, 0.3, 0.4, 0.5} × 6 σ = 24 settings. In 23 of them combining was worse by at least 2 paired SE; the 24th (α = 0.2, σ = 0) showed no clear difference.
