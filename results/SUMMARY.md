# Summary

- 40-cell simulated trees, 20 paired replicates per setting of SNV dropout α × CNA jitter σ (6 levels each; false-positive rate 0.01).
- Rebuilt by neighbor joining and scored by normalized Robinson–Foulds distance (nRF; 0 = exact tree).
- Values are means ± 1 SE. Differences are paired; under 2 SE counts as "no clear difference."

## Findings

1. **Combining helped only without SNV dropout.** At α = 0, combining (w = 0.5) beat the better single source at all six σ. At σ = 0 it scored 0.109 ± 0.012 vs CNA-only 0.161 ± 0.011 (−0.051 ± 0.010); at σ = 1.0 its lead over SNV-only was −0.028 ± 0.008.
2. **From α = 0.2, combining hurt.** It was worse than CNA-only in 23 of 24 settings, by up to +0.357 ± 0.036 (α = 0.5, σ = 0.8). At α = 0.1 it was worse only at σ = 0.4 and 0.6, both borderline (+0.039 ± 0.019, +0.049 ± 0.019).
3. **The two noise types differed in strength.** SNV-only rose from 0.178 ± 0.014 (α = 0) to 0.711 ± 0.026 (α = 0.5). CNA-only stayed within 0.158–0.176 up to σ = 0.6 and reached 0.266 ± 0.013 at σ = 1.0.
4. **Tuning the weight didn't help.** Chosen on replicates 0–9 and scored on 10–19, the tuned weight beat the better single source only at α = 0 (5 of 6 σ). For α ≥ 0.3 it picked CNA-only every time.

## Limitations

- Naive distance averaging; SNVs and CNAs evolve independently.
- No doublets, missing data or boundary errors.
- Neighbor joining only, on 40 cells.
- The zero-noise calibration (5000 bins, μ_CNA = 8) sets each source's noise sensitivity.
