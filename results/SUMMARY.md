# Summary

- 40-cell simulated trees: 20 replicates per setting of SNV dropout α and CNA jitter σ (6 levels each; false-positive rate 0.01).
- The same 20 trees are reused across settings.
- Error: normalized Robinson–Foulds distance after neighbor joining (nRF; 0=exact).
- Means ±1 SE. Paired differences under 2 SE count as no clear difference.

## Findings

1. **Without SNV dropout, combining helped.** At α=0, the w=0.5 average beat the better single source at all six σ, by 0.028–0.051. At σ=0 it scored 0.109±0.012 vs CNA-only 0.161±0.011.
2. **From α=0.2, combining hurt.** It was worse than CNA-only in 23 of 24 settings, by up to 0.357±0.036. Even at σ=0 it was worse for α≥0.3 (by 0.050–0.091). α=0.1 was mixed.
3. **Normalization tilted the mix.** Jitter inflates CNA distances, so mean-normalizing shrank CNA's effective share of the "0.5" mix to 0.27 at σ=0.4 and 0.11 at σ=1.0. This likely drives much of the harm's growth with σ.
4. **Dropout hurt SNVs far more than jitter hurt CNAs** over these grids. SNV-only went from 0.178 to 0.711; CNA-only stayed at 0.158–0.176 up to σ=0.6.
5. **Tuning w didn't rescue combining.** Chosen on replicates 0–9 and tested on 10–19, it beat the better single source only at α=0 (5 of 6 σ), and was worse once.

## Limitations

- Naive averaging only.
- SNVs and CNAs evolve independently.
- False positives occur only at truly mutated sites.
- No doublets or missing data.
- Neighbor joining only, on 40 cells.
