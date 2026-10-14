# Adaptive Modular complete-search hybrid pilot

The hybrid search preserves the complete CPU files and winning policies on all three full-resolution inputs. Every candidate used Metal token evaluation; no CPU fallback occurred. The photographic, palette and 16-bit-alpha smoke captures also match the saved CPU files.

**Timing scope:** one fresh CPU/hybrid pair per image on a busy desktop, with alternating pair order. These are diagnostic observations, not repeated throughput qualification. Both routes use one CPU participant, the original 262144-sample/256-leaf learner, retained baseline candidates and checkpoint selection. Backend initialization is outside the encoder-call timer and is reported separately. CPU transforms, weighted-feature preparation, cache creation, every candidate encode and output publication are inside it.

Frozen implementation: `bbf86a037485df92ada815a449f1c41de2022083`. Artifacts: `/Users/yunhocho/GitHub/gjxl-modular-adaptive-gpu/build/adaptive-hybrid-pilot-20261011`. Inputs are the original 4000×3000 RGB8 PAMs used by the CPU baseline.

| Image | Encoded bytes | CPU seconds | Hybrid seconds | Observed CPU/hybrid ratio | Feature builds / reuses |
|---|---:|---:|---:|---:|---:|
| 1029 | 10,821,744 | 204.253 | 137.383 | 1.49× | 24 / 46 |
| 1039 | 7,230,909 | 193.706 | 129.631 | 1.49× | 24 / 45 |
| 1049 | 7,336,102 | 191.649 | 129.435 | 1.48× | 24 / 44 |

The aggregate observed time ratio is 1.49×. The paired outputs, full selected policies and training work counters match exactly. Each output is independently decoded by pinned libjxl before capture succeeds.

## Attribution

| Caller stage | CPU total seconds | Hybrid total seconds |
|---|---:|---:|
| input | 3.627 | 1.735 |
| transforms | 15.963 | 8.673 |
| training | 111.562 | 112.406 |
| tokens | 231.094 | 47.983 |
| model | 184.000 | 181.901 |
| emission | 43.050 | 43.331 |
| assembly | 0.238 | 0.222 |

Within hybrid `tokens`, feature preparation accounts for 34.230 seconds and token generation/handoff for 12.860 seconds. These are nested intervals; do not add them again to the table. Feature preparation currently computes weighted history on CPU. The token interval includes dispatch setup, GPU wait and host handoff, not just device execution. Training and entropy modeling remain CPU work.

Exact ANS token-cost simulation alone takes 109.474 seconds (27.6% of hybrid complete-call time), nested inside `model`. It is the next implementation target: preserve each section's reverse-order rANS recurrence while processing independent sections and candidate models in parallel. Leave model selection and ties unchanged.

## Resource tradeoff

| Image | CPU peak backing MiB | Hybrid peak backing MiB | CPU admission MiB | Hybrid admission MiB | Backend setup seconds |
|---|---:|---:|---:|---:|---:|
| 1029 | 689.3 | 1307.3 | 5021.1 | 6394.7 | 0.039 |
| 1039 | 675.7 | 1293.7 | 5021.1 | 6394.7 | 0.032 |
| 1049 | 674.9 | 1292.9 | 5021.1 | 6394.7 | 0.041 |

These are managed backing charges and conservative admitted capacities, not process RSS or GPU-only memory. Retained transformed planes and weighted maps increase the hybrid peak; each call fits its plan and releases its domain before returning.

## Compression and limits

Summed output is 25,388,755 bytes: 1.20% above saved default libjxl e7 and 2.18% above e9. These are the unchanged CPU-baseline rates; GPU evaluation neither improves nor weakens the coding policy. The reference sizes come from verified saved files, not new libjxl timings. This pilot does not establish an end-to-end speedup over libjxl.

The captured process snapshots show background desktop activity. A single pair cannot separate scheduling variation from smaller stage changes. Use the large phase trends to prioritize work, then repeat paired captures before claiming throughput. Resource peaks, conservative planned capacities, setup times and complete nested counters are retained in the companion JSON. Public encoder defaults are unchanged.

Reproduce the summary from the retained capture and CPU baseline:

```sh
python3 tools/modular_training_study/analyze_adaptive_hybrid.py \
  --capture build/adaptive-hybrid-pilot-20261011 \
  --baseline build/adaptive-baseline-20261011 \
  --json docs/gpu-adaptive-modular-hybrid-results.json \
  --markdown docs/gpu-adaptive-modular-hybrid.md
```
