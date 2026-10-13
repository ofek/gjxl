# Conditional WP search with the native reference learner

The reference-style WP selector leaves a measurable rate opportunity on the
current corpus, concentrated in the 16-bit alpha ramp. Keeping the selected
palette/RCT transforms, tree scope, Best learner and entropy settings fixed,
the smallest tested complete files total **8,512,368 bytes**, versus the
selected policy's **8,544,487** and stock e9's **8,511,250**. The tested minimum
is **+0.0131% versus e9**. This is an offline search result, not a new encoder
policy or a demonstrated fast way to obtain those bytes.

The selected policy is **0.3773% larger** than that tested minimum, already
exceeding the proposed +0.2% aggregate search-loss target. Four files exceed
the +1% per-image target. This limited control can disprove those bounds for
an exhaustive domain containing its candidates; it cannot establish the
exhaustive optimum. No production source or encoder default changes here.

The [machine-readable ledger](modular-reference-wp-sweep-results.json) retains
every candidate's bytes, learning counts and diagnostic timing, alongside
the frozen study and validation hashes.

## Candidate domain and controls

For every original input, first prepare and encode the qualified automatic
Best workflow. Then replay its selected WP vector through fresh native sample
preparation and fitting, requiring identical raw trees, learning counts and
complete bytes. Finally, try five additional assignments: apply one WP preset
uniformly to every stream on which the reference selector evaluated WP.
Streams without WP selection retain their original parameters.

Each assignment refits every original learning scope once. Local scopes stay
local; shared-tree cases retain all original stream IDs, holes, excluded
descriptors, per-stream parameters and the original global leaf bound. The
transformed pixel planes are borrowed unchanged. WP-dependent predictions and
properties, and therefore the resulting learned trees, can change. Every
completed file is independently decoded by pinned libjxl and compared to all
source integers, including invisible color and 16-bit alpha.

The candidate set contains the selected vector plus all five uniform vectors.
It does **not** enumerate the Cartesian product of independent stream choices,
RCTs, palette decisions, buffering policies or predictor sets. Buffered group
choices interact through their shared tree and entropy model. Even local
group choices cannot be declared exhaustively qualified from this uniform
sweep alone. Best always means the same two-predictor learner; the expensive
all-predictor variant is not mixed into these minima.

If `B` is the selected file size, `M` the minimum tested size, and `E` the
exhaustive optimum over a containing domain, then `E <= M <= B`. Therefore
`B/M - 1` is a **lower bound** on the exhaustive search loss. A failure of the
target is decisive; passing this subset would leave the target unqualified.
Minima use complete native files, including tree, transform and entropy
headers, with earlier candidates winning byte ties.

## Complete-file results

Loss below uses the tested minimum as denominator. Absolute byte deltas are
retained to distinguish tiny-file percentages from material corpus savings.

| Input | Selected | Tested minimum | Winner | Saved bytes | Selected loss |
|---|---:|---:|---|---:|---:|
| photo-macan | 252,179 | 252,179 | selected | 0 | 0.000% |
| photo-ria | 208,127 | 208,127 | selected | 0 | 0.000% |
| photo-bliznaca | 240,327 | 240,000 | WP 2 | 327 | 0.136% |
| photo-flower | 2,531,609 | 2,531,609 | selected | 0 | 0.000% |
| gray-macan | 90,129 | 90,108 | WP 4 | 21 | 0.023% |
| alpha-ria | 228,118 | 228,118 | selected | 0 | 0.000% |
| alpha-flower | 3,706,105 | 3,706,105 | selected | 0 | 0.000% |
| chessboard | 131,233 | 131,233 | selected | 0 | 0.000% |
| splines | 257,365 | 256,412 | WP 1 | 953 | 0.372% |
| gray-patches | 21,591 | 21,096 | WP 1 | 495 | 2.346% |
| palette-358 | 1,241 | 1,232 | WP 0 | 9 | 0.731% |
| gray16-ramp | 2,484 | 2,361 | WP 1 | 123 | 5.210% |
| screen-pattern | 329 | 329 | selected | 0 | 0.000% |
| palette-pattern | 1,036 | 993 | WP 0 | 43 | 4.330% |
| noise16 | 825,067 | 825,067 | selected | 0 | 0.000% |
| alpha16-ramp | 47,547 | 17,399 | WP 1 | 30,148 | 173.274% |
| **Total** | **8,544,487** | **8,512,368** | | **32,119** | **0.377%** |

The alpha ramp accounts for **93.86% of the savings**. Excluding it only for
concentration analysis, the other 15 inputs save 1,971 bytes and selected-policy
loss is 0.0232%. All 16 remain in the reported qualification totals.

On the alpha ramp, the reference proxy chooses WP 3 on both large full groups.
Summed over eligible streams, the selected estimates cost 831,413.746 bits;
uniform WP 1 costs 866,196.902, **4.18% worse by that proxy**. After fitting,
the global tree has 185 nodes instead of 477, and the complete file is
**63.41% smaller**. The fixed-WP estimator does not evaluate the eventual
Weighted/Gradient tree and native entropy model. The measurements establish
a ranking inversion; they do not separately attribute the saving to tree
headers, residuals or context grouping. A very narrow proxy-tie rule would
miss this opportunity.

The earlier stock-tree/native-entropy control still establishes a downstream
gap for the *matched reference decisions*, including missing LZ77. This new
control changes WP and refits, so it does not contradict that conditional
result. It does show that some of the native gap can be reduced upstream
without adding LZ77. Even this alpha-ramp winner remains above e9's 8,375 bytes.
The patch-frame exception also remains: its minimum is 21,096 versus e9's
4,912. Neither feature difference is solved by this sweep.

## Validation and provenance

The study produces **112 independently decoded files**: baseline, selected
replay and five preset files for each of 16 inputs. Every baseline reproduces
the previously qualified Best file. Selected replay reproduces **204 learning
calls, 200 fits and 79,424 raw tree nodes**. The six replay/trial assignments
perform another 1,200 fits; the total research run performs 1,400. These counts
are deliberately visible and are not a proposed production search budget.

The new harness also passes Release and ASan/UBSan checks on six pixel formats,
single/shared/local tree scopes, clipped nine-group geometry, hidden RGB,
Best/all controls, deterministic repetition and overwrite rejection. All
**56 generated files per build** match across builds. Eight analyzer corruption
probes reject altered output, byte counts, decode status, missing/duplicate WP
candidates, assignment counts, learning coverage, tree scope and artifact
inventory. The linked reference archives are uninstrumented; reference vptr
checks and macOS LeakSanitizer are disabled. Explicit managed-capacity release
checks remain active.

Managed backing peaks at 315,473,133 bytes and committed capacity at
629,326,600 in this harness. These include the retained baseline frame/output
and trial state, exclude packed source/decode/IO storage and allocator overhead,
and are neither a production peak nor an RSS bound. Per-candidate CPU/wall
fields are single-pass diagnostics, without warmups or counterbalancing.
The [qualified integrated timing](modular-reference-workflow-timing.md)
remains the performance result for the unchanged production-private workflow.

`build/reference-wp-sweep-study` freezes 772 artifacts: source, executable, build settings,
reference libraries, all 16 original inputs and qualified controls, complete
files, logs and results. The study source base is `1efa40d`; its pre-commit
source snapshot is authoritative. Its manifest SHA-256 is
`43c28f9a2db7e43c128154009e3ad269c8096f0ba4e5768628cb77902914d5fa`.
The 174-artifact validation supplement freezes both final test builds/results,
the final analyzer and its eight rejection probes:
`build/reference-wp-sweep-validation-v2`, manifest SHA-256
`c2cececa10acdc572a1696868f2d1dd1555d5edeadce618c62110d6d79b2ab98`.
The final Release executable is byte-identical to the frozen study executable.
The ledger binds both manifests by SHA-256. All 390 production source/header
files were also checked unchanged against the integrated timing snapshot.

Recompute without encoding:

```sh
python3 tools/modular_reference_learning/analyze_reference_wp_sweep.py \
  build/reference-wp-sweep-study \
  --validation build/reference-wp-sweep-validation-v2 \
  --out /tmp/gjxl-reference-wp-sweep.json
```

## Consequences for the remaining plan

The efficient reference-style learner remains qualified; this finding concerns
its surrounding selection policy. The strict search-loss targets are now known
to fail on the original corpus even within this limited WP domain. More
exhaustive search cannot turn that failure into a pass.

Before designing a fallback, test this WP opportunity on held-out high-depth
gradients, natural images, RGB/RGBA and structured content. Preserve the
current selected policy as the runtime control and freeze any proposed trigger
before its qualification set is measured. If an additional candidate is
justified, cap it explicitly and measure complete learning/encode cost; the
five-preset research search is not a deployment recommendation. Broader
RCT/palette controls and per-image legacy regressions remain open. This result
does not qualify a blanket preset switch, all-predictor switch, default
promotion or GPU implementation policy.
