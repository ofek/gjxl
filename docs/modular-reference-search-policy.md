# Reference Modular search and training scope

The next performance milestone should reproduce the reference's **choice of
what to learn**, after qualifying the cost of native preparation and fitting.
The [matched pilot](modular-native-learning-timing.md) finds no remaining large
cost gap in one controlled learning call. It does not establish that one global
Best tree on our frozen transforms reproduces stock e9's compression policy.

This began as a source audit and a plan for the next controls.
The pristine reference checkout was verified at
`e8ff09762481785938d8e4e01333ed3917571161`. The saved 16-input capture ledger
corroborates its fit counts. The subsequent
[50-encode buffering ablation](modular-reference-policy-ablation.md) now captures
actual stream pixels/options and quantifies the global/local workflow tradeoff.
It confirms that per-stream WP/RCT settings matter even with a global tree, and
that larger palettes are required on the structured cases. The immediate native
preparation gap is now closed by
[per-stream WP preparation and actual-stream replay](modular-native-stream-preparation.md).
The [per-stream coding contract](modular-prepared-streams.md) now reproduces
captured e9 tree/sample tokens and exact decodes with local/shared model scopes.
Its conditional rate gap still needs downstream ablations; reference palette
construction and cheap selection remain unported. A blanket global-tree
replacement is not justified.

## Selection happens before expensive tree fitting

In pinned [enc_modular.cc](https://github.com/libjxl/libjxl/blob/e8ff09762481785938d8e4e01333ed3917571161/lib/jxl/enc_modular.cc),
`try_palettes` / `maybe_do_transform` evaluate palette proposals with
`EstimateCost` at e9 and undo a successful transform if its estimate is worse.
The proposals include all-channel palettes, RGB palettes with separate alpha,
and single-channel compaction, subject to color-count heuristics.

`PrepareStreamParams` also tries local palettes. For eligible lossless color
streams, e9 ranks **19 RCTs** with `EstimateCost`, applies the selected RCT, then
tries **five weighted-predictor modes** with `EstimateWPCost`. Neither decision
requires a learned MA tree for every transform candidate. RCT and WP winners
use strict improvement in the defined candidate order.

The [fixed-predictor estimator](https://github.com/libjxl/libjxl/blob/e8ff09762481785938d8e4e01333ed3917571161/lib/jxl/enc_modular_simd.cc)
uses clamped-gradient residuals and contextual token histograms, with entropy
plus extra-bit cost. `EstimateWPCost` uses WP residuals and 34 contexts derived
from its predictor property. These use the default 4/2/0 HybridUint proxy,
separately from the 4/1/2 training proxy. The SIMD estimator must be studied and
qualified directly; a superficially similar scalar entropy calculation is not
a reference-equivalent ranking oracle.

GJXL's current `RankModularRcts` uses bounded samples and Gradient/Left/Top
profiles; it is not the reference estimator. Its existing ranking should not
be treated as validated for the new learner. The legacy search's repeated full
fits remain a separate source of excess work, even with a fast fitting kernel.

## Large images can use local group trees

The pinned [frame encoder](https://github.com/libjxl/libjxl/blob/e8ff09762481785938d8e4e01333ed3917571161/lib/jxl/enc_frame.cc)
permits streaming for ordinary lossless e9 when there are more than eight
groups and the other eligibility checks pass. `ComputeEncodingData` skips
global `ComputeTree` / `ComputeTokens` in streaming mode. With no global tokens,
`EncodeStream` calls
[`ModularGenericCompress`](https://github.com/libjxl/libjxl/blob/e8ff09762481785938d8e4e01333ed3917571161/lib/jxl/modular/encoding/enc_encoding.cc),
which learns a tree for that individual stream. Ordinary nonstreaming lossless
e9 instead takes the global-tree path. Metadata and other special learning
events must still be counted separately.

The existing stock-reference captures agree with this scope difference:

| Input | Dimensions | 256-pixel groups | Captured reference fits |
|---|---:|---:|---:|
| photo-macan | 500×500 | 4 | 1 |
| photo-flower | 2268×1512 | 54 | 54 |
| alpha-flower | 2268×1512 | 54 | 54 |
| chessboard | 1024×1024 | 16 | 16 |
| splines | 2048×2048 | 64 | 64 |

By comparison, the current bridge combines the native canonical stream images
into one global learning call. The reference large-image controls therefore
do not fit the same learning problem as those bridge cases. More fit events
can still mean less expensive learning when each problem is much smaller.
These observations establish a policy difference, not its isolated rate or
speed effect.

The established whole-image path emits global-tree stream headers. The new
private prepared-stream path also supports local trees/models and ordered
per-stream transform/WP headers, qualified against actual captures. Automatic
policy selection has not been connected. Splitting a global fit into smaller
fits and merging the trees would not reproduce local tree/entropy coding.

## Next controls and implementation order

1. Capture actual e9 stream transforms, WP settings and global/local tree
   scope, alongside samples, fit counts, bytes and preparation/fitting times.
   Keep the current frozen-native-transform control as a separate experiment.
   An automatic-streaming versus forced-buffered e9 comparison is a useful
   workflow ablation, but changes beyond tree scope must be recorded rather
   than attributing its entire result to learning.
2. On identical reference-transformed streams, compare global and local
   learning/coding, retaining each stream's predictor parameters. Measure rate
   as complete files, total preparation plus all fits, tree/header cost and
   memory. Preserve exact reconstruction and distinguish a pure scope change
   from per-group palette/RCT/WP changes. Use the reference implementation first.
3. Port the supported selection/scope behavior that those controls qualify.
   Match reference estimator values, ranking/ties, selected transforms and
   custom-WP samples before evaluating any native approximation. Retain Best
   and all-predictor controls. The current all-predictor bridge is within 1%
   aggregate e9 bytes, while Best alone is not; a blanket two-predictor switch
   is not yet qualified.
4. Compare the resulting candidate policy against exhaustive search with the
   **new** learner. Count one fit per selected coding scope, including local
   groups and auxiliary events; separately count repeated candidate fits.
   Report per-image rate losses, shortlist recall if a shortlist is used,
   complete learning/encode CPU and wall time, and peak storage. A hidden legacy
   fallback must not mask the candidate's cost or rate.
5. Apply the existing rate/resource/held-out gates before default promotion,
   then profile the resulting pipeline to choose the first GPU learning task.

The proposed targets remain: ≤1.5× reference CPU per identical fit, ≤+0.2%
aggregate and ≤+1% per-image bytes for a search approximation versus exhaustive
new-learner search, a whole-learning-stage aim of ≤2× stock e9, and eventual
aggregate rate within +1% of e9 without regressing the qualified native total.
These are separate gates. Core learner parity does not satisfy the search,
whole-stage, held-out or rate gates by itself.
