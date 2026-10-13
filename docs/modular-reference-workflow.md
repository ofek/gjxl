# Native reference-style Modular workflow

The private CPU path now accepts original packed integer pixels, selects
palette/RCT/WP decisions, prepares and fits each selected learning scope once,
and emits a complete file through direct estimated ANS with a 64-cluster cap.
No captured reference decision enters this path. The public encoder still
uses its existing policy; this is integration and qualification, not default
promotion.

On the existing **16-input corpus**, Best produces **8,544,487 bytes**, versus
stock e9's **8,511,250**: **+0.3905%**. This is **1.6538% smaller** than the
qualified legacy native total of 8,688,169 bytes. The all-predictor control
produces **8,386,273 bytes**, **1.4684% smaller than e9** and **3.4748% smaller
than that native baseline**. These are ratios of summed complete-file bytes,
including every input. All files decode to the original integers, including
invisible RGB and 16-bit alpha.

The [qualification ledger](modular-reference-workflow-results.json) records
provenance and per-image results. Runs overlapped correctness work and are
**not timing measurements**. Passing this corpus's aggregate rate target does
not establish per-image, held-out, search, or performance gates.

The subsequent [integrated timing study](modular-reference-workflow-timing.md)
qualifies the whole-learning and complete-encode intervals separately.

## Selection, learning, and ownership

`PrepareReferenceModularFrame` owns the unpacked samples, selected transformed
planes, and learned trees. Its view can be encoded without selecting or
training again. `EncodeReferenceModularImage` is the complete private wrapper.
Both preserve previous output on failure; the wrapper publishes statistics
only with a successful completed file.

The supported ordinary lossless profile follows pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`:

- Use 256-pixel groups. Automatic policy selects local trees above eight
  groups, skips global palettes, and selects palette/RCT/WP per nonempty
  group. Buffered policy retains global palettes and one shared tree.
- A single-group frame performs global palette selection then RCT/WP,
  without a second local palette pass.
- Preserve canonical stream IDs and empty slots during shared-tree
  preparation. Retain excluded global channel descriptors: they still affect
  the reference learning problem even when their pixels are not scanned.
- Local learning uses coordinates `[0,1)`, nominal 256-square image dimensions,
  and clipped edge-channel planes. Initial global descriptors come from the
  first 2048-pixel DC chunk.
- Prepare each scope once and fit each nonempty sample set once. Empty global
  calls return the reference trivial Weighted tree without invoking the
  fitter. Count calls and actual fits separately. There is no candidate
  refitting or hidden legacy fallback.
- Best uses Weighted and Gradient in reference order; the all-predictor
  control uses the reference Variable order. Sampling, quantization,
  thresholds and growth come from the qualified native preparation/fitter.
  Finished trees retain only live nodes, without pruning or refitting.

The input contract is the existing packed Gray/RGB/unassociated-RGBA 8/16-bit
profile. Patches, responsive transforms, arbitrary extra channels, and custom
frame settings are outside it. Admission rejects instead of truncating:
at most 2^28 original samples, 16M training samples per scope, representable
group IDs, and the existing image-dependent tree/depth limits. Results own
their samples independently of packed-input lifetime; moves preserve the
addresses behind their views.

“Automatic” describes coding policy, not bounded-memory streaming. This
implementation retains all selected streams before coding. Incremental
encoding can later improve that memory schedule without changing the learner.

## Original-input reference parity

Fifteen automatic-policy inputs and five forced-buffered controls match stock
e9 in transforms, WP presets/coefficients, channel shapes, included transformed
pixels, learning ranges, and every raw learned tree node. These **20 comparisons**
cover **208 learning calls, 204 fits, 101,788 tree nodes, and 77,475,455 directly
compared sample values**. The four extra calls are empty global scopes.
Coding events are matched by stream ID because libjxl visits DC chunks in a
different order. Metadata roles normalize to reference `-1` shifts; component
labels, unused by learning, are outside the comparison.

The original ten auto/buffered pilot files are also byte-identical to the
earlier direct estimated ANS64 files produced from captured reference
decisions. Selection and learning are now native. This is native-to-native
byte identity, not stock libjxl bitstream identity.

`gray-patches` is a parity exclusion, **not a rate exclusion**. Stock e9 emits
a 70×63 auxiliary patch-reference frame and learns on patch-subtracted main
pixels. Its capture has two global stream-0 events and two learning scopes;
the native workflow has one frame. The verifier rejects that mismatch and
the rate study explicitly records the unsupported feature. It neither drops
an event nor disables stock patches to claim a match.

## Complete-file rate

| Input | Stock e9 | Native Best | Native all | Best vs e9 |
|---|---:|---:|---:|---:|
| photo-macan | 252,153 | 252,179 | 248,954 | +0.010% |
| photo-ria | 208,131 | 208,127 | 207,372 | -0.002% |
| photo-bliznaca | 240,347 | 240,327 | 239,978 | -0.008% |
| photo-flower | 2,537,909 | 2,531,609 | 2,531,402 | -0.248% |
| gray-macan | 90,153 | 90,129 | 89,205 | -0.027% |
| alpha-ria | 228,013 | 228,118 | 226,175 | +0.046% |
| alpha-flower | 3,715,769 | 3,706,105 | 3,708,977 | -0.260% |
| chessboard | 136,329 | 131,233 | 59,393 | -3.738% |
| splines | 258,951 | 257,365 | 243,859 | -0.612% |
| gray-patches | 4,912 | 21,591 | 15,369 | +339.556% |
| palette-358 | 1,243 | 1,241 | 975 | -0.161% |
| gray16-ramp | 2,502 | 2,484 | 1,105 | -0.719% |
| screen-pattern | 343 | 329 | 333 | -4.082% |
| palette-pattern | 1,047 | 1,036 | 573 | -1.051% |
| noise16 | 825,073 | 825,067 | 791,211 | -0.001% |
| alpha16-ramp | 8,375 | 47,547 | 21,392 | +467.725% |
| **Total** | **8,511,250** | **8,544,487** | **8,386,273** | **+0.391%** |

The alpha16 ramp has exact learning parity, so its residual rate difference
is downstream of that matched tree. The existing
[entropy study](modular-prepared-entropy.md) records the missing LZ77 path.
The patch case changes frame construction and learning inputs themselves.
Neither outlier supports enlarging or repeatedly fitting the matched Best
trees as a general remedy.

Best still regresses individual legacy-native files, including chessboard
(131,233 versus 62,805), palette-pattern (1,036 versus 438), and alpha16-ramp
(47,547 versus 19,866). All predictors recover much of that loss but are not
uniformly better; their whole-stage cost is not qualified here. No runtime
Best/all or old/new policy selector has been enabled.

## Resources and correctness

Each allocating stage gets a nonblocking reservation from its existing
planner while earlier retained backing remains charged to the same budget.
Busy capacity returns `Unavailable`; a request exceeding the domain limit
returns `OutOfMemory`. The workflow never waits for admission while holding
earlier buffers. This is staged admission, not a new monolithic shape-only
bound or an RSS limit.

Across the 16 automatic runs, maximum managed backing was 310,601,848 bytes
for Best and 310,868,332 for all. Maximum committed capacity, including unused
conservative reservations, was 624,136,767 and 624,606,891 bytes. Packed input,
reference decode/trace storage, allocator bookkeeping, and fixed stack state
are excluded. Callers must also account for existing outputs.

Release and ASan/UBSan each pass 19 targeted CTests. New coverage includes:

- 98 exact files: six formats, both byte orders, padding, invisible color,
  Best/all, auto/buffered, eight/nine-group boundaries, and clipped 2D groups.
- Frame moves, detached input lifetime, wrapper input aliasing, invalid
  options, sample caps, tiny/busy budgets, and atomic publication.
- 5,138 injected managed-allocation failures across preparation and complete
  encoding, including global/local scopes, with no leaked managed capacity.

All 2,356 prior file controls remain unchanged. The 98 new files match across
builds, giving 2,454 compared generated files per build. Automatic photo-flower,
buffered chessboard, and automatic alpha16-ramp also pass under ASan/UBSan
with identical decisions, samples and files. Nine corruption probes are
rejected by the comparison tool. Linked libjxl archives are uninstrumented;
reference vptr checks and macOS LeakSanitizer are excluded. Explicit managed
backing checks remain active.

## Provenance and remaining gates

`build/reference-workflow-validation` binds 6,945 APFS-cloned artifacts,
including source, binaries/compiler settings, inputs, reference traces,
generated files, logs, and the capture script. Manifest SHA-256:
`5139f66910170d947dda811394970a87136fa912cb0fb4314e11fdbf249f5694`.
The source snapshot is authoritative for these pre-commit builds. Partial
corpus runs remain preserved; the final study reuses completed captures only
from identical frozen executables and rechecks their decisions.

Recheck without encoding:

```sh
python3 tools/modular_reference_learning/analyze_reference_workflow.py \
  --validation build/reference-workflow-validation \
  --baseline-validation build/reference-selection-validation-v2 \
  --output /tmp/gjxl-reference-workflow-rechecked.json
```

`gjxl_modular_reference_workflow_check` and
`tools/modular_reference_learning/run_reference_workflow_study.py` reproduce
original-input encodes. The driver uses the pinned test oracle only for
independent decoding; production libraries have no libjxl dependency.

The subsequent timing study meets the current-corpus whole-learning aim for
Best. Next compare selection with exhaustive search using the **new learner**,
retain Best/all controls, and apply held-out and per-image gates before
promotion. Patch-frame and high-depth entropy differences remain explicit
limitations. The
[search plan](modular-reference-search-policy.md) governs those separate
decisions and subsequent GPU prioritization.

The subsequent [conditional WP ablation](modular-reference-wp-sweep.md) finds
a 0.3773% aggregate selected-policy loss against its limited candidate set,
mostly on the 16-bit alpha ramp. It establishes a search-gate failure and a
specific held-out validation target; it does not enable repeated fitting or
change the qualified workflow above.
