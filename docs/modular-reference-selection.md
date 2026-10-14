# Native reference palette, RCT and WP selection

The private CPU selector now reproduces pinned libjxl's supported lossless e9
selection sequence: palette proposals and cost acceptance, followed by the
19 RCT candidates and five weighted-predictor presets. It performs **no tree
fits and no ANS candidate coding**. This is the selection stage needed before
connecting native preparation and one fit per selected coding scope.

The [qualification ledger](modular-reference-selection-results.json) records
exact costs, decisions, transformed pixels, allocation behavior and complete
file decoding. This milestone does not connect the selector to the public
encoder's search or choose automatic frame/group scope. The current encoder's
rate and timing are therefore unchanged by this private addition.

## Policy and ownership

`ReferenceSelectionOptions` distinguishes the global 95% and local 80%
channel-compaction passes and permits a pass without palettes. This supports
global palettes followed directly by RCT/WP for a single-group frame, or global
palettes followed by local palette/RCT/WP selection on separate group views.
The caller still decides which sequence and coding scopes to use.

The native sequence preserves the reference's less obvious behavior:

- Try all-channel palettes, then an all-minus-one proposal for four channels
  if the first proposal was not accepted. The multichannel limit uses the
  original cost/pixel heuristic and the requested cap, with ordered or
  first-occurrence construction.
- Accept a constructed palette when its gradient estimate is **no greater**
  than the baseline, including equality. Rejected proposals leave the input
  unchanged.
- Compact remaining channels with the appropriate percentage and pixel-count
  limit. Their cost baseline is shared; it is not updated after each accepted
  single-channel palette. The tests include a second compaction that slightly
  increases the current estimate but is still accepted against that baseline.
- Use channel zero's initial population for the palette heuristics, including
  when that channel is metadata. Keep the same channel coordinates as metadata
  planes are inserted.
- Rank the same 19 RCTs in reference order, with strict improvement and identity
  first. Estimate the whole image, including unchanged metadata/alpha planes.
  Apply the winner, then rank the five WP presets with strict improvement and
  preset zero first.

The selector borrows source planes until a transform is accepted. RCT trials
reuse three scratch planes and always read the same pre-RCT input. The winner
is applied in place when the selector already owns an image; otherwise it
creates one final owned image. RCT scratch is released before WP estimation.
Identity selection ordinarily returns no owned image. Exact aliasing of the
previous output is supported and retains ownership on identity; partial
overlap is rejected. Errors preserve the entire previous result.

The private input contract accepts 1–16-bit frame depths, up to four normal
planes, strided views and empty palette metadata. Normal samples must be in
`[0,65535]` before RCT; this includes indices from an earlier palette pass.
Metadata may be signed. Total input population is at most `INT32_MAX`, and
the first three normal planes must have matching geometry/shifts when RCT
selection is enabled. Unsupported inputs fail explicitly. This contract does
not expand the public image-format API.

The shape-only planner bounds newly owned images, overlapping palette
candidates, three RCT scratch planes and one channel's WP history. Admitted
single-channel values fit the dense palette bitmap, so selection needs no
sparse dictionary allocation. All newly allocated backing belongs to
`kPreparation`. Borrowed input, previous output backing, allocator overhead and
fixed stack objects are excluded; the bound is not an RSS limit.

## Qualification

The oracle uses pristine libjxl
`e8ff09762481785938d8e4e01333ed3917571161`. CMake checks the SHA-256 of
`enc_modular.cc`, extracts `do_transform`, `maybe_do_transform`, `try_palettes`
and the RCT/WP decision block verbatim, and compiles them in a test-only
namespace. Cost wrappers record returned values without changing decisions.
The production library neither compiles that extracted code nor links libjxl.
Exact estimator comparisons use the previously qualified four-lane Apple
AArch64 reference arithmetic.

Both Release and ASan/UBSan pass **17 selected CTest entries**:

| Check | Result per build |
|---|---:|
| Matched generated selection cases | 496 |
| RCT candidate costs compared | 3,914 |
| WP candidate costs compared | 2,475 |
| Constructed / accepted palette proposals | 205 / 180 |
| Constructed palettes rejected by cost | 25 |
| Explicit palette tie / shared-baseline controls | 1 / 1 |
| Selected single-channel palettes wider than 1,024 | 4 |
| Identity results retaining borrowed input | 237 |
| Injected allocation failures | 799 |
| Maximum observed generated selection managed peak | 1,106,256 bytes |
| New complete files decoded exactly | 216 |
| Prior files unchanged | 884 / 884 |
| Files identical across Release and sanitizer builds | 1,100 |

Every generated case compares the complete gradient/WP cost-call sequence,
selected transform descriptors, WP winner and final pixels. All five WP
presets win on at least one case. The controls include Gray/two-plane/RGB/RGBA
inputs, 1/8/16-bit depths, empty/nonempty metadata, strided input, shifted alpha,
ordered/unordered palettes, disabled stages, wide accepted/rejected palettes,
prescribed forward/inverse reconstruction, and exact input/output aliasing.
Allocation failures are swept through first success, checking atomic output,
class ownership and release of all temporary managed backing.

The 216 new files cover 36 Gray/RGB/RGBA 8/16-bit cases with global selection
or global palettes plus local selection. They exercise shared/local trees
and Prefix, ANS-from-Prefix and estimated direct ANS. These controls use a
**prescribed weighted tree**, so they validate selected transforms/WP headers
and decoding without attributing rate or timing to a learner. Their additional
per-stream decisions are also compared with the extracted reference code.

All four generated-decision ledgers (native/oracle × Release/sanitizer) are
identical. Earlier palette and prepared-stream files retain their bytes.
Native code and test adapters are sanitized; linked reference archives remain
uninstrumented, reference vptr checks are excluded, and macOS LeakSanitizer is
disabled. Managed backing leak checks pass. Correctness runs overlap and are
not performance measurements.

## Reproduction and next integration

The tests are `modular_reference_selection` and
`modular_reference_selection_oracle`. The latter requires the pinned reference
build. The validation archive binds 5,546 artifacts, including source,
generated oracle code, binaries, build commands, test logs, decision ledgers
and files. Recheck it with a fresh output path:

```sh
python3 tools/modular_reference_learning/analyze_reference_selection.py \
  --validation build/reference-selection-validation-v2 \
  --baseline-validation build/channel-palette-validation \
  --output /tmp/gjxl-reference-selection-rechecked.json
```

`build/reference-selection-validation-v2/freeze.py` records the capture.
The first incomplete freeze is preserved separately; the second includes the
exported compiler commands. Exporting that metadata changed no tested binary.
The pre-commit source snapshot is authoritative.

The subsequent [native workflow integration](modular-reference-workflow.md)
connects this selector to automatic/buffered scope, native preparation/fitting,
and complete coding. It qualifies decisions from original inputs and reports
16-input Best/all rate controls. The exhaustive new-learner, held-out,
whole-stage timing and promotion gates in the
[search plan](modular-reference-search-policy.md) remain open. Patch-frame
support and the high-depth LZ77 exception remain explicit limitations.
