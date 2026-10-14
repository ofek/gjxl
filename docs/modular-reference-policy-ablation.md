# Reference e9 buffering, stream policy and learning scope

The reference controls confirm that **local learning remains valuable**, even
after the fitting implementation is efficient. Forcing the large spline image
through the buffered/global-tree path costs **2.96× learning CPU** and makes its
file 0.63% larger. The large photos trade about 49–56% more learning CPU for
1.96–2.64% smaller files. A blanket global-tree policy is therefore not justified.

The [verified ledger](modular-reference-policy-results.json) covers five inputs,
two reference configurations and **50 complete encodes**. Every decoded source
integer matches, including preserved hidden RGB. Every automatic-mode output
matches the earlier pinned stock e9 file; each timing output matches its traced
control byte-for-byte. This is a reference workflow study, not a native rate
result or a pure fitting-kernel comparison.

## Controls and capture

The reference is pristine libjxl
`e8ff09762481785938d8e4e01333ed3917571161`, lossless Modular e9 with invisible
pixels preserved and no thread pool. `auto` leaves input buffering at its
default. `buffered` changes only the public
`JXL_ENC_FRAME_SETTING_BUFFERING` setting to zero, disabling streaming. This
changes learning/coding scope and can also change selected transforms/WP modes.
It is not labelled a pure tree-scope ablation.

Each input/configuration has one trace encode, one discarded warmup and three
measured fresh-process encodes. Configurations alternate order between rounds.
The study is serial, on an uncontrolled M4 Pro desktop, without concurrent
builds or other study processes. Encoder timing covers the reference adapter
call, including the input-buffer packing it performs; source unpack, decoding
and file writes are outside that timer. Learning and fitting timers are summed
within each encode before taking medians. Trace and warmup times are excluded.

Generated hooks leave the reference checkout untouched. Successful learning
calls record their actual local/global call site, static range, multiplier
information, complete raw tree, channel descriptors, options with exact float
bits, and transformed pixels. Pixels are tightly packed little-endian signed
32-bit channel rows; excluded channels keep descriptors but no pixel payload.
Each payload is read back and compared against every original included sample.
Separate coding-stream traces retain selected transforms and WP parameters,
including streams with no sampled rows. The metadata and roughly 280 MiB of
pixel captures are bound by SHA-256 in the ignored study artifacts.

The local/global label comes from the `ModularGenericCompress` call site, not
from guessing that a one-image range means a local tree. A global tree on a
single-group image would otherwise be ambiguous. Both fit and learning-event
counts are retained: on these large streaming inputs, one additional empty
global-data stream learns a trivial tree without calling the fitter.

## Rate and complete learning cost

CPU seconds are medians of three measured encodes:

| Input | Auto bytes | Buffered bytes | Byte change | Auto learning CPU | Buffered learning CPU | Learning ratio |
|---|---:|---:|---:|---:|---:|---:|
| photo-macan | 252,153 | 252,153 | 0% | 0.4522 | 0.4496 | 0.994× |
| photo-flower | 2,537,909 | 2,470,836 | −2.6428% | 6.3682 | 9.9076 | 1.556× |
| alpha-flower | 3,715,769 | 3,643,080 | −1.9562% | 8.9280 | 13.2866 | 1.488× |
| chessboard | 136,329 | 140,629 | +3.1541% | 0.7557 | 1.0187 | 1.348× |
| splines | 258,951 | 260,574 | +0.6268% | 3.3857 | 10.0053 | 2.955× |

Buffered/auto complete encoder CPU ratios are respectively **0.990×, 1.496×,
1.435×, 1.311× and 2.537×**. All raw CPU and wall repetitions are retained.
The combined bytes are 6,901,111 auto and 6,767,272 buffered, but that aggregate
is dominated by the photos and does not justify accepting the structured-image
tradeoffs. These five inputs are a targeted scope study, not a replacement for
the full 16-input or held-out rate corpus.

| Input | Auto learning calls / fits | Buffered learning calls / fits | Auto / buffered summed fit nodes |
|---|---:|---:|---:|
| photo-macan | 1 / 1 | 1 / 1 | 879 / 879 |
| photo-flower | 55 / 54 | 1 / 1 | 23,094 / 6,487 |
| alpha-flower | 55 / 54 | 1 / 1 | 33,344 / 8,289 |
| chessboard | 17 / 16 | 1 / 1 | 7,132 / 3,815 |
| splines | 65 / 64 | 1 / 1 | 9,786 / 3,665 |

The small photo is a useful negative control: both settings already choose the
same nonstreaming workflow and produce the same file, tree and prepared-stream
semantics. On the large photos and splines, included transformed pixel bytes,
descriptors, transforms and learning settings also match **by recorded stream
ID**, excluding histogram-coding settings. Sampling/quantization group scope,
tree growth and entropy scope still differ; this does not establish which of
those mechanisms causes each rate/time change. Chessboard additionally changes
some single-channel palettes and one WP choice, so it is a broader workflow
comparison.

## Concrete native gaps exposed by the traces

Per-stream settings matter before local trees are considered. Photo-macan's
four data groups use RCTs **8, 0, 40, 40** and WP mode **1** throughout, although
its tree is global. Photo-flower uses four nonidentity RCT types; **52 of 54**
data groups select WP mode 2. The alpha version selects WP mode 2 in **53 of
54** groups. The earlier native preparation bridge instead used default WP
parameters and frozen native transforms. Matching its samples/trees was a
valid common-profile control, not preparation parity with these actual e9
stream choices.

The structured traces also contain palettes larger than the native control's limit:
chessboard uses palettes of **662 and 862 colors**, and splines includes **285,
338 and 417 colors**. That control's `kMaximumPaletteColors` was 256, and its
coding policy and frame writer apply one shared WP/global transform policy.
Chessboard also has palette-before-RCT chains, including an RCT starting after
two metadata channels. That native policy represented RCT only as the first
transform on the first three color planes. The per-stream contract must preserve
transform order and channel ranges too. These are concrete representation/policy
boundaries, not inferred causes of the entire compression gap.

The later [ordered-transform contract](modular-ordered-transforms.md) qualifies
RCT sequences/ranges and palettes up to 1,024 entries. This supplies the
prescribed mechanics; per-stream coding scope and reference palette ordering
still require implementation.

The next implementation order is:

1. Add per-stream WP parameters to native sample preparation and replay these
   actual captured streams, matching reference samples and trees. Keep the
   existing default-WP bridge as a regression control. This is now qualified
   in [actual-stream preparation](modular-native-stream-preparation.md).
2. Define and qualify per-stream coding policies, including ordered local
   palette/RCT chains, their channel ranges, and the palette sizes actually
   required by these controls. Preserve exact
   pixels through native tokenization, headers and entropy emission.
   This is now qualified in the
   [prepared-stream coding replay](modular-prepared-streams.md), which also
   identifies a remaining downstream conditional rate gap.
3. Qualify global and local learning/coding on the same prepared streams, then
   port reference-style cheap transform/WP selection. Keep the all-predictor
   control and the existing exhaustive new-learner comparison; do not turn
   a global/local byte oracle or filename-specific choice into an automatic
   policy.
4. Apply the existing full-stage, rate, resource and held-out gates before
   promotion. Native defaults and the GPU handoff are unchanged by this study.

## Validation

Release and ASan/UBSan pass all seven research CTest entries. Additional
sanitizer captures cover a global photographic tree, local palette trees and
true 16-bit RGBA; all retain the stock reference bytes and exact decodes. The
validated Release executable is identical to the timed frozen executable.
Twelve negative controls reject corrupt trace metadata/pixel lengths,
inconsistent float bits, invalid WP parameters, mismatched tree/fit counts,
trace data passed as a timing control and invalid nested timing.

The analyzer verifies source/binary/artifact hashes, complete interleaved
coverage, serial process order, every pixel slice, fit-to-learning coverage,
raw stdout and all repeated byte/fit-shape identities. It computes stage
differences per encode before taking medians. Sanitizers instrument the native
objects, harness and generated reference units; linked archives remain
uninstrumented and vptr checks are excluded for no-RTTI interoperability.
The frozen validation manifest records these limits.
