# Native reference palette construction

The private CPU path now reproduces pinned libjxl's lossless palette
construction on the qualified controls, including implicit colors, entry
ordering and actual dictionary size. This completes the construction primitive
needed alongside the [gradient](modular-selection-cost.md) and
[weighted-predictor](modular-weighted-cost.md) selection estimators. Choosing
palette proposals, RCT/WP settings and training scope remains the next
integration step; no public default changes here.

The subsequent [channel-compaction extension](modular-channel-palette.md)
adds single-channel dictionaries through the 70,911-entry wire maximum. The
bounded construction and measurements below describe this earlier milestone;
the extension retains its outputs and adds sparse scratch planning and wider
header support.

The [qualification ledger](modular-reference-palette-results.json) binds the
source, binaries, tests and generated files. These checks establish construction
and coding correctness, not whole-policy parity, rate recovery or a speedup.

## Construction and representation

[`TryBuildReferencePalette`](../src/codec/modular/transform/reference_palette.h)
borrows strided channel views and returns an owned candidate only after a
successful proposal. It supports one to four equally sized normal channels,
depths 1–24, requested color limits 0–1,024, and ordered/unordered construction.
The current frame encoder still admits only its existing 8/16-bit formats.
Lossy palettes and delta entries remain unsupported.

The behavior follows pristine reference
`e8ff09762481785938d8e4e01333ed3917571161`:

- A single-channel dictionary uses ascending distinct values. Uniform channels
  produce one entry when the requested limit is at least two.
- Multichannel discovery preserves first occurrence order and excludes the
  189 implicit RGB cube entries from the requested explicit-color limit.
  Uniform multichannel proposals are rejected, as in libjxl.
- Implicit entries occurring more than ten times also enter the explicit
  dictionary. Enumeration retains duplicate implicit colors, which matters for
  two-channel and low-depth inputs. The final explicit width can be **zero**
  or as large as **1,213**, despite a requested limit of 1,024.
- Ordered RGB/RGBA dictionaries use the reference float luma/alpha expression
  and frequency threshold of four. Image order is retained for unordered and
  two-channel palettes; repeated entries use the reference's last-index rule.

The native code uses a fixed hash dictionary and saturating frequency counts
because only the two thresholds matter. It avoids libjxl's per-color heap
containers and unused lossless WP histories. Its fixed dictionary occupies
less than 35 KiB of stack. This is a design/resource property, not a measured
timing result. Equivalent-key ordering is qualified against the same Apple
Clang/libc++ build; it is not a cross-standard-library ordering guarantee.

The new descriptor fields select reference construction and carry its bit
depth and ordering requirement. Existing descriptors keep their previous
explicit-capacity behavior. Bit depth is supplied by the enclosing frame on
the wire, so frame, workflow and prepared-stream admission reject mismatches.
The existing palette header encoding already accommodates the expanded width.

Zero-width metadata planes retain their channel IDs without sample backing.
Image validation accepts only the bounded palette shape, while slicing,
transform copies and tokenization handle it explicitly. A global stream with
only an empty dictionary can therefore use local group trees without a shared
tree. Nonempty metadata and index planes continue through the existing
prediction and entropy paths.

## Resource and failure contract

The planner reads descriptors/backing extents without scanning pixels or
allocating. Its bound covers fresh output channel records and planes at the
largest possible explicit width for the requested limit. There is no managed
scratch beyond that output. Borrowed input, a pre-existing output, fixed stack
objects and allocator overhead are excluded; this is not an RSS bound.

Too many colors, a uniform multichannel image, or a requested limit below two
returns a successful rejection with no output image or managed allocation.
Invalid views/options and allocation/admission failures preserve the previous
output. Publication also supports input views borrowed from the image being
replaced. The prescribed-transform path can reconstruct the same candidate
from its descriptor, and its inverse supports explicit and generated implicit
indices with checked rejection of unsupported indices.

## Qualification

| Check | Result per build |
|---|---:|
| Generated construction cases | 876 |
| Accepted / rejected proposals | 588 / 288 |
| Accepted zero-width dictionaries | 116 |
| Explicit widths above the requested limit | 8 |
| Injected managed-allocation failures | 70 |
| Maximum observed candidate managed peak | 27,424 bytes |
| CTest entries passing | 13 / 13 |
| New complete files decoded exactly | 172 |
| Existing prepared-stream files unchanged | 648 / 648 |

Both Release and ASan/UBSan compare acceptance, predictor, palette width,
dictionary entries, indices, dimensions, shifts and untouched channels against
the actual pinned `FwdPalette`, then compare both inverses with the original
pixels. Tests cover depths 1, 2, 3, 4, 7, 8, 12, 16 and 24; all component counts;
strided/shifted planes; existing metadata; signed extremes; shuffled discovery;
equal sorting keys; promotion thresholds; the maximum explicit width; and
palette/squeeze composition. Additional contracts check invalid input,
underplanned admission, atomic allocation failure, recovery and released
backing. Empty-dictionary checks also verify preserved training channel IDs,
estimator costs and invalid inverse-index rejection.

The 172 files cover 24 Gray/RGB/RGBA 8/16-bit cases with global/local palettes,
shared/local trees, and Prefix, ANS-from-Prefix and estimated direct ANS.
They include a dictionary wider than 1,024 and global/local empty dictionaries.
All decode exactly with pinned libjxl. The 648 prior files cover the established
108 prepared-stream cases and six entropy controls; they match the
prepared-entropy milestone byte for byte. All **820 files** also match between
Release and sanitizer builds.

The 13 checks include the existing image/stream/transform contracts, ordered
transforms, sample preparation, prepared streams and both estimator backends.
Gradient and WP reference costs/rankings remain unchanged. The validation
freeze binds **4,963 artifacts**. Native code and test adapters are instrumented
in the sanitizer build; linked reference archives are uninstrumented, reference
vptr checks are excluded, and unavailable macOS LeakSanitizer is disabled.
Managed backing leak checks pass. Test durations are not performance evidence.

## Reproduction and next boundary

The native tests are `modular_reference_palette` and
`modular_reference_palette_oracle`; the latter needs
`GJXL_MODULAR_ORACLE_BUILD` pointing to the clean pinned reference build.
The report can be independently rechecked against the local frozen artifacts:

```sh
python3 tools/modular_reference_learning/analyze_reference_palette.py \
  --validation build/reference-palette-validation \
  --baseline-validation build/prepared-entropy-validation \
  --output /tmp/gjxl-reference-palette-rechecked.json
```

Use a new output path. `build/reference-palette-validation/freeze.py` records
the exact source/build/log/file capture procedure. The generated-source
snapshot is authoritative for the pre-commit build.

Next implement the reference's proposal ordering and acceptance rule using
these primitives, then select RCT/WP and shared/local scope before preparation
and fitting. Compare complete native decisions with reference decisions on
matched original inputs; the earlier transformed-plane controls cannot establish
that equality. The exhaustive new-learner comparison, held-out rate, whole-stage
timing/resource gates and high-depth LZ77 exception remain open under the
[search plan](modular-reference-search-policy.md).
