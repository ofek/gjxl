# Public Modular learned default

The C++, C, Rust and integer CLI entry points now default to the qualified
one-fit histogram workflow with Best predictors (Weighted/Gradient) and direct
ANS modeling using estimated selection and at most 64 clusters. Palette, RCT and
weighted-predictor selection precede one sample preparation and one fit per
nonempty learning scope. The automatic scope policy uses a shared tree through
eight 256-pixel groups and local trees above eight. This is the implementation
qualified in [the reference workflow](modular-reference-workflow.md) and
[its timing study](modular-reference-workflow-timing.md).

[The integration report](modular-public-learner-integration.md) qualifies this
default together with the CPU/Metal performance line on `feat/modular`.

The original default promotion added no further fitting/search, WP fallback,
LZ77, patches, or GPU dispatch. Explicit Metal integration is described below.
Rate and timing measurements in those
studies retain their original corpus and revision boundaries. In particular,
the learned policy is not guaranteed to beat legacy search on every image.

## Selecting behavior

| Choice | C++ / Rust policy | C policy suffix | CLI |
| --- | --- | --- | --- |
| Improved learner (default) | `kLearned` / `Learned` | `LEARNED` | `--policy learned` |
| Legacy identity/gradient | `kBaseline` / `Baseline` | `BASELINE` | `--policy baseline` |
| Legacy encoded-size portfolio | `kSearch` / `Search` | `SEARCH` | `--policy search` |

C policies have the `GJXL_MODULAR_POLICY_` prefix. The existing `search=true`
(C: `search=1`, CLI: `--search`) remains an alias for the legacy search policy.
Combining that alias with explicit baseline is rejected. Explicit prefix entropy
is supported with each policy; ANS is the new public entropy default. Private
prescribed CPU/GPU workflow options and their default bytes are unchanged.

`GJXLModularOptions` appends `policy` after `entropy`, preserving the old field
offsets and enum values. Null options, four/eight-byte prefixes, and the
initializer select learned/ANS. A 12-byte caller retains its explicitly supplied
search/entropy fields and defaults to learned policy. Full zero-initialization
with `struct_size` set explicitly requests prefix (its existing enum value is
zero); use `gjxl_modular_options_init` for current defaults. Rust struct literals
can use `..ModularOptions::default()` for the new field.

All six integer layouts, endian/padded-row handling, hidden RGB, atomic output
and automatic CPU backend resolution retain their public contracts.
The learner supports up to 2^28 source samples and canonical group IDs below
65535; preparation admits at most 16M training samples and rejects overflow
rather than silently truncating. Explicit legacy policies retain their existing
profile limits.

## Explicit Metal execution

The learned policy accepts `ModularBackend::kMetal`, a C/Rust context with the
Metal backend, or CLI `--backend metal`. Automatic continues to select CPU.
CUDA, CPU-only builds requesting Metal, and public legacy-policy GPU requests
return Unsupported. The existing private legacy GPU operations remain available.

After CPU transform/WP selection, all coded channels are batched with their own
WP parameters, canonical stream/channel IDs and tree indices. One completed
weighted-feature map supplies both the CPU sample learner and Metal adaptive
token generation. Each scope still fits exactly once. Tokens are generated in
one batch after fitting and borrowed directly by the existing CPU entropy
encoder; data tokens and weighted predictions are not regenerated there.
Transform/WP selection, fitting, entropy modeling and bitstream emission remain
CPU work.

The weighted producer retains the existing automatic cutoff: at least 4M
samples and an average of 16,384 samples per channel job. Eligible channels up
to 256x256 use the wavefront kernel. Small batches and oversized metadata
channels use the scalar producer before GPU submission, preserving exact
channel reset boundaries. Mixed batches share the completed map. Execution
failures propagate without a CPU retry. Explicit Metal still generates data
tokens on the GPU when weighted features were produced on the CPU.

Feature preparation retains a packed 4-byte plane and 16-byte feature per coded
sample until token generation finishes. The completed split token arena uses
4-byte values and 2-byte contexts, plus alignment. These buffers, descriptors,
tree tables and stage scratch are admitted in the same managed domain as the
CPU stages. Features are released before entropy coding; token backing survives
through encoding. This integration adds retained memory and has not established
an end-to-end performance win; Automatic awaits performance qualification.

`metal_modular_reference_workflow` checks exact trees, all tokens, ANS/prefix
codestream parity and pinned-decoder samples across all six formats, padded and
unaligned input, endianness, hidden RGB, shared/local trees and buffered All
predictors. Its mixed batch covers distinct WP headers, tree tables, strided
channels, wide/tall scalar fallback, and allocation/submission/completion
failures. Public API and CLI tests also compare explicit Metal with CPU bytes.

The 2026-10-15 integration checks passed: 10 Release Metal tests, five Release
CPU-only tests, and the two host handoff/preparation tests under ASan/UBSan.
The new workflow test matched 36 complete CPU/Metal codestream pairs and 104,482
learned tokens. The Metal CLI matrix decoded 90 files exactly and preserved
outputs for 16 invalid requests. The existing adaptive-token, weighted-feature,
weighted-workflow and shared-feature Metal regression tests also passed.
Logs are `build/modular-integration-20261013/{cpu,metal}/feature-token-tests.log`
and `build/modular-rct-search/sanitize/feature-token-tests.log`. These establish
correctness and failure cleanup, not throughput qualification.

## Resources

CPU work in the learned workflow currently executes serially, within the requested
CPU and shared-domain participation limits. A higher thread limit does not imply a
speedup. The existing parallel tokenization/emission path remains available in
the legacy policies.

Each learned stage uses its checked storage bound and nonblocking `TryReserve`
in the public execution domain. Retained planes, trees and output backing stay
charged while subsequent stages run. The C adapter reserves its final output
copy in that same domain while the native codestream is still live. There is no
claim of a shape-only whole-workflow bound or bounded-memory streaming.
Insufficient capacity for an individual request returns OutOfMemory; unavailable
capacity alongside retained backing or other work returns Unavailable. No memory
wait holds earlier stage backing or CPU permits. Caller-owned input and published
output remain outside the managed-memory limit, as before.

The implementation shares option validation and owned-output dispatch between
C++ and C. Stage resource contexts preserve the execution domain. Errors leave
output/summary unchanged and release managed backing, reservations and CPU slots.

## Original CPU-default qualification

Validated on Apple Silicon with AppleClang 21.0.0, using the clean pinned libjxl
decoder at `e8ff09762481785938d8e4e01333ed3917571161`. Native production libraries
retain no libjxl dependency. Build and test artifacts are under this worktree's
`build/public-default`, `build/public-default-asan`, and `build/public-rust`.

| Check | Result |
| --- | --- |
| Release `modular_api`, `modular_cli`, `modular_reference_workflow` | Pass |
| ASan/UBSan same three tests | Pass |
| Release `c_api`, `c_api_cpp`, `workflow_admission_cpu` | Pass |
| Installed C/C++ consumer, relocation and installed-header boundary | Pass |
| Rust `modular_tests` with native Metal-enabled build | Pass |

The public API matrix covers 112 cases: 72 explicit legacy and 40 learned cases.
It compares C/C++ output at CPU limits 1 and 4 and independently decodes exact
samples, including endian/padding, hidden RGB and eight/nine-group boundaries.
Every learned case also matches the private qualified workflow byte for byte;
its limited domain admits the measured private peak without bypassing accounting.
The tests cover default/null/4/8/12/full-size C options, initializer canaries,
invalid options, tiny/busy shared C/C++ domains, mixed VarDCT calls, forced GPU
rejection, and atomic output/summary replacement. The learned implementation
uses one CPU participant even with a limit of four.

All 1,197 C++ managed-allocation failure positions and 513 C publication-path
failure positions pass, including cleanup/accounting checks. The CLI independently
decodes 70 files, rejects 14 invalid requests without replacing output, and
matches its legacy search alias. Rust verifies defaults against C null options
for all nine format/endian layouts. The saved 71 CLI files (including the alias
control) and 98 private workflow files are byte-identical across Release and
ASan/UBSan; hashes are in `build/public-default-byte-parity.json`.

Configure with `GJXL_BUILD_TESTS=ON`, `GJXL_ENABLE_METAL=OFF`, and
`GJXL_MODULAR_ORACLE_BUILD` pointing at the pinned libjxl build. Release uses
`CMAKE_BUILD_TYPE=Release`; sanitizer validation uses `RelWithDebInfo`,
`-fsanitize=address,undefined -fno-omit-frame-pointer`, and the matching linker
flags. The final sanitizer command is:

```sh
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/public-default-asan --output-on-failure \
  -R '^modular_(api|cli|reference_workflow)$'
CARGO_TARGET_DIR=build/public-rust CMAKE_BUILD_PARALLEL_LEVEL=6 \
  cargo test --manifest-path rust/Cargo.toml -p gjxl modular_tests --offline
```

Logs are `build/public-default-tests.log`, `build/public-default-asan-tests-v2.log`,
`build/public-api-tests.log`, `build/public-install-tests.log`, and
`build/public-rust-tests.log`. The initial broader Release run's exhaustive
legacy `modular_workflow` allocation sweep was intentionally stopped and is
not counted as passing; all explicit legacy cases in the public matrix passed.
The installed-consumer check passed after building the remaining installed
`gjxl_gpu_butteraugli` library. An initial sanitizer invocation requested
unsupported macOS LeakSanitizer and aborted before testing; the final run
disables only that facility. Linked libjxl is uninstrumented. Explicit managed
resource leak checks remain active. These are correctness checks, not new
compression-rate or performance measurements.
