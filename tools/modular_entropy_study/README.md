# Identical-token entropy study

`entropy_replay` is a research executable linked against the built native static
libraries and pinned libjxl internals. It does not expose a new production policy.

```sh
cmake -S tools/modular_entropy_study -B build/entropy-replay -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD="$PWD/build/release" \
  -DREFERENCE_SOURCE=/Users/yunhocho/GitHub/gjxl/third_party/libjxl \
  -DREFERENCE_BUILD=/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl
cmake --build build/entropy-replay --parallel 6
python3 tools/modular_entropy_study/study.py capture \
  --pipeline build/entropy-pipeline-qualified-20261011 \
  --root build/entropy-rate-qualified-20261011 \
  --binary build/entropy-replay/entropy_replay
python3 tools/modular_entropy_study/study.py analyze \
  --root build/entropy-rate-qualified-20261011 \
  --json docs/modular-entropy-rate-results.json
```

First freeze the complete exhaustive winner's policy and scalar ordered tokens
for each of the 19 retained images. Every replay reads the same little-endian
archive and checks its stream counts against the canonical layout, context bounds,
truncation and trailing bytes. The harness hashes the input, policy, archive,
executable and all resulting complete files. Refuse to overwrite existing archives
or outputs. The native prefix-derived estimated baseline must reproduce the
complete-search winning file exactly. Every cell is independently decoded against
the original samples.

Fourteen cells hold tokens, contexts, transforms, tree, prediction, framing and
stream boundaries fixed, with ANS forced and LZ77 disabled:

- Native prefix-derived estimated selection, maximum 32 clusters.
- Native direct balanced/high-density construction at capacities 32 and 64.
  These change several model choices and are explicitly multifactor controls.
- Pinned libjxl e7 streaming model construction at capacities 16, 32, 64 and 128.
  The four cells isolate its capacity limit; actual cluster counts are reported.
- Native construction using each libjxl e7 partition at capacities 16, 32 and 64.
  These hold the downstream native configuration/normalization/width policy fixed,
  varying only the supplied partition. The private prepared-partition entry skips
  prefix validation but uses the same maximum-compression estimated ANS policy.
  Discovery of the reference partition is excluded from these diagnostic times.
- Pinned libjxl e9 streaming and nonstreaming construction at capacity 128.
  This exposes its effort/HybridUint policy difference on the same tokens; it does
  not reproduce default complete-encoder behavior with adaptive trees and LZ77.

Report actual payload bits, context-map bits, remaining histogram/configuration
header bits, the LZ77 flag, common tree bits and complete file bytes. Re-serialize
the selected context map for attribution and verify decomposition. Do not infer
complete encoder gains from this fixed-token study. Builder timing includes final
model-header serialization; reference token representation conversion and separate
attribution serialization are excluded. Native-on-reference-partition timing also
excludes discovery of that supplied partition. These are single diagnostic calls,
not a throughput qualification.

The native ANS representation remains capped at 64. No native capacity-128 cell is
invented by merely changing an option. A production expansion would need separate
representation, storage, decoding and rate/time qualification. A policy is promoted
only after material measured benefit; a negative result is useful evidence.

The local pilot decodes all 14 model cells for both a single-context tree and a
split tree with an empty context. The latter also checks the reference writer's
required reservation for the three-bit singleton context map. It preserves frozen
archives and rejects truncated, extended, wrong-magic, wrong-count and overwrite
inputs. Run it with:

```sh
python3 tools/modular_entropy_study/test_replay.py build/entropy-replay/entropy_replay
```

Full corpus qualification is pending.
