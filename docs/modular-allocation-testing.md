# Modular allocation-failure qualification

Ordinary CI exhausts the public learned workflow and the inexpensive legacy
fixtures. The four expensive legacy-search sweeps are explicit qualification
work. This changes test selection; encoder policies and allocation behavior are
unchanged.

## Coverage

| Test / mode | Failure coverage | Ordinary CI | Full qualification |
| --- | --- | --- | --- |
| `modular_public_allocations` | Every managed allocation in four public learned RGBA16 cases, prefix and ANS | Yes | Yes |
| `modular_workflow --regular` | Every managed allocation in 40 legacy baseline/prescribed cases; eight probes in each of four search cases | Yes | Yes |
| `modular_legacy_search_faults_shard_N` | Every allocation in the four legacy search cases, partitioned across shards | No | Yes |
| Existing API, reference-workflow, training and component tests | Their existing C/C++ publication, resource and failure coverage | Yes | Yes |

The learned fixtures are 17x19 and 257x3 RGBA16, big-endian, with the same byte
pattern, input offset and padded rows as the legacy search fixtures. They use
the public learned default, with explicit prefix/ANS selection and one CPU
participant. Every injected failure must preserve output, summary and input;
release managed backing, reservations and CPU participation; and return
OutOfMemory. The first non-failing index must reproduce the successful file.
When the pinned Modular oracle is enabled, libjxl independently decodes that
file and checks every source sample.

The regular legacy search probes include the first and last allocation plus six
evenly spaced positions. A successful encode measures the current allocation
count before selecting positions; no compiler-specific allocation indices are
hardcoded. These are bounded smoke checks, not exhaustive search coverage.

The learned path has staged resource admission and a different algorithm from
legacy candidate search. Passing its fault tests does not establish failure
handling inside the legacy portfolio or replace its prescribed-transform tests.
Full qualification therefore retains the exact legacy failure loop, its four
search fixtures, every failure index, and the 7,200-second per-shard timeout.

## Ordinary tests

`GJXL_ENABLE_EXHAUSTIVE_MODULAR_SEARCH_TESTS` defaults to `OFF`. Expensive shards
are absent from CTest registration, so an unfiltered `ctest` is also safe from
accidentally starting them. Labels alone would not provide that exclusion.

```sh
cmake -S . -B build/allocations -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_ENABLE_METAL=OFF -DGJXL_ENABLE_CUDA=OFF \
  -DGJXL_ENABLE_EXHAUSTIVE_MODULAR_SEARCH_TESTS=OFF
cmake --build build/allocations --parallel 4 --target \
  gjxl_modular_workflow_test gjxl_modular_public_allocation_test
ctest --test-dir build/allocations --output-on-failure --parallel 1 \
  --no-tests=error -L '^modular_allocations$'
```

Add `-DGJXL_MODULAR_ORACLE_BUILD=/path/to/clean/pinned-libjxl-build` to enable
independent decoding. Native hosted CI already configures that oracle. Reports
explicitly record whether decoding was enabled; production libraries still do
not link libjxl.

## Full legacy-search qualification

Use a separate build directory, or explicitly enable the option in an existing
build. Four shards are used on hosted Linux/macOS; Windows uses sixteen smaller
shards at concurrency four. The union covers the same indices in either case.

```sh
cmake -S . -B build/modular-exhaustive -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_ENABLE_METAL=OFF -DGJXL_ENABLE_CUDA=OFF \
  -DGJXL_ENABLE_EXHAUSTIVE_MODULAR_SEARCH_TESTS=ON \
  -DGJXL_MODULAR_WORKFLOW_SHARDS=4
cmake --build build/modular-exhaustive --parallel 4 \
  --target gjxl_modular_workflow_test
ctest --test-dir build/modular-exhaustive --output-on-failure --parallel 4 \
  --no-tests=error -L '^modular_search_exhaustive$'
```

The executable also accepts `--all-faults` for the original 44-fixture sweep.
Historical `--allocation-shard INDEX COUNT` commands without an explicit suite
still select that complete sweep. Unqualified invocation now selects regular
coverage. `--regular` rejects sharding, avoiding accidental sampling of the
otherwise exhaustive inexpensive fixtures.

The existing Native Libraries workflow has an opt-in manual input:

```sh
gh workflow run native.yml --ref BRANCH \
  -f exhaustive_modular_search=true
```

An explicit dispatch runs regular qualification and the exhaustive search job
across Linux/macOS/Windows and C++20/C++23. PR and main-push runs omit the
exhaustive job. Weekly qualification is scheduled for Monday 06:23 UTC, effective
once this workflow change reaches the default branch. The exhaustive job builds
only the CPU legacy-workflow target and does not need the reference decoder.

## Reports and evidence

CTest records status and duration. Each target also writes complete JSONL under
`modular-allocation-reports/`, because CTest truncates successful stdout. Reports
include fixture, policy, entropy, failure count, shard/probe positions and elapsed
time. Every concurrent shard has a distinct report filename. A passing row alone
does not establish whole-suite completion: check CTest status, expected fixture
counts and the legacy suite's final summary.

[The results ledger](modular-allocation-testing-results.json) records the
matched diagnostic, local qualification and registration checks. Historical
complete CI coverage is retained in the
[PR #48 run](https://github.com/yunho-c/gjxl/actions/runs/38522028697), rather than
being represented as a run of this new workflow.

On the matched Apple Silicon Release fixtures, search failure positions fell
from 150,007 to 7,221 with the public learned workflow (20.8x fewer). Keeping the
other 18,094 failures and adding 32 bounded search probes gives 25,347 positions
in the two regular targets. This excludes existing API/component fault tests,
whose coverage is unchanged. Counts vary by compiler and platform; they are
evidence, not fixed acceptance constants.

The timing improvement applies to this allocation-test portion, not the whole
native CI suite. On an Apple M4 Pro with AppleClang 21, the committed test code
passed these local checks:

| Target | Release | ASan/UBSan |
| --- | ---: | ---: |
| `modular_workflow` | 9.93 s | 53.51 s |
| `modular_public_allocations` | 18.48 s | 97.37 s |

Independent libjxl decoding was enabled. The final regular-test rerun and the
unchanged public-allocation test's preceding run are recorded separately in the
ledger. LeakSanitizer is unavailable on this macOS setup; explicit managed
resource-release checks remain active and the linked decoder is uninstrumented.
Default exclusion and opt-in 1/4/16-shard registration passed, as did the CLI
rejection checks and `actionlint`.

Hosted qualification of this new workflow has not yet run. The results ledger
distinguishes the bounded opt-in shard smoke check from a complete exhaustive
run; neither that smoke check nor historical PR #48 results establish a new
complete hosted matrix.
