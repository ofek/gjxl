# Public learner integration

PR #48 integrates the public histogram learner with the existing CPU/Metal
Modular performance work. Merge commit `cff30958a3f7f618a66da34ffa5a1520506d1542`
joins public-learner commit `4d43c77` and `feat/modular` commit `2520a7f`, preserving
both histories. Integration took place in the existing public-learner worktree;
uncommitted population-fusion work in the `feat/modular` worktree is outside this
change and qualification.

The public C++, C, Rust and integer CLI default remains the one-fit CPU histogram
learner with estimated direct ANS, as described in
[the public-default contract](modular-public-default.md). The legacy search and
its optional GPU feature, token, population and exact-cost operations remain
available. This integration does not connect those GPU operations to the new
learner or establish a new performance/rate result.

## Integration changes

- Keep both sets of sources in CMake, and retain process-CPU timing alongside
  legacy training, entropy and GPU diagnostics.
- Count exact ANS candidates once before the optional GPU evaluator or CPU
  fallback. Estimated selection still avoids simulation unless exact costs are
  requested for the selected model.
- Preserve direct-ANS and prefix-partition tests. Evaluator tests compare against
  the prefix-partition path; direct ANS can choose a different partition/model.
  Check candidate counters when the evaluator requests CPU fallback.
- Retain all 17 entropy storage cases, including estimated direct ANS with and
  without costs, and prefix-population retention without costs.
- Use `kMaximumSearchPaletteColors` (256) in the GPU search storage planner. The
  new learner raises the general explicit palette limit to 1,024; passing that
  limit into the unchanged legacy search rejects every GPU search plan. Existing
  population-boundary and GPU candidate-search tests caught this integration
  failure. The fix preserves the qualified legacy search portfolio.
- Build all registered native targets before the CI Modular test step. The
  previous explicit target list omitted new learner/oracle tests. This moves the
  existing full build earlier rather than maintaining a second incomplete list.

## Qualification

Validation uses AppleClang 21 on Apple Silicon, Release CPU-only and Metal
builds, and a CPU-only ASan/UBSan build. Independent decoding uses pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`; production libraries do not link it.

The [committed results](modular-public-learner-integration-results.json) record
individual test outcomes, initial failures and targeted reruns, and artifact
hashes. Final outcomes are:

| Build | Passed tests | Unresolved failures |
| --- | ---: | ---: |
| CPU-only Release | 58 | 0 |
| Metal Release | 72 | 0 |
| CPU-only ASan/UBSan | 7 | 0 |

The initial CPU suite failed `entropy_population`; the initial Metal suite
failed that test and `modular_candidate_search`, all with the palette-limit
error described above. Both executables were rebuilt and the affected tests
passed on rerun. The remaining GPU workflow checks ran after the correction.
The evidence retains the initial failures rather than presenting the original
whole-suite runs as failure-free.

The public API checks cover 112 C/C++ cases, 1,197 learned C++ allocation-failure
positions, and 513 C publication-failure positions. The CLI independently
decodes 70 files, preserves output for 14 invalid requests, and checks the legacy
search alias. All 169 saved CLI/reference workflow files match their hashes
from before integration in the CPU, Metal and sanitizer builds.

The separate exhaustive legacy `modular_workflow` allocation sweep is excluded
from this integration run because of its previously documented excessive
runtime. Dedicated API, training, ANS and GPU workflow failure tests remain in
scope. This is selected correctness qualification, not the full repository suite
or a matched performance benchmark. Hosted CI results are tracked by PR #48.

Hosted CI still runs the exhaustive legacy allocation sweep. It now sets
four shards on Linux/macOS and sixteen on Windows, with four CTests concurrent. The existing
`--allocation-shard INDEX COUNT` interface partitions failure indices; all shard
indices are registered, so every fixture and every failure position remain in
scope. This follows the complete four-shard qualification in
[the palette-context report](modular-palette-context.md#validation-and-evidence).
The default CMake value remains one, preserving the standalone test and its
name. Each shard retains the existing 7,200-second timeout. This schedules the
existing test across cores; it does not reduce coverage or change codec behavior.
The four generated commands and the unchanged single-test default were checked
locally. A smoke run of shard 0/64 passed all 44 fault-case fixtures and 2,650
injected failures; this subset is not a new exhaustive qualification. The full
allocation coverage is assigned to hosted CI.

### Hosted Windows follow-up

The initial hosted Windows builds exposed pre-existing portability errors in
`earlier_dc_contract_test.cpp` and `entropy_readiness_contract_test.cpp`; both
files were unchanged from `2520a7f`. Their POSIX `setenv`/`unsetenv` calls do not
compile with MSVC. A shared test helper now uses `_putenv_s` on Windows and the
original POSIX operations elsewhere, checking errors on either platform. The
concurrent test calls also capture an explicitly constructed options object,
avoiding the brace-initializer expression rejected by MSVC 14.37.

Both scheduling contract tests passed locally after this test-only correction.
Hosted validation of the Windows branch remains recorded on the PR; the local
Apple Silicon result is not a Windows qualification. The original diagnostic
is [the Windows C++20 build log](https://github.com/yunho-c/gjxl/actions/runs/38501488749/job/115649229436),
with a local copy at `build/pr48-windows-first-build.log`. The follow-up CTest
report is `build/public-default/build/pr48-portable-tests.xml`.

### Hosted runtime and checkout follow-up

At `f63843c`, both Linux native jobs passed 47 Modular and 120 other CTests;
both macOS jobs passed 50 Modular and 185 other CTests. CUDA compilation and
all three Rust checks passed. These results include the complete four-shard
allocation sweep on Linux and macOS. Logs are in `build/pr48-*-final-tests.log`;
the [hosted run](https://github.com/yunho-c/gjxl/actions/runs/38506867831)
retains the platform evidence.

Windows C++20 reported eight CTest entries unable to load reference DLLs,
two frozen-manifest hash failures, and four allocation-shard timeouts. The
candidate and training tests now receive the same reference DLL search path
as the existing oracle tests, along with both ANS selection modes that use the
same dependency. The exact-byte oracle manifests now explicitly retain LF on
Windows checkout; `git -c core.autocrlf=true cat-file --filters` reproduced the
original hash failure and verified the corrected checkout against the unchanged
expected SHA-256. Capture artifacts already use binary streams.

Each timed-out Windows shard completed 33 of 44 fault fixtures before the
existing 7,200-second limit. Windows now uses sixteen smaller, disjoint shards
at concurrency four, keeping every failure index and the timeout unchanged.
The generated sixteen commands were verified locally. Native CI runs the
quicker Modular checks and remaining repository tests before the exhaustive
sweep, with separate JUnit reports. Their test selections were verified to be
disjoint and collectively exhaustive. This follow-up changes test setup only;
Windows qualification remains pending on the updated PR revision.

## Reproduction and evidence

Configure with tests and benchmarks enabled, CUDA disabled, and
`GJXL_MODULAR_ORACLE_BUILD` pointing at the pinned libjxl build. The Release
builds differ only in `GJXL_ENABLE_METAL`. Run:

```sh
ctest --test-dir BUILD --output-on-failure --parallel 2 --timeout 600 \
  --no-tests=error \
  -R '^(modular_|ans_|entropy_|metal_modular_|metal_ans_token_cost$|c_api$|c_api_cpp$|workflow_admission_cpu$)' \
  -E '^modular_workflow$'
```

The sanitizer build uses `RelWithDebInfo`,
`-fsanitize=address,undefined -fno-omit-frame-pointer` and matching linker flags:

```sh
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir BUILD_ASAN --output-on-failure --parallel 2 --timeout 600 \
  -R '^(modular_(api|cli|reference_workflow|profile)|ans_selection|entropy_(storage_plan|population))$'
```

macOS LeakSanitizer is unavailable; managed allocation/accounting checks remain
enabled. Linked libjxl is uninstrumented. `actionlint` validates the CI change.
Local build commands/logs and the pre-integration hash manifest are under
`build/pr48-*`. CTest JUnit/log files are under each build directory's `build/`
subdirectory. Release directories are `build/public-default` and
`build/pr48-metal`; sanitizer artifacts are in `build/public-default-asan`.
