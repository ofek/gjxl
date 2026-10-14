# Prescribed GPU token performance qualification

This extends [the unpack/RCT protocol](GPU_PERFORMANCE.md) with a third route:
GPU unpack/RCT and prediction/token generation in one command buffer and wait,
followed by the same CPU entropy encoder. All routes use the same prescribed
policy and produce identical complete bytes. The public dispatch policy is not
changed by this experiment.

Use `prepare_gpu_performance.py --token-slice --large-photos PHOTO1 PHOTO2 PHOTO3`
with three existing RGB photographs of at least four megapixels, optionally
adding the existing `--wesaturate` pilot. It retains decoded integer samples
without resizing or color conversion, and hashes both source photographs and
generated PAMs. The ninth TSV column selects `gradient` or `split7`. The latter
has seven nodes, properties 8/4/9, and Gradient/Select/Average4/Left leaves with
zero offsets and unit multipliers. The frozen benchmark source defines the
exact tree. The original matrix, six synthetic split-tree cases and six cases
per larger photograph cover one/four CPU participants and four simultaneous
calls. This is a selected matrix, not a full factorial photographic corpus.

Build `gjxl_modular_gpu_benchmark` from one frozen Release source snapshot with
Metal enabled, Metal profiling disabled and the pinned Modular oracle enabled.
Call the benchmark with a final `tokens` argument for three-route qualification
under Metal API/GPU validation. Then run `run_gpu_performance.py --tokens` and
`analyze_gpu_token_performance.py`. Complete correctness/resource/sanitizer
qualification and stop this task's builds/tests before ordinary measurement.

Each round measures all three complete synchronous calls. Six route permutations
balance every position and every pair's relative order over six measured rounds
per case in each of three fresh processes (seeds 17/43/91), after two warmup
rounds. Case order is shuffled. An additional process records six diagnostic
profile rounds for single-call cases; those times are never pooled with ordinary
measurements. Setup is recorded separately. Allocation, upload, dependency,
completion, token handoff, CPU continuation and publication are timed. Loading,
domain setup, output comparisons, decoding, hashing and I/O are outside timing.
Earlier results remain live through later calls in a round. Each route uses the
same four-participant domain and capacity: the largest complete-call bound among
the three routes, multiplied by simultaneous calls. Four-call workloads use one
CPU participant per call and include thread creation/join in batch wall time.

Every warmup, ordinary and diagnostic output must match the independently
decoded qualified file. CPU has zero GPU submissions/allocations; preparation
has one/two and tokens has one/four per call. All managed charges and CPU
participants must be released, with observed peaks within the common cap.
Token generation profiles have a distinct `gpu_input_tokens` interval containing
metadata preparation, upload, both kernels, wait and handoff; it is not GPU kernel
time. Scalar input/transform/token and CPU model/emission/assembly intervals keep
their existing meanings. No token traces or guard buffers run in timed calls.

Freeze `acceptance.json` alongside the input manifest before collecting timings.
For this pilot, the six primary cases are each large photograph under prefix
coding with four CPU participants, for gradient and split7. Each must reach a
median paired speedup of at least 1.05 with a 95% interval strictly above 1.0,
against both CPU and the earlier GPU-preparation route. For every four-call
case, the median and lower interval must be at least 0.98 against both controls
(a 2% regression tolerance). The serial cases and all other policies remain
reported, but cannot substitute for the four-participant primary gate. Passing
would support follow-up dispatch design only within tested policies and sizes;
failing leaves this path private. Correctness or a kernel-only gain cannot
satisfy the gate.

Intervals use 10,000 bootstrap replicates of the median paired ratio, resampling
within process cohorts, with a fixed seed. Report process and permutation
medians, actual managed peaks separately from capacity, ordinary throughput,
diagnostic stages and the full matrix. Intervals describe this run's noise,
not cross-device uncertainty or multiplicity-adjusted significance. Retain host
activity and thermal samples and qualify any background interference. Source,
binary, library, input, expected-output and raw-result hashes remain part of the
record. A second independent session is required before calling a small gain
repeatable enough for public dispatch.
