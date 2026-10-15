# Learned Modular Metal qualification

Qualification of `05f2f2fcd4d3a2c3bf418bc522d96998b22c6b5e` with a separate benchmark driver. Production defaults and the feature heuristic are unchanged.

All 396 complete encodes (374 primary and 22 repeat) matched their CPU control byte for byte and decoded exactly with the pinned independent libjxl decoder. All 651 prepared maps matched the serial predictor (3,547,578,678 feature entries compared). All managed domains released their backing and CPU participation.

The 16 original corpus outputs also match the earlier frozen CPU qualification files. The pinned decoder revision is `e8ff09762481785938d8e4e01333ed3917571161`; its loaded-library hashes are retained.

On the primary 17-image capture, explicit Metal with the current feature heuristic uses 5.97% less elapsed time than CPU (ratio of sums of image medians). With GPU tokens held fixed, forcing GPU feature preparation uses 1.74% less time than forcing the serial CPU producer. These are different comparisons: the first includes the feature-reuse/token integration; the second isolates producer choice within that integration.

## Measured scope

Serial processes, one input per process. Public CPU/Metal-auto calls; forced producers mirror the public wrapper. First public CPU/Metal calls are cold controls, then one warmup and rotating-order measured rounds. Encoder timer includes admission, source preparation, selection, features, fitting, tokens, entropy, frame cleanup and output publication. Source IO, domain construction, verification and output IO excluded. Operation timer includes allocation, copy, predictor computation, dispatch, wait and publication; selected channels, backend setup, admission and output destruction outside. Managed peaks exclude borrowed inputs, published output, immutable backend, driver and oracle storage. Active desktop; no concurrent qualification/build/test processes launched here.

Four measured rounds follow one warmup for each encoder variant. Each variant occupies every order position once per image. Feature operations use six measured rounds after one warmup, with each producer occupying each position twice. Cold public CPU/Metal calls are retained separately. No instrumentation environment variables were set during this capture.

CPU and Metal-auto use the public C++ API with learned policy, estimated ANS and a one-participant execution domain. Metal-cpu and Metal-gpu use the same learned workflow and publication boundary through a qualification-only wrapper, forcing only feature preparation. Token generation remains on Metal in both. The current learned CPU pipeline is serial even when a larger thread ceiling is allowed. This measures application of already selected WP parameters. Transform search and the five-preset WP selection sweep remain on CPU.

`metal-auto` means explicit Metal encoding with automatic **feature-producer** selection. It does not mean `ModularBackend::kAutomatic`, which still selects CPU.

## Complete encoder

The aggregate is a sum of per-image medians, with one encode of each image. It is not a corpus-independent throughput estimate. Ranges and all trials are retained in the JSON report and raw JSONL captures.

| Variant | Sum of wall medians (s) | Speedup over CPU |
|---|---:|---:|
| cpu | 68.5092 | 1.000× |
| metal-auto | 64.4221 | 1.063× |
| metal-cpu | 65.3657 | 1.048× |
| metal-gpu | 64.2251 | 1.067× |

| Image | CPU (s) | Metal-auto (s) | Metal-cpu (s) | Metal-gpu (s) | CPU / Metal-auto | CPU / Metal peak backing (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| photo-macan | 0.5786 | 0.5373 | 0.5437 | 0.5257 | 1.077× | 23.15 / 36.42 |
| photo-ria | 0.5677 | 0.5321 | 0.5312 | 0.5321 | 1.067× | 22.39 / 35.67 |
| photo-bliznaca | 0.5842 | 0.5496 | 0.5542 | 0.5393 | 1.063× | 23.89 / 37.14 |
| photo-flower | 8.0642 | 7.4939 | 7.5988 | 7.5542 | 1.076× | 223.03 / 334.96 |
| gray-macan | 0.1822 | 0.1756 | 0.1754 | 0.1774 | 1.037× | 7.25 / 11.13 |
| alpha-ria | 0.7914 | 0.7739 | 0.7874 | 0.7705 | 1.023× | 29.86 / 46.21 |
| alpha-flower | 11.3501 | 10.4516 | 10.5031 | 10.4468 | 1.086× | 296.21 / 446.71 |
| chessboard | 0.9428 | 0.9204 | 0.9102 | 0.9014 | 1.024× | 38.83 / 50.10 |
| splines | 4.7829 | 4.5559 | 4.6528 | 4.5478 | 1.050× | 218.51 / 336.08 |
| gray-patches | 0.2369 | 0.2294 | 0.2293 | 0.2318 | 1.033× | 8.05 / 13.38 |
| palette-358 | 0.0122 | 0.0123 | 0.0120 | 0.0145 | 0.987× | 1.14 / 1.28 |
| gray16-ramp | 0.0488 | 0.0479 | 0.0485 | 0.0526 | 1.018× | 3.61 / 5.92 |
| screen-pattern | 0.0534 | 0.0523 | 0.0521 | 0.0571 | 1.021× | 3.63 / 6.09 |
| palette-pattern | 0.0564 | 0.0551 | 0.0548 | 0.0594 | 1.024× | 3.78 / 6.09 |
| noise16 | 0.5570 | 0.5120 | 0.5197 | 0.5216 | 1.088× | 13.46 / 18.03 |
| alpha16-ramp | 0.4110 | 0.3998 | 0.3972 | 0.3907 | 1.028× | 15.31 / 25.36 |
| photo-large-1029 | 39.2895 | 37.1231 | 37.7953 | 36.9020 | 1.058× | 785.42 / 1180.65 |

### Stage attribution

These are sums of per-image stage medians; they need not sum exactly to the sum of total medians. Feature preparation is nested in training, and GPU token generation is nested in tokens. Do not add those diagnostics again. The older learning_calls/population/samples profiler fields are not instrumented for this learner and remain zero; validation uses the current weighted/reused prediction sample counters, plus exact complete codestreams.

| Stage (s) | CPU | Metal-auto | Metal-cpu | Metal-gpu |
|---|---:|---:|---:|---:|
| input_wall | 0.0144 | 0.0144 | 0.0145 | 0.0144 |
| transforms_wall | 7.7769 | 7.7704 | 7.8284 | 7.7774 |
| training_wall | 52.8179 | 52.6052 | 53.4769 | 52.4003 |
| tokens_wall | 3.9523 | 0.1513 | 0.1491 | 0.1409 |
| model_wall | 3.1984 | 3.1639 | 3.1874 | 3.1778 |
| emission_wall | 0.6791 | 0.6699 | 0.6700 | 0.6691 |
| assembly_wall | 0.0018 | 0.0018 | 0.0022 | 0.0022 |
| gpu_input_tokens_wall | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| feature_seconds | 0.0000 | 0.2078 | 1.0329 | 0.1994 |
| token_seconds | 0.0000 | 0.1384 | 0.1359 | 0.1273 |

### First call per process

Each first Metal call includes lazy backend initialization. Device/OS shader caches were not cleared. Differences from steady medians also include ordinary allocator/cache warming and desktop variation, so they are not isolated setup costs.

| Image | First CPU (s) | First Metal-auto (s) | Steady Metal-auto median (s) |
|---|---:|---:|---:|
| photo-macan | 0.5878 | 0.5669 | 0.5373 |
| photo-ria | 0.5495 | 0.5758 | 0.5321 |
| photo-bliznaca | 0.5743 | 0.5745 | 0.5496 |
| photo-flower | 7.8904 | 7.4118 | 7.4939 |
| gray-macan | 0.1860 | 0.2154 | 0.1756 |
| alpha-ria | 0.7938 | 0.7739 | 0.7739 |
| alpha-flower | 10.9191 | 10.0313 | 10.4516 |
| chessboard | 0.9703 | 0.9852 | 0.9204 |
| splines | 4.7028 | 4.4852 | 4.5559 |
| gray-patches | 0.2391 | 0.2675 | 0.2294 |
| palette-358 | 0.0125 | 0.0447 | 0.0123 |
| gray16-ramp | 0.0494 | 0.0815 | 0.0479 |
| screen-pattern | 0.0537 | 0.0834 | 0.0523 |
| palette-pattern | 0.0573 | 0.0871 | 0.0551 |
| noise16 | 0.5432 | 0.5311 | 0.5120 |
| alpha16-ramp | 0.4127 | 0.4300 | 0.3998 |
| photo-large-1029 | 39.6939 | 37.0056 | 37.1231 |

### Disturbed-case repeat

The primary alpha-flower capture had two marked timing disturbances: a CPU call took 41.5 s wall / 15.7 s process CPU, and a later Metal-cpu call took 24.2 s wall / 12.4 s process CPU. The cause is undetermined. All original trials remain in the primary tables and aggregates. After the suite finished, the complete image comparison was repeated in a fresh process with the same binary, input and round/order policy. No individual trials were discarded.

The repeat added 22 exact, independently decoded encodes. Replacing only this image's medians gives the sensitivity aggregate below; it is reported separately from the original aggregate.

| Variant | Repeated image median (s) | Sensitivity aggregate (s) |
|---|---:|---:|
| cpu | 11.9967 | 69.1558 |
| metal-auto | 11.1987 | 65.1692 |
| metal-cpu | 11.1978 | 66.0603 |
| metal-gpu | 12.8109 | 66.5892 |

With the repeated image, Metal-auto uses 5.76% less aggregate elapsed time than CPU. Forced GPU preparation uses 0.80% more time than forced CPU preparation. That small incremental whole-encoder difference is sensitive to desktop variation; it should not be treated as a stable speedup. The isolated feature-operation measurements provide the clearer evidence for producer choice.

Repeat capture: `/Users/yunhocho/GitHub/gjxl-modular/build/learned-metal-qualification-20261015-alpha-repeat`; manifest SHA-256 `44bbbda6c42eefffdc55388fc3158e3ea0aad19564aa24889e223cd9b3bbfb98`.

## Forced feature preparation

Each case uses channels and per-stream WP parameters selected by the current learned workflow. Preparation includes buffer allocation, sample copies, predictor computation, submission/wait and host-view publication. Every output prediction and property-15 value is compared with the independent serial producer outside the timed interval. Forced GPU still uses CPU for unsupported channel shapes.

| Image | Samples | Jobs | CPU (ms) | GPU (ms) | Auto (ms) | GPU speedup | Auto GPU samples | Unsupported samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| photo-macan | 750,000 | 12 | 9.935 | 5.404 | 9.871 | 1.84× | 0 | 0 |
| photo-ria | 750,000 | 12 | 10.274 | 3.648 | 10.277 | 2.82× | 0 | 0 |
| photo-bliznaca | 750,000 | 12 | 10.372 | 6.375 | 10.564 | 1.63× | 0 | 0 |
| photo-flower | 10,287,648 | 162 | 142.457 | 17.982 | 14.935 | 7.92× | 10,287,648 | 0 |
| gray-macan | 250,000 | 4 | 3.335 | 6.239 | 3.345 | 0.53× | 0 | 0 |
| alpha-ria | 1,000,000 | 16 | 13.690 | 4.634 | 13.729 | 2.95× | 0 | 0 |
| alpha-flower | 13,716,864 | 216 | 191.630 | 25.057 | 19.292 | 7.65× | 13,716,864 | 0 |
| chessboard | 1,316,754 | 37 | 16.329 | 7.577 | 16.277 | 2.16× | 0 | 4,572 |
| splines | 10,621,695 | 177 | 124.918 | 19.752 | 16.805 | 6.32× | 10,618,575 | 3,120 |
| gray-patches | 280,133 | 9 | 3.374 | 3.250 | 3.446 | 1.04× | 0 | 0 |
| palette-358 | 20,854 | 6 | 0.239 | 1.696 | 0.235 | 0.14× | 0 | 0 |
| gray16-ramp | 131,841 | 6 | 1.640 | 5.988 | 1.666 | 0.27× | 0 | 0 |
| screen-pattern | 131,889 | 7 | 1.549 | 2.402 | 1.560 | 0.64× | 0 | 0 |
| palette-pattern | 131,969 | 7 | 1.590 | 2.080 | 1.536 | 0.76× | 0 | 0 |
| noise16 | 395,523 | 18 | 5.335 | 3.173 | 5.436 | 1.68× | 0 | 0 |
| alpha16-ramp | 527,364 | 24 | 6.370 | 3.988 | 6.509 | 1.60× | 0 | 0 |
| photo-large-1029 | 36,000,000 | 576 | 488.147 | 45.580 | 39.218 | 10.71× | 36,000,000 | 0 |
| photo-large-1039 | 31,344,224 | 538 | 422.068 | 41.054 | 34.601 | 10.28× | 31,325,174 | 19,050 |
| photo-large-1049 | 29,871,324 | 528 | 414.699 | 42.833 | 37.010 | 9.68× | 29,868,813 | 2,511 |
| rgb8-128x128 | 49,152 | 3 | 0.687 | 1.363 | 0.688 | 0.50× | 0 | 0 |
| rgb8-256x256 | 196,608 | 3 | 2.895 | 2.900 | 2.875 | 1.00× | 0 | 0 |
| rgb8-512x512 | 786,432 | 12 | 11.194 | 6.512 | 11.295 | 1.72× | 0 | 0 |
| rgb8-1024x1024 | 3,145,728 | 48 | 44.379 | 9.151 | 43.999 | 4.85× | 0 | 0 |
| rgb8-1200x1200 | 3,961,628 | 71 | 46.720 | 10.764 | 46.779 | 4.34× | 0 | 0 |
| rgb8-2048x2048 | 12,582,912 | 192 | 183.167 | 22.394 | 19.867 | 8.18× | 12,582,912 | 0 |
| gray8-2048x2048 | 4,194,304 | 64 | 50.840 | 11.069 | 11.367 | 4.59× | 4,194,304 | 0 |
| rgb8-1x65536 | 196,608 | 768 | 3.045 | 6.141 | 3.098 | 0.50× | 0 | 0 |
| rgb8-65536x1 | 196,608 | 768 | 2.667 | 4.328 | 2.680 | 0.62× | 0 | 0 |
| rgb8-17x8192 | 417,792 | 96 | 5.120 | 6.441 | 5.201 | 0.79× | 0 | 0 |
| rgb8-8192x17 | 417,792 | 96 | 5.213 | 4.066 | 5.229 | 1.28× | 0 | 0 |
| rgb8-17x262144 | 4,508,672 | 2,048 | 53.399 | 13.721 | 53.682 | 3.89× | 0 | 0 |

## Interpretation

Reusing the existing GPU weighted predictor is worthwhile. On the three 12 MP photos, selected-channel feature preparation is 9.7–10.7× faster than the serial CPU producer, including allocations, copies and synchronization. Complete-file parity and independent decoding hold with either producer.

The current 4M-sample cutoff is conservative for this serial learned path. The 750k-sample photos gain 1.6–2.8× in preparation, and the 3.15M-sample RGB case gains 4.85× while Auto still chooses CPU. Average job size also needs qualification: the large narrow 17×262144 input gains 3.89× despite being rejected by the 16,384-samples-per-job preference. Total work and number/shape of independent jobs both matter. A new cutoff is not selected by this study.

Unconditional GPU preparation is unsuitable. The small palette map takes 0.24 ms on CPU versus 1.70 ms on GPU; the 250k-sample gray photo takes 3.34 ms versus 6.24 ms. Narrow inputs also show different crossovers. GPU and Auto execute the same feature jobs on eligible large batches; timing differences between those columns reflect runtime variation, not another implementation.

Most of the complete-encoder gain comes from token work falling from a summed 3.95 s to 0.15 s. On explicit Metal, the learning stage still accounts for about 82% of the summed total and transform/selection work about 12%. Faster weighted application does not remove either the fitter or the WP-selection sweep.

Keep explicit Metal available and keep Automatic on CPU for now. The measured speed benefit comes with retained memory: the 12 MP encoder case rises from 785 MiB to 1,181 MiB of managed peak backing. Retuning feature-producer selection is a separate, promising follow-up; changing the public backend default needs a policy that considers small/cold calls and the memory tradeoff.

## Evidence and limits

Raw capture: `/Users/yunhocho/GitHub/gjxl-modular/build/learned-metal-qualification-20261015`. Manifest SHA-256: `6621b0eec96f25441bd6acf8e7d3dcba4e2d519f87ab605e59ea07af94d62932`. Binary SHA-256: `5a7dd04811520e9118d3957910a92a8191964e788be0b7fa3778eaa67d42f8d4`.

The capture retains frozen input hashes, source snapshot, source patch, executable, build settings, every trial, one exact output per encoder input, and power/thermal/process snapshots before and after each case. The analyzer checks hashes, coverage, exactness flags, sample reuse, dispatch counts and released resources.

The [JSON ledger](modular-learned-metal-qualification-results.json) retains every per-image median, range and measured observation. Reproduction uses `gjxl_modular_metal_workflow_profile` (Release Metal build with the pinned Modular oracle), `tools/modular_reference_learning/run_metal_workflow_qualification.py` and `tools/modular_reference_learning/report_metal_workflow_qualification.py`. Both scripts expose their inputs and output destinations through `--help`; the exact command for every captured process is in its manifest.

This is an Apple M4 Pro 14-core, 48 GiB, macOS 27.0.1 qualification on AC power with an active desktop. Background OS work is present. Small end-to-end changes need to be judged against the recorded spread; these results do not qualify other devices, concurrent encodes, all possible image layouts or process RSS.
