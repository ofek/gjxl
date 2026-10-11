# Modular entropy pipeline qualification

The byte-preserving GPU population/exact-sizing route takes **64.685 s** per original 12 MP image on average, versus **117.678 s** for the frozen original exact-ANS route: **1.819x** on this pilot.

The combined shortlist route takes **63.333 s**, or **1.858x** relative to that same directly measured control. Its aggregate large-image byte change is **+0.3969%**. The shortlist is a separate rate/time trade-off and remains private.

Use GPU populations with exhaustive exact sizing as the next private performance baseline. The combined shortlist reduces measured complete time by only 2.09% beyond it, while its worst corpus size regression is 7.69%. This does not justify automatic shortlist selection.

## Complete-call timing

Each large-image entry is the median of two serial complete calls. Aggregate ratios use the sum of the three image medians. Model/emission columns are means of the corresponding image stage medians, already included in complete time. Backend setup, independent decode and file I/O are outside the encoder timer. No isolated speedups are multiplied.

| Mode | Mean complete s | Mean model s | Mean emission s | Ratio to original |
|---|---:|---:|---:|---:|
| Original exact ANS | 117.678 | 38.446 | 17.187 | 1.000x |
| Estimated ANS before reuse | 106.601 | 27.680 | 16.640 | 1.104x |
| Reuse prefix populations | 95.986 | 17.677 | 16.614 | 1.226x |
| Current exhaustive emission + ledger | 95.298 | 17.473 | 16.434 | 1.235x |
| Exhaustive exact sizing | 83.817 | 21.742 | 0.261 | 1.404x |
| Four-candidate shortlist | 66.941 | 2.306 | 0.247 | 1.758x |
| GPU populations + exact sizing | 64.685 | 4.188 | 0.248 | 1.819x |
| GPU populations + shortlist | 63.333 | 1.113 | 0.264 | 1.858x |

Population reuse alone changes the directly measured estimated control by 1.111x. Exact candidate sizing versus current exhaustive emission is 1.137x. GPU populations versus exact sizing with CPU populations is 1.296x. These comparisons have different controls and must not be combined arithmetically.

| Image | Original exact s (range) | GPU exact sizing s (range) | Combined s (range) |
|---|---:|---:|---:|
| 1029 | 134.320 (113.666–154.974) | 72.215 (61.548–82.881) | 73.225 (61.431–85.019) |
| 1039 | 110.294 (108.798–111.790) | 61.323 (59.127–63.519) | 59.206 (58.873–59.539) |
| 1049 | 108.421 (106.783–110.059) | 60.516 (60.225–60.807) | 57.569 (56.973–58.164) |

## Rate and winner selection

Exact sizing and GPU population reuse preserve the estimated exhaustive control’s complete files and policies on all 19 images. Every exact ledger entry, including shortlisted finalists, matches actual emission of that candidate. The old exact and estimated frozen controls also match those exhaustive files on the three original photos.

The four-candidate shortlist recalls the exhaustive winning candidate on **14/19** images. Across unique images, its aggregate byte change is **+0.3733%**; the worst is **splines: +7.6910% (+22704 bytes)**. GPU populations do not change the shortlist’s selected files. All outputs independently decode to the original samples.

| Image | Exhaustive bytes | Shortlist bytes | Change | Winner recalled |
|---|---:|---:|---:|:---:|
| photo-macan | 260654 | 260912 | +0.0990% | no |
| photo-ria | 211044 | 211044 | +0.0000% | yes |
| photo-bliznaca | 242885 | 242885 | +0.0000% | yes |
| photo-flower | 2609895 | 2611329 | +0.0549% | no |
| gray-macan | 90321 | 90321 | +0.0000% | yes |
| alpha-ria | 237730 | 239801 | +0.8712% | no |
| alpha-flower | 3833995 | 3833995 | +0.0000% | yes |
| chessboard | 62805 | 62805 | +0.0000% | yes |
| splines | 295203 | 317907 | +7.6910% | no |
| gray-patches | 18299 | 18299 | +0.0000% | yes |
| palette-358 | 1046 | 1046 | +0.0000% | yes |
| gray16-ramp | 1292 | 1292 | +0.0000% | yes |
| screen-pattern | 385 | 385 | +0.0000% | yes |
| palette-pattern | 438 | 438 | +0.0000% | yes |
| noise16 | 802311 | 802311 | +0.0000% | yes |
| alpha16-ramp | 19866 | 19866 | +0.0000% | yes |
| 1029 | 10821744 | 10821744 | +0.0000% | yes |
| 1039 | 7230909 | 7331667 | +1.3934% | no |
| 1049 | 7336102 | 7336102 | +0.0000% | yes |

## Remaining work and managed storage

The shortlist still performs every learning pass and generates every candidate’s canonical tokens. It reduces complete-population modeling to at most four finalists plus the final winner, and emits one file. GPU population production applies only to eligible full models; sampled screening, small populations, unsupported shapes and wide values retain explicit bypass/fallback paths.

| Stage | GPU exact sizing s | Combined shortlist s |
|---|---:|---:|
| input | 0.627 | 0.610 |
| transforms | 3.368 | 3.369 |
| training | 37.142 | 36.988 |
| tokens | 19.101 | 20.967 |
| model | 4.188 | 1.113 |
| emission | 0.248 | 0.264 |
| assembly | 0.002 | 0.013 |
| gpu_input_tokens | 0.000 | 0.000 |

Learning and canonical token generation account for 86.9% of the exhaustive GPU-population route. Reducing that repeated work is the next major complete-call opportunity; further entropy micro-optimization has a much smaller ceiling.

| Mode | Largest measured managed peak MiB | Largest admission bound MiB |
|---|---:|---:|
| Original exact ANS | 1307.27 | 6418.38 |
| Estimated ANS before reuse | 1307.27 | 6412.71 |
| Reuse prefix populations | 1128.32 | 6136.74 |
| Current exhaustive emission + ledger | 1128.32 | 6141.05 |
| Exhaustive exact sizing | 1105.15 | 6146.71 |
| Four-candidate shortlist | 1105.18 | 6146.71 |
| GPU populations + exact sizing | 1105.15 | 6225.71 |
| GPU populations + shortlist | 1105.18 | 6225.71 |

Admission is a conservative capacity reservation, not resident memory use. Peaks cover managed backing; caller-owned diagnostic strings are outside that accounting. Actual GPU submissions, population routes, prefix/ANS work counters and per-call bounds are retained in the audited artifacts.

## Evidence and limits

The independent artifact audit validates **136 complete calls**. The mixed corpus has one call per current mode; only the three original 12 MP inputs have repeated timing comparisons. This is an active-desktop Apple M4 Pro pilot, not broad device qualification. Public defaults remain unchanged.

See [machine-readable results](modular-entropy-pipeline-results.json), [candidate evaluation](modular-candidate-evaluation.md), [GPU population contract](modular-gpu-populations.md), and [the five-step plan](modular-entropy-efficiency.md). The [fixed-token native/libjxl rate study](modular-entropy-rate.md) is separate from full encoder latency. These results do not establish an end-to-end speedup over libjxl.

The measured binaries are frozen independently of later source and audit-tool changes. A subsequent storage-planner correction covers a search whose largest palette layout exceeds `UINT32_MAX` tokens while another layout remains eligible. Every measured layout is below that limit, so the correction does not change these calls or their reservations.

| Frozen milestone | Revision | Binary SHA-256 |
|---|---|---|
| Original exact ANS | `cd9d1aa931d8fe9fd0c651018c8409e0f3abdcc5` | `b66d5b4f7257071cebdef5104e9067b9185559712342667813be5f66387d8648` |
| Reuse prefix populations | `e12cdbd205256d9f706dfaaafa18329098535b73` | `95676f273979d1eef055dee7ae48fb979c0086bfd4b2dbfdf47b9156eeb9accd` |
| Current exhaustive emission + ledger | `a31fc4e19068beb8767e6d13b1556a9422549c8c` | `4ae5294cf6ddee6652171faece7b8dc600a9cdf6140b056e8687ed9afa879639` |

The original exact and pre-reuse estimated modes share the original binary; all five current modes share the last binary. Host/power snapshots before and after the capture are retained in the results JSON. The run began on battery power. A temporary idle-sleep guard covered the serial calls; the auditor requires uninterrupted heartbeats and hashes the retained sleep log.

Capture manifest: `/Users/yunhocho/GitHub/gjxl-modular-adaptive-gpu/build/entropy-pipeline-qualified-20261011/manifest.json`. SHA-256: `788d263594598a30e8ddf894293b5ccc02ec3a0e5e3aef3a3a1a297629d519c1`.

### Power transition and whole-comparison recheck

The original 1039 repetition 1 changed power source mid-comparison. All eight modes were repeated in the same order under monitored AC power. The qualified table uses that complete replacement, including its original control; the raw 128-call capture is retained unchanged. No time is subtracted and no fastest run is selected.

Recheck manifest: `/Users/yunhocho/GitHub/gjxl-modular-adaptive-gpu/build/entropy-pipeline-power-recheck-20261011/manifest.json`. SHA-256: `f75bc32fd1e10e24ed4f53104f96fa0b71aa9e1c2f59f00948e95e3991104b4a`.

The two 1029 repetitions span battery and AC operation, with each whole comparison on one source. The remaining qualified large-image comparisons use AC. These are paired active-desktop observations with visible timing ranges, not a fixed-power throughput claim.

Serial complete encoder calls; 16 mixed images once per current mode and 3 original 12MP images twice per mode. Full 262144/256 learner, checkpoints, one CPU participant. No excluded warmup. Backend setup, independent decode, file I/O outside encoder timer. Current modes include candidate ledger serialization; old frozen controls predate that option. Active desktop; no fresh libjxl encoder latency claim.
