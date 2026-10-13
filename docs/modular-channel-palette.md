# Reference single-channel palette capacity

The [palette construction port](modular-reference-palette.md) now also supports
large single-channel dictionaries, through the wire maximum of **70,911
entries**. This closes a prerequisite found while connecting reference
selection: libjxl's 1,024-color setting limits multichannel proposals, but does
not limit channel compaction. Automatic selection remains to be implemented.

In pinned `enc_modular.cc::try_palettes`, the single-channel requested limit is
the smaller of `int(nb_pixels / 16)` and
`int(channel_colors_percent / 100. * (max - min + 1))`. The defaults are 95%
before global transforms and 80% for local compaction. A sufficiently large
16-bit input can therefore request 62,259 colors at the global stage. Capping
that proposal at 1,024 would change the reference policy before fitting begins.

The [qualification ledger](modular-channel-palette-results.json) records the
construction, header, storage and full-file checks. This change establishes
the wider primitive; it makes no automatic-policy, rate or timing claim.

## Implementation

Single-channel construction retains ascending unique values and exact indices.
Requests through 1,024 keep the earlier fixed dictionary. Larger requests use
a separate channel dictionary:

- A value span of at most 65,536 uses an 8 KiB membership bitmap and 4 KiB of
  per-word ranks. This includes unsigned 16-bit values and shifted signed
  spans. Discovery/ranking needs no managed scratch allocation.
- Wider signed spans use a reserved value array and an open-addressed index
  table. Values are sorted and the index table rebuilt before emission. Both
  arrays are bounded and released before returning the candidate.

The wider dictionary object is under 13 KiB on the stack. The planner reserves
the worst sparse scratch without reading pixels. For
`capacity = min(requested_limit, source_pixels)`, its managed peak is bounded by
`4 * capacity + 4 * bit_ceil(2 * capacity)` bytes, with zero retained scratch.
Output records/planes are additional. Borrowed input, a previous output,
allocator overhead and stack objects are excluded; this is not an RSS bound.

Large sparse proposals can allocate scratch even when too many colors cause
policy rejection. Errors retain the previous output; a successful rejection
publishes an empty, unapplied candidate. Allocation tests cover that distinction
and the prescribed forward-transform path, including aliased input/output.
The whole-frame storage planner now includes forward palette scratch as well
as replacement images. Prepared-stream coding and inverse transforms do not
construct dictionaries and need no such scratch.

The descriptor's palette count is now 32-bit so that 65,536 entries cannot
truncate to zero. The header writer supports all four reference count ranges:
0–255, 256–1,279, 1,280–5,375 and 5,376–70,911. Its existing conservative bit
bound covers the larger fields. Multichannel reference construction remains
limited to 1,024 requested entries plus up to 189 promoted implicit entries;
legacy explicit-capacity palettes keep their previous limit and output.

## Qualification

Both Release and ASan/UBSan pass all **15 selected CTest entries**.

| Check | Result per build |
|---|---:|
| New wide construction cases | 34 |
| Accepted / rejected wide proposals | 22 / 12 |
| Injected failures in wide discovery/forward construction | 44 |
| Maximum observed wide candidate managed peak | 1,899,964 bytes |
| Headers parsed exactly by pinned libjxl | 160 |
| New complete files decoded exactly | 64 |
| Prior construction cases retained | 876 |
| Prior files unchanged | 820 / 820 |
| Total files identical across Release/sanitizer | 884 |

The wide cases compare acceptance, dictionaries, indices and both inverses
with actual pinned `FwdPalette`/`InvPalette`, including the 70,911-entry
construction maximum. They cover dense and sparse signed ranges, endpoints
at `INT32_MIN`/`INT32_MAX`, a tiny dictionary with a large requested limit,
strided/shifted input, existing metadata, policy rejection and allocation
failure/recovery. The existing palette fault sweeps also expand from 70 to
82 failures by checking prescribed forward publication for the earlier
1,213-entry multichannel cases.

Header checks cover all bit offsets, shared/local-tree flags and count
boundaries through 70,911. The 64 new Gray16 files cover eight dictionary
widths: 1,025, 1,279, 1,280, 5,375, 5,376, 62,259, 65,535 and 65,536. They use
the whole-frame encoder and prepared streams with shared/local trees across
Prefix, ANS-from-Prefix and estimated direct ANS controls. Full-file coverage
ends at 65,536 entries; the larger wire maximum is qualified by direct
construction/inverse and header controls.

The 172 earlier palette files and 648 prepared-stream files match the previous
palette milestone byte for byte. Gradient/WP estimator controls, stream/image
contracts, ordered transforms and histogram preparation still pass. The
validation freeze binds **5,077 artifacts**. Native code and test adapters are
sanitized; linked reference archives remain uninstrumented, reference vptr
checks are excluded, and macOS LeakSanitizer is disabled. Managed backing leak
checks pass. Overlapping correctness runs are not timing measurements.

## Reproduction and remaining work

The additional tests are `modular_reference_channel_palette` and
`modular_reference_channel_palette_oracle`. The latter links the pristine
pinned libjxl build. Recheck the frozen evidence with a new output path:

```sh
python3 tools/modular_reference_learning/analyze_channel_palette.py \
  --validation build/channel-palette-validation \
  --baseline-validation build/reference-palette-validation \
  --output /tmp/gjxl-channel-palette-rechecked.json
```

`build/channel-palette-validation/freeze.py` records the capture procedure.
The source snapshot is authoritative for the pre-commit build.

Next reproduce palette proposal ordering and cost acceptance, the 19-RCT and
five-WP choices, and global/local stream scope on matched original inputs.
Then connect native preparation and one fit per selected scope. The exhaustive
new-learner comparison, whole-stage timing, held-out rate, resource gates and
the separate high-depth LZ77 issue remain open under the
[selection plan](modular-reference-search-policy.md).
