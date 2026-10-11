#!/usr/bin/env python3
"""Render the audited pipeline results without running encoders."""
import argparse
import json
from pathlib import Path

LABELS = {
    'original': 'Original exact ANS', 'pre_reuse': 'Estimated ANS before reuse',
    'reuse': 'Reuse prefix populations', 'emit': 'Current exhaustive emission + ledger',
    'measure': 'Exhaustive exact sizing', 'shortlist': 'Four-candidate shortlist',
    'population': 'GPU populations + exact sizing', 'combined': 'GPU populations + shortlist'}


def render(results):
    if not results['qualified_time_comparisons']:
        raise RuntimeError('Power-interrupted comparisons need whole-comparison rechecks before reporting qualified times')
    sums = results['sum_large_medians']
    stages = results['mean_large_stages']
    cases = [c for c in results['candidate_cases'] if c['repetition'] == 1]
    large = [c for c in cases if c['image'] in ('1029', '1039', '1049')]
    large_rate = (sum(c['shortlist_bytes'] for c in large) / sum(c['base_bytes'] for c in large) - 1) * 100
    worst = results['worst_regret']
    lines = [
        '# Modular entropy pipeline qualification', '',
        f"The byte-preserving GPU population/exact-sizing route takes **{sums['population'] / 3:.3f} s** "
        f"per original 12 MP image on average, versus **{sums['original'] / 3:.3f} s** for the frozen "
        f"original exact-ANS route: **{sums['original'] / sums['population']:.3f}x** on this pilot.", '',
        f"The combined shortlist route takes **{sums['combined'] / 3:.3f} s**, "
        f"or **{sums['original'] / sums['combined']:.3f}x** relative to that same directly measured control. "
        f"Its aggregate large-image byte change is **{large_rate:+.4f}%**. "
        'The shortlist is a separate rate/time trade-off and remains private.', '',
        '## Complete-call timing', '',
        'Each large-image entry is the median of two serial complete calls. Aggregate ratios use the '
        'sum of the three image medians. Model/emission columns are means of the corresponding image '
        'stage medians, already included in complete time. Backend setup, independent decode and file '
        'I/O are outside the encoder timer. No isolated speedups are multiplied.', '',
        '| Mode | Mean complete s | Mean model s | Mean emission s | Ratio to original |',
        '|---|---:|---:|---:|---:|']
    for v, label in LABELS.items():
        lines.append(f"| {label} | {sums[v] / 3:.3f} | {stages[v]['model']:.3f} | "
                     f"{stages[v]['emission']:.3f} | {sums['original'] / sums[v]:.3f}x |")
    lines += ['', f"Population reuse alone changes the directly measured estimated control by "
        f"{sums['pre_reuse'] / sums['reuse']:.3f}x. Exact candidate sizing versus current exhaustive "
        f"emission is {sums['emit'] / sums['measure']:.3f}x. GPU populations versus exact sizing "
        f"with CPU populations is {sums['measure'] / sums['population']:.3f}x. "
        'These comparisons have different controls and must not be combined arithmetically.', '',
        '| Image | Original exact s (range) | GPU exact sizing s (range) | Combined s (range) |',
        '|---|---:|---:|---:|']
    for image, variants in results['large'].items():
        cells = []
        for v in ('original', 'population', 'combined'):
            r = variants[v]
            cells.append(f"{r['median_seconds']:.3f} ({min(r['seconds']):.3f}–{max(r['seconds']):.3f})")
        lines.append('| ' + image + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Rate and winner selection', '',
        'Exact sizing and GPU population reuse preserve the estimated exhaustive control’s complete '
        'files and policies on all 19 images. Every exact ledger entry, including shortlisted finalists, '
        'matches actual emission of that candidate. The old exact and estimated frozen controls also '
        'match those exhaustive files on the three original photos.', '',
        f"The four-candidate shortlist recalls the exhaustive winning candidate on "
        f"**{results['winner_recall']}/{results['images']}** images. Across unique images, its aggregate "
        f"byte change is **{results['aggregate_regret_percent']:+.4f}%**; the worst is "
        f"**{worst['image']}: {worst['regret_percent']:+.4f}% ({worst['regret_bytes']:+d} bytes)**. "
        'GPU populations do not change the shortlist’s selected files. All outputs independently decode '
        'to the original samples.', '',
        '| Image | Exhaustive bytes | Shortlist bytes | Change | Winner recalled |',
        '|---|---:|---:|---:|:---:|']
    for c in cases:
        lines.append(f"| {c['image']} | {c['base_bytes']} | {c['shortlist_bytes']} | "
                     f"{c['regret_percent']:+.4f}% | {'yes' if c['winner_recalled'] else 'no'} |")
    lines += ['', '## Remaining work and managed storage', '',
        'The shortlist still performs every learning pass and generates every candidate’s canonical '
        'tokens. It reduces complete-population modeling to at most four finalists plus the final '
        'winner, and emits one file. GPU population production applies only to eligible full models; '
        'sampled screening, small populations, unsupported shapes and wide values retain explicit '
        'bypass/fallback paths.', '',
        '| Combined stage | Mean seconds per 12 MP image |', '|---|---:|']
    for k, seconds in stages['combined'].items():
        lines.append(f'| {k} | {seconds:.3f} |')
    lines += ['', '| Mode | Largest measured managed peak MiB | Largest admission bound MiB |',
              '|---|---:|---:|']
    for v in LABELS:
        peak = max(max(r[v]['managed_peak_bytes']) for r in results['large'].values()) / 2**20
        planned = max(r[v]['planned_capacity'] for r in results['large'].values()) / 2**20
        lines.append(f'| {LABELS[v]} | {peak:.2f} | {planned:.2f} |')
    lines += ['', 'Admission is a conservative capacity reservation, not resident memory use. Peaks '
        'cover managed backing; caller-owned diagnostic strings are outside that accounting. Actual '
        'GPU submissions, population routes, prefix/ANS work counters and per-call bounds are retained '
        'in the audited artifacts.', '', '## Evidence and limits', '',
        f"The independent artifact audit validates **{results['audited_calls']} complete calls**. "
        'The mixed corpus has one call per current mode; only the three original 12 MP inputs have '
        'repeated timing comparisons. This is an active-desktop Apple M4 Pro pilot, not broad device '
        'qualification. Public defaults remain unchanged.', '',
        'See [machine-readable results](modular-entropy-pipeline-results.json), '
        '[candidate evaluation](modular-candidate-evaluation.md), '
        '[GPU population contract](modular-gpu-populations.md), and '
        '[the five-step plan](modular-entropy-efficiency.md). The fixed-token native/libjxl rate study '
        'is separate from full encoder latency. These results do not establish an end-to-end speedup '
        'over libjxl.', '',
        'The measured binaries are frozen independently of later source and audit-tool changes. '
        'A subsequent storage-planner correction covers a search whose largest palette layout '
        'exceeds `UINT32_MAX` tokens while another layout remains eligible. Every measured layout '
        'is below that limit, so the correction does not change these calls or their reservations.', '',
        '| Frozen milestone | Revision | Binary SHA-256 |', '|---|---|---|']
    for v in ('original', 'reuse', 'emit'):
        b = results['binaries'][v]
        lines.append(f"| {LABELS[v]} | `{b['revision']}` | `{b['sha256']}` |")
    lines += ['', 'The original exact and pre-reuse estimated modes share the original binary; '
        'all five current modes share the last binary. Host/power snapshots before and after the '
        'capture are retained in the results JSON. The run began on battery power. A temporary '
        'idle-sleep guard covered the serial calls; the auditor requires uninterrupted heartbeats '
        'and hashes the retained sleep log.', '',
        f"Capture manifest: `{results['manifest']}`. SHA-256: `{results['manifest_sha256']}`.", '',
        '### Power transition and whole-comparison recheck', '']
    for recheck in results['rechecks']:
        lines += [f"The original {recheck['image']} repetition {recheck['repetition']} changed power source "
            'mid-comparison. All eight modes were repeated in the same order under monitored AC power. '
            'The qualified table uses that complete replacement, including its original control; '
            'the raw 128-call capture is retained unchanged. No time is subtracted and no fastest '
            'run is selected.', '',
            f"Recheck manifest: `{recheck['manifest']}`. SHA-256: `{recheck['manifest_sha256']}`.", '']
    lines += ['The two 1029 repetitions span battery and AC operation, with each whole comparison '
        'on one source. The remaining qualified large-image comparisons use AC. These are paired '
        'active-desktop observations with visible timing ranges, not a fixed-power throughput claim.', '',
        results['scope'], '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    args.markdown.write_text(render(json.loads(args.json.read_text())))
