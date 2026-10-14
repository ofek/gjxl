#!/usr/bin/env python3
"""Validate and summarize repeated complete-search CPU/Metal-ANS captures."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def distribution(values):
    return dict(median=statistics.median(values), minimum=min(values),
                maximum=max(values), values=values)


def analyze(capture, baseline):
    manifest = json.loads((capture / 'manifest.json').read_text())
    control = json.loads((baseline / 'manifest.json').read_text())
    require(manifest.get('finished_utc') and not manifest.get('interrupted_utc'),
            'Capture is incomplete or interrupted')
    require(not manifest['tracked_status'], 'Capture started from a dirty revision')
    require(manifest['libjxl_source'] == control['libjxl_source'], 'Decoder source provenance differs')
    hashes = json.loads((capture / 'sha256-manifest.json').read_text())
    actual_files = {str(p.relative_to(capture)) for p in capture.rglob('*')
                    if p.is_file() and p.name != 'sha256-manifest.json'}
    require(actual_files == set(hashes), 'Artifact inventory differs')
    for name, entry in hashes.items():
        path = capture / name
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                f'Artifact changed: {name}')
    require(sha(Path(manifest['binary']['path'])) == manifest['binary']['sha256'],
            'Frozen binary differs')
    records = {r['tag']: r for r in manifest['commands']}
    require(len(records) == len(manifest['commands']) == 21, 'Expected 21 unique captures')
    expected_order = [name + '-gpu-ans' for name in
                      ('photo-macan', 'palette-pattern', 'alpha16-ramp')]
    rotations = (('1029', '1039', '1049'), ('1039', '1049', '1029'),
                 ('1049', '1029', '1039'))
    for repetition, images in enumerate(rotations, 1):
        for position, image in enumerate(images):
            order = ('gpu-ans', 'cpu') if (repetition + position) % 2 else ('cpu', 'gpu-ans')
            expected_order.extend(f'{image}-r{repetition}-{route}' for route in order)
    require(list(records) == expected_order, 'Pair rotation/order differs')

    for record in records.values():
        r = record['result']
        tag = record['tag']
        require(record['returncode'] == 0 and record['prior_bytes_equal'] and
                record['prior_policy_equal'] and r['decoded_exact'], f'Parity failed: {tag}')
        require(json.loads((capture / (tag + '.stdout')).read_text()) == r,
                f'Stdout and manifest differ: {tag}')
        output = capture / (tag + '.jxl')
        require(output.stat().st_size == r['encoded_bytes'] and
                sha(output) == record['output_sha256'], f'Output differs: {tag}')
        require((r['maximum_samples'], r['maximum_leaves'], r['checkpoints'],
                 r['profile_enabled'], r['entropy']) == (262144, 256, True, True, 'ans'),
                f'Policy budget differs: {tag}')
        require(r['peak_backing'] <= r['planned_capacity'] and
                r['peak_cpu_participants'] <= r['threads'], f'Resource bound exceeded: {tag}')
        require(r['stages']['gpu_input_tokens'] == 0 and
                sum(r['training']['stages'].values()) <= r['stages']['training'] + 1e-6,
                f'Timer nesting differs: {tag}')
        g = r['adaptive_gpu']
        if tag.endswith('-cpu'):
            require(r['route'] == 'cpu-complete-search' and
                    r['gpu_submissions'] == r['gpu_allocations'] == 0 and
                    all(v == 0 for v in g.values()), f'CPU route differs: {tag}')
        else:
            require(r['route'] == 'hybrid-ans-complete-search' and
                    g['encoded_candidates'] == r['candidates'] > 0 and
                    g['cpu_fallback_candidates'] == 0 and
                    r['gpu_submissions'] == g['encoded_candidates'] + g['ans_gpu_groups'],
                    f'GPU route differs: {tag}')
            require(g['feature_preparations'] + g['feature_reuses'] == g['encoded_candidates'] and
                    g['feature_preparations'] == g['evictions'] + 1 and
                    g['feature_seconds'] + g['token_seconds'] <= r['stages']['tokens'] + 1e-6,
                    f'Feature accounting differs: {tag}')

    smoke = []
    for name in ('photo-macan', 'palette-pattern', 'alpha16-ramp'):
        r = records[name + '-gpu-ans']['result']
        g = r['adaptive_gpu']
        require(r['threads'] == 4 and g['ans_gpu_groups'] == g['ans_gpu_states'] == 0 and
                g['ans_cpu_fallback_groups'] == g['ans_cpu_small_groups'] >= r['candidates'],
                f'Small-section CPU crossover differs: {name}')
        smoke.append(dict(image=name, encoded_bytes=r['encoded_bytes'], seconds=r['seconds'],
                          exact_ans_seconds=r['entropy_work']['ans_token_cost_ns']/1e9,
                          adaptive_gpu=g, prior_bytes_equal=True, decoded_exact=True))

    rows = []
    work_keys = ('learning_calls', 'population', 'samples', 'split_evaluations',
                 'sorted_properties', 'sorted_samples')
    for image in ('1029', '1039', '1049'):
        source = manifest['inputs'][image]
        require(source['sha256'] == control['inputs'][image]['sha256'] == sha(Path(source['path'])),
                f'Input differs: {image}')
        prior = next(r for r in control['commands'] if r['tag'] == image + '-gjxl-adaptive')
        reference_hash = sha(baseline / (image + '-gjxl-adaptive.jxl'))
        require(reference_hash == prior['output_sha256'], 'Prior CPU file differs')
        pairs = []
        for repetition in (1, 2, 3):
            c = records[f'{image}-r{repetition}-cpu']
            h = records[f'{image}-r{repetition}-gpu-ans']
            cpu, hybrid = c['result'], h['result']
            for record in (c, h):
                r = record['result']
                require(record['paired_bytes_policy_training_equal'] and
                        record['image'] == image and record['repetition'] == repetition and
                        record['input_sha256'] == source['sha256'] and
                        record['output_sha256'] == reference_hash, f'Pair provenance differs: {image}')
                require((r['width'], r['height'], r['channels'], r['bits'], r['threads']) ==
                        (4000, 3000, 3, 8, 1), f'Paired geometry/participants differ: {image}')
                require(r['candidates'] == prior['result']['candidates'] and
                        all(r['training'][key] == prior['result']['training'][key] for key in work_keys),
                        f'Training/candidate work differs: {image}')
                for key in ('rct', 'transforms', 'nodes'):
                    require(r['policy'][key] == prior['result']['policy'][key], 'Prior policy differs')
                for key, default in (('weighted_coefficients', [16, 10, 7, 7, 7, 0, 0]),
                                     ('weighted_weights', [13, 12, 12, 12])):
                    require(r['policy'][key] == prior['result']['policy'].get(key, default),
                            'Prior weighted parameters differ')
            require(cpu['policy'] == hybrid['policy'], 'Full paired policies differ')
            g = hybrid['adaptive_gpu']
            require(g['ans_gpu_groups'] >= hybrid['candidates'] and
                    g['ans_gpu_states'] >= g['ans_gpu_groups'] and
                    g['ans_cpu_fallback_groups'] == g['ans_cpu_small_groups'] == 0,
                    f'Full-resolution GPU cost route differs: {image}')
            pairs.append(dict(repetition=repetition, cpu_seconds=cpu['seconds'],
                              hybrid_seconds=hybrid['seconds'], ratio=cpu['seconds']/hybrid['seconds'],
                              cpu={k: v for k, v in cpu.items() if k != 'policy'},
                              hybrid={k: v for k, v in hybrid.items() if k != 'policy'}))
        references = {}
        for effort in (7, 9):
            tag = f'{image}-libjxl-default-e{effort}'
            ref = next(r for r in control['commands'] if r['tag'] == tag)
            path = baseline / (tag + '.jxl')
            require(ref['decoded_exact'] and sha(path) == ref['output_sha256'] and
                    path.stat().st_size == ref['output_bytes'], f'libjxl reference differs: {tag}')
            references[f'e{effort}'] = ref['output_bytes']
        row = dict(image=image, encoded_bytes=pairs[0]['cpu']['encoded_bytes'],
                   output_sha256=reference_hash, libjxl_bytes=references, pairs=pairs)
        for key in ('cpu_seconds', 'hybrid_seconds', 'ratio'):
            row[key] = distribution([p[key] for p in pairs])
        for route in ('cpu', 'hybrid'):
            row[route + '_stages'] = {
                stage: distribution([p[route]['stages'][stage] for p in pairs])
                for stage in pairs[0][route]['stages']}
            row[route + '_exact_ans_seconds'] = distribution(
                [p[route]['entropy_work']['ans_token_cost_ns']/1e9 for p in pairs])
            row[route + '_resources'] = {
                key: distribution([p[route][key] for p in pairs])
                for key in ('peak_backing', 'planned_capacity', 'backend_setup_seconds')}
        rows.append(row)

    totals = dict(encoded_bytes=sum(row['encoded_bytes'] for row in rows))
    for route in ('cpu', 'hybrid'):
        totals[route + '_sum_median_seconds'] = sum(row[route + '_seconds']['median'] for row in rows)
        totals[route + '_stages_sum_medians'] = {
            stage: sum(row[route + '_stages'][stage]['median'] for row in rows)
            for stage in rows[0][route + '_stages']}
        totals[route + '_exact_ans_sum_median_seconds'] = sum(
            row[route + '_exact_ans_seconds']['median'] for row in rows)
    totals['ratio_of_sum_medians'] = (totals['cpu_sum_median_seconds'] /
                                     totals['hybrid_sum_median_seconds'])
    totals['libjxl_bytes'] = {key: sum(row['libjxl_bytes'][key] for row in rows) for key in ('e7', 'e9')}
    totals['size_increase_percent'] = {
        key: 100*(totals['encoded_bytes']/size - 1) for key, size in totals['libjxl_bytes'].items()}
    return dict(schema=1, revision=manifest['revision'], binary=manifest['binary'],
                libjxl_revision=control['libjxl_source']['.']['revision'],
                artifact_root=str(capture.resolve()), artifacts_verified=len(hashes),
                started_utc=manifest['started_utc'], finished_utc=manifest['finished_utc'],
                qualification='Three alternating CPU/hybrid pairs per image on this M4 Pro desktop; '
                              'no general throughput or libjxl speedup claim',
                smoke=smoke, images=rows, totals=totals)


def markdown(result):
    t = result['totals']
    lines = ['# Adaptive Modular exact-ANS GPU qualification', '',
             'The complete hybrid search preserves every CPU file and winning policy across three '
             'repeated pairs on each of the original full-resolution inputs. All full-resolution '
             'candidates use Metal tokens and exact ANS cost evaluation, with no fallback. '
             'Pinned libjxl decodes every output sample exactly.', '',
             f'Frozen implementation: `{result["revision"]}`. '
             f'Artifacts: `{result["artifact_root"]}`; {result["artifacts_verified"]} verified files.', '',
             f'Pinned libjxl: `{result["libjxl_revision"]}`. '
             'Machine-readable results: [gpu-adaptive-modular-ans-results.json](gpu-adaptive-modular-ans-results.json).', '',
             '## Timing boundary', '',
             'Both routes use one CPU participant, the original 262144-sample/256-leaf learner, '
             'tree checkpoints and all retained baseline candidates. Each repetition rotates image '
             'order and alternates CPU/hybrid pair order. Encoder-call timing includes transforms, '
             'learning, feature-cache preparation, every candidate, model upload, GPU waits, CPU '
             'emission and publication. Backend initialization is outside that timer and reported '
             'separately; file reading, writing and independent decoding are also outside it. '
             'There is no sampling profiler. Background desktop activity and between-pair '
             'power/thermal snapshots are retained.', '',
             '| Image | CPU seconds, median [range] | Hybrid seconds, median [range] | Paired ratio, median [range] |',
             '|---|---:|---:|---:|']
    for row in result['images']:
        cells = [row['image']]
        for key in ('cpu_seconds', 'hybrid_seconds', 'ratio'):
            d = row[key]
            cells.append(f'{d["median"]:.3f} [{d["minimum"]:.3f}–{d["maximum"]:.3f}]')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f'The ratio of summed per-image median CPU and hybrid times is '
              f'**{t["ratio_of_sum_medians"]:.2f}×** '
              f'({t["cpu_sum_median_seconds"]:.3f} / {t["hybrid_sum_median_seconds"]:.3f} seconds). '
              'Paired ratios above are computed from each actual adjacent pair. This is a local '
              'three-image qualification on the captured Apple M4 Pro desktop, not an uncontended '
              'benchmark or a comparison of equal-budget GJXL/libjxl throughput.', '',
              '## Attribution', '',
              'The following values sum the three per-image stage medians. Each stage median is '
              'computed independently, so the table need not sum exactly to the complete-call median.', '',
              '| Caller stage | CPU seconds | Hybrid seconds |', '|---|---:|---:|']
    for stage in t['cpu_stages_sum_medians']:
        if stage != 'gpu_input_tokens':
            lines.append(f'| {stage} | {t["cpu_stages_sum_medians"][stage]:.3f} | '
                         f'{t["hybrid_stages_sum_medians"][stage]:.3f} |')
    lines += ['', f'Within `model`, exact ANS cost changes from '
              f'{t["cpu_exact_ans_sum_median_seconds"]:.3f} to '
              f'{t["hybrid_exact_ans_sum_median_seconds"]:.3f} seconds. The hybrid interval includes adapter '
              'preparation, model upload, dispatch, wait and cost conversion; it is not kernel-only '
              'time. Do not add this nested interval to the table. Full per-run training, entropy '
              'and GPU counters are retained in the companion JSON.', '',
              'Tree learning, model construction/normalization, weighted feature preparation and '
              'bitstream emission remain CPU work. The GPU preserves the selected algorithm and '
              'each section\'s ordered ANS recurrence. It prepares tokens within a SIMD group and '
              'processes independent section/model recurrences concurrently.', '',
              '## Compression and resources', '',
              '| Image | GJXL bytes | Saved libjxl e7 bytes | Saved libjxl e9 bytes | CPU / hybrid peak MiB | CPU / hybrid admission MiB | Hybrid setup seconds, median |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['images']:
        c, h = row['cpu_resources'], row['hybrid_resources']
        lines.append(f'| {row["image"]} | {row["encoded_bytes"]:,} | '
                     f'{row["libjxl_bytes"]["e7"]:,} | {row["libjxl_bytes"]["e9"]:,} | '
                     f'{c["peak_backing"]["maximum"]/2**20:.1f} / {h["peak_backing"]["maximum"]/2**20:.1f} | '
                     f'{c["planned_capacity"]["maximum"]/2**20:.1f} / {h["planned_capacity"]["maximum"]/2**20:.1f} | '
                     f'{h["backend_setup_seconds"]["median"]:.3f} |')
    lines += ['', f'Summed output is {t["encoded_bytes"]:,} bytes, '
              f'{t["size_increase_percent"]["e7"]:.2f}% above saved default libjxl e7 and '
              f'{t["size_increase_percent"]["e9"]:.2f}% above e9. GPU execution preserves the '
              'richer CPU learner\'s rate exactly; it does not disable learning or reduce search. '
              'These reference sizes are verified saved artifacts, not fresh libjxl timing runs. '
              'An end-to-end speedup over libjxl is not established.', '',
              'Peak backing is the maximum observed across repetitions; admission is the '
              'conservative plan. Both measure managed backing, not process RSS or driver memory. '
              'Every call fits admission and releases its resource domain. Feature caching retains '
              'transformed planes and weighted maps; ANS costing adds descriptor/model/result '
              'storage without another token arena.', '',
              '## Small work and limitations', '',
              '| Smoke image | Bytes | Complete seconds | Exact ANS seconds | CPU cost groups |',
              '|---|---:|---:|---:|---:|']
    for row in result['smoke']:
        lines.append(f'| {row["image"]} | {row["encoded_bytes"]:,} | {row["seconds"]:.3f} | '
                     f'{row["exact_ans_seconds"]:.3f} | {row["adaptive_gpu"]["ans_cpu_small_groups"]} |')
    lines += ['', 'The smoke cases use GPU token generation and explicitly counted CPU ANS costing '
              'below 64 nonempty sections. Each matches the saved CPU file and policy and passes '
              'independent decoding. These four-CPU-participant smoke captures are single runs, '
              'not repeated small-image performance qualification. The crossover is conservative; '
              'intermediate sizes, different devices and multi-CPU full-resolution throughput '
              'remain unqualified. Public encoder defaults are unchanged.', '',
              'The earlier one-lane capture was stopped after small-image regressions and is '
              'retained separately. Diagnostic SIMD probes guided the final mapping; none are '
              'mixed into these repeated statistics. See [the operation contract](gpu-ans-token-cost.md) '
              'for exact arithmetic, storage, failure behavior and test coverage.', '',
              'Reproduce the report from the saved artifacts:', '', '```sh',
              'python3 tools/modular_training_study/analyze_adaptive_ans.py \\',
              '  --capture build/adaptive-ans-qualified-20261011 \\',
              '  --baseline build/adaptive-baseline-20261011 \\',
              '  --json docs/gpu-adaptive-modular-ans-results.json \\',
              '  --markdown docs/gpu-adaptive-modular-ans.md', '```', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.capture, args.baseline)
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    args.markdown.write_text(markdown(result))
