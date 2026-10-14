#!/usr/bin/env python3
"""Validate paired complete-search CPU, Metal-ANS and Metal-learner captures."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

IMAGES = ('1029', '1039', '1049')
SMOKE = ('photo-macan', 'palette-pattern', 'alpha16-ramp')
ROUTES = ('cpu', 'gpu-ans', 'gpu-learn')
WORK = ('learning_calls', 'population', 'samples', 'split_evaluations',
        'sorted_properties', 'sorted_samples')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def distribution(values):
    return dict(median=statistics.median(values), minimum=min(values),
                maximum=max(values), values=values)


def same_policy(actual, prior):
    for key in ('rct', 'transforms', 'nodes'):
        require(actual[key] == prior[key], f'Prior {key} differs')
    for key, default in (('weighted_coefficients', [16, 10, 7, 7, 7, 0, 0]),
                         ('weighted_weights', [13, 12, 12, 12])):
        require(actual[key] == prior.get(key, default), f'Prior {key} differs')


def order():
    result = [name + '-gpu-learn' for name in SMOKE]
    routes = ('gpu-learn', 'gpu-ans', 'cpu')
    for repetition, images in enumerate((IMAGES, ('1039', '1049', '1029')), 1):
        for image in images:
            rotation = IMAGES.index(image)
            selected = routes[rotation:] + routes[:rotation]
            if repetition == 2:
                selected = tuple(reversed(selected))
            result.extend(f'{image}-r{repetition}-{route}' for route in selected)
    return result


def check_result(r, route, small):
    require((r['maximum_samples'], r['maximum_leaves'], r['checkpoints'],
             r['profile_enabled'], r['entropy'], r['decoded_exact']) ==
            (262144, 256, True, True, 'ans', True), 'Policy/decoder validation differs')
    require(r['peak_backing'] <= r['planned_capacity'] and
            r['peak_cpu_participants'] <= r['threads'], 'Resource bound exceeded')
    require(r['seconds'] > 0 and r['stages']['gpu_input_tokens'] == 0 and
            sum(r['training']['stages'].values()) <= r['stages']['training'] + 1e-6,
            'Timer boundary differs')
    g = r['adaptive_gpu']
    expected_route = {'cpu': 'cpu-complete-search', 'gpu-ans': 'hybrid-ans-complete-search',
                      'gpu-learn': 'hybrid-learning-complete-search'}[route]
    require(r['route'] == expected_route, 'Execution route differs')
    if route == 'cpu':
        require(r['gpu_submissions'] == r['gpu_allocations'] == 0 and
                all(v == 0 for v in g.values()), 'CPU route used the GPU')
        return
    require(g['encoded_candidates'] == r['candidates'] > 0 and
            g['cpu_fallback_candidates'] == 0 and
            r['gpu_submissions'] == g['encoded_candidates'] + g['ans_gpu_groups'] +
            g['tree_gpu_groups'], 'GPU candidates/submissions differ')
    require(g['feature_preparations'] + g['feature_reuses'] == g['encoded_candidates'] and
            g['feature_preparations'] == g['evictions'] + 1 and
            g['feature_seconds'] + g['token_seconds'] <= r['stages']['tokens'] + 1e-6,
            'Feature accounting differs')
    if small:
        require(g['ans_gpu_groups'] == g['ans_gpu_states'] == 0 and
                g['ans_cpu_fallback_groups'] == g['ans_cpu_small_groups'] >= r['candidates'],
                'Small-section ANS fallback differs')
    else:
        require(g['ans_gpu_groups'] >= r['candidates'] and
                g['ans_gpu_states'] >= g['ans_gpu_groups'] and
                g['ans_cpu_fallback_groups'] == g['ans_cpu_small_groups'] == 0,
                'Full-resolution ANS route differs')
    if route == 'gpu-ans':
        require(all(v == 0 for k, v in g.items() if k.startswith('tree_')),
                'ANS control used GPU learning')
    else:
        require(g['tree_cpu_fallback_properties'] == g['tree_cpu_small_properties'] and
                g['tree_cpu_unsupported_learning_calls'] == 0 and
                g['tree_gpu_groups'] >= g['tree_gpu_preparations'] and
                g['tree_prepare_seconds'] + g['tree_evaluate_seconds'] <=
                r['training']['stages']['split_scanning'] + 1e-6,
                'Learner fallback/accounting differs')
        if not small:
            require(g['tree_gpu_preparations'] > 0 and g['tree_gpu_groups'] > 0 and
                    g['tree_gpu_samples'] > 0, 'Full-resolution GPU learning omitted')


def analyze(capture, baseline):
    manifest = json.loads((capture / 'manifest.json').read_text())
    control = json.loads((baseline / 'manifest.json').read_text())
    require(manifest.get('finished_utc') and not manifest.get('interrupted_utc'),
            'Capture is incomplete or interrupted')
    require(not manifest['tracked_status'], 'Capture started from a dirty revision')
    require(manifest['libjxl_source'] == control['libjxl_source'], 'Decoder provenance differs')
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
    require(len(records) == len(manifest['commands']) == 21 and list(records) == order(),
            'Expected 21 unique captures in the prescribed three-route order')
    for tag, record in records.items():
        r = record['result']
        require(record['returncode'] == 0 and record['prior_bytes_equal'] and
                record['prior_policy_equal'], f'Prior validation failed: {tag}')
        require(json.loads((capture / (tag + '.stdout')).read_text()) == r,
                f'Stdout and manifest differ: {tag}')
        path = capture / (tag + '.jxl')
        require(path.stat().st_size == r['encoded_bytes'] and
                sha(path) == record['output_sha256'], f'Output differs: {tag}')
        small = '-r1-' not in tag and '-r2-' not in tag
        route = next(key for key in ROUTES if tag.endswith('-' + key))
        check_result(r, route, small)
        require(r['threads'] == (4 if small else 1), 'CPU participation setting differs')
        require(sha(Path(record['argv'][1])) == record['input_sha256'], 'Capture input changed')
        require(record['argv'][0] == manifest['binary']['path'] and
                record['argv'][3:6] == ['262144', '256', str(r['threads'])] and
                record['argv'][6:] == ([] if route == 'cpu' else
                                       ['--metal-learn' if route == 'gpu-learn' else '--metal-ans']),
                'Capture arguments differ')

    rows = []
    for image in IMAGES:
        source = manifest['inputs'][image]
        require(source['sha256'] == control['inputs'][image]['sha256'] == sha(Path(source['path'])),
                f'Original input differs: {image}')
        prior = next(r for r in control['commands'] if r['tag'] == image + '-gjxl-adaptive')
        reference_hash = sha(baseline / (image + '-gjxl-adaptive.jxl'))
        require(reference_hash == prior['output_sha256'], 'Prior CPU output changed')
        triplets = []
        for repetition in (1, 2):
            triplet = dict(repetition=repetition)
            policies = []
            for route in ROUTES:
                record = records[f'{image}-r{repetition}-{route}']
                r = record['result']
                require(record['paired_bytes_policy_training_equal'] and
                        record['image'] == image and record['repetition'] == repetition and
                        record['input_sha256'] == source['sha256'] and
                        record['output_sha256'] == reference_hash, 'Triplet provenance differs')
                require((r['width'], r['height'], r['channels'], r['bits']) == (4000, 3000, 3, 8),
                        'Full-resolution geometry differs')
                require(r['candidates'] == prior['result']['candidates'] and
                        all(r['training'][key] == prior['result']['training'][key] for key in WORK),
                        'Original learner/candidate work differs')
                same_policy(r['policy'], prior['result']['policy'])
                policies.append(r['policy'])
                triplet[route] = {k: v for k, v in r.items() if k != 'policy'}
            require(policies[0] == policies[1] == policies[2], 'Full triplet policies differ')
            triplet['cpu_ratio'] = triplet['cpu']['seconds'] / triplet['gpu-learn']['seconds']
            triplet['incremental_ratio'] = triplet['gpu-ans']['seconds'] / triplet['gpu-learn']['seconds']
            triplets.append(triplet)
        references = {}
        for effort in (7, 9):
            tag = f'{image}-libjxl-default-e{effort}'
            record = next(r for r in control['commands'] if r['tag'] == tag)
            path = baseline / (tag + '.jxl')
            require(record['decoded_exact'] and sha(path) == record['output_sha256'] and
                    path.stat().st_size == record['output_bytes'], 'libjxl reference changed')
            references[f'e{effort}'] = record['output_bytes']
        row = dict(image=image, encoded_bytes=prior['result']['encoded_bytes'],
                   output_sha256=reference_hash, libjxl_bytes=references, triplets=triplets)
        for key in ('cpu_ratio', 'incremental_ratio'):
            row[key] = distribution([t[key] for t in triplets])
        row['routes'] = {}
        for route in ROUTES:
            data = [t[route] for t in triplets]
            summary = {key: distribution([r[key] for r in data]) for key in
                       ('seconds', 'peak_backing', 'planned_capacity', 'backend_setup_seconds')}
            for key in ('stages', 'adaptive_gpu'):
                summary[key] = {name: distribution([r[key][name] for r in data])
                                for name in data[0][key]}
            summary['training_stages'] = {
                name: distribution([r['training']['stages'][name] for r in data])
                for name in data[0]['training']['stages']}
            summary['exact_ans_seconds'] = distribution(
                [r['entropy_work']['ans_token_cost_ns'] / 1e9 for r in data])
            row['routes'][route] = summary
        rows.append(row)
    totals = dict(encoded_bytes=sum(r['encoded_bytes'] for r in rows), routes={})
    for route in ROUTES:
        totals['routes'][route] = dict(
            seconds=sum(r['routes'][route]['seconds']['median'] for r in rows),
            exact_ans_seconds=sum(r['routes'][route]['exact_ans_seconds']['median'] for r in rows),
            stages={k: sum(r['routes'][route]['stages'][k]['median'] for r in rows)
                    for k in rows[0]['routes'][route]['stages']},
            training_stages={k: sum(r['routes'][route]['training_stages'][k]['median'] for r in rows)
                             for k in rows[0]['routes'][route]['training_stages']})
    totals['cpu_ratio'] = totals['routes']['cpu']['seconds'] / totals['routes']['gpu-learn']['seconds']
    totals['incremental_ratio'] = (totals['routes']['gpu-ans']['seconds'] /
                                   totals['routes']['gpu-learn']['seconds'])
    totals['libjxl_bytes'] = {k: sum(r['libjxl_bytes'][k] for r in rows) for k in ('e7', 'e9')}
    totals['size_increase_percent'] = {
        k: 100 * (totals['encoded_bytes'] / v - 1) for k, v in totals['libjxl_bytes'].items()}
    smoke_root = baseline.parent / 'adaptive-profile-smoke'
    saved_smoke = json.loads((smoke_root / 'results.json').read_text())
    smoke = []
    for name in SMOKE:
        record = records[name + '-gpu-learn']
        r = record['result']
        prior = next(p for p in saved_smoke
                     if p['profile_enabled'] and Path(p['command'][1]).stem == name)
        require(record['output_sha256'] == sha(smoke_root / (name + '-profile.jxl')) and
                r['encoded_bytes'] == prior['encoded_bytes'] and
                r['candidates'] == prior['candidates'] and
                all(r['training'][key] == prior['training'][key] for key in WORK),
                f'Smoke CPU file or training work differs: {name}')
        same_policy(r['policy'], prior['policy'])
        smoke.append(dict(image=name, result=r))
    return dict(schema=1, revision=manifest['revision'], binary=manifest['binary'],
                libjxl_revision=control['libjxl_source']['.']['revision'],
                artifact_root=str(capture.resolve()), artifacts_verified=len(hashes),
                started_utc=manifest['started_utc'], finished_utc=manifest['finished_utc'],
                qualification='Two rotated/reversed CPU/ANS/learner triplets per image on one M4 Pro desktop; '
                              'local observations, no libjxl or general throughput speedup claim',
                smoke=smoke, images=rows, totals=totals)


def markdown(result):
    totals = result['totals']
    lines = ['# Adaptive Modular GPU learner qualification', '',
             'The complete-search learner route preserves all CPU files, winning policies and '
             'semantic training work across two three-route repetitions on each original '
             'full-resolution input. Pinned libjxl independently decodes every output sample exactly.', '',
             f'Frozen implementation: `{result["revision"]}`. '
             f'Artifacts: `{result["artifact_root"]}`; {result["artifacts_verified"]} verified files. '
             f'Pinned libjxl: `{result["libjxl_revision"]}`.', '',
             'Machine-readable results: [gpu-adaptive-modular-learning-results.json]'
             '(gpu-adaptive-modular-learning-results.json).', '',
             '## Complete-call timing', '',
             'All full-resolution calls use one CPU participant, the original learner with '
             '262,144 samples and 256 leaves, checkpoints, and the full retained candidate set. '
             'The controls are CPU-only, GPU tokens plus exact ANS costs, and those GPU operations '
             'plus exact learner statistics. CPU property sorting, weighted sample gathering, '
             'tree decisions, model search, emission and every candidate remain inside the timer. '
             'Uploads, waits, cache creation and publication are included. Backend setup, file I/O '
             'and independent decode are outside the encoder-call timer. No sampling profiler runs.', '',
             'Image and route order rotate; the second repetition reverses each image\'s route order. '
             'Desktop process activity and between-triplet thermal/power snapshots are retained. '
             'Values below are medians and observed ranges from two repetitions, not confidence intervals.', '',
             '| Image | CPU seconds [range] | GPU tokens + ANS seconds [range] | Plus GPU learning seconds [range] |',
             '|---|---:|---:|---:|']
    for row in result['images']:
        cells = [row['image']]
        for route in ROUTES:
            d = row['routes'][route]['seconds']
            cells.append(f'{d["median"]:.3f} [{d["minimum"]:.3f}–{d["maximum"]:.3f}]')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f'The ratio of summed per-image medians is **{totals["cpu_ratio"]:.2f}×** '
              f'for CPU versus the learner hybrid, and **{totals["incremental_ratio"]:.3f}×** '
              'for adding GPU learning to the token/ANS hybrid '
              f'(**{100 * (1 - 1 / totals["incremental_ratio"]):.2f}% less elapsed time**). '
              'Actual per-triplet ratios and '
              'all per-run values are retained in JSON. These are local observations on three '
              'images; they do not establish a speedup over libjxl or broad device/corpus performance.', '',
              '## Attribution', '',
              'Each cell below sums three per-image stage medians. Nested training rows must '
              'not be added to the top-level training total.', '',
              '| Caller phase | CPU seconds | GPU tokens + ANS seconds | Plus GPU learning seconds |',
              '|---|---:|---:|---:|']
    for stage in totals['routes']['cpu']['stages']:
        if stage != 'gpu_input_tokens':
            lines.append('| ' + stage + ' | ' + ' | '.join(
                f'{totals["routes"][route]["stages"][stage]:.3f}' for route in ROUTES) + ' |')
    lines += ['', '| Nested training phase | CPU seconds | GPU tokens + ANS seconds | Plus GPU learning seconds |',
              '|---|---:|---:|---:|']
    for stage in totals['routes']['cpu']['training_stages']:
        lines.append('| ' + stage + ' | ' + ' | '.join(
            f'{totals["routes"][route]["training_stages"][stage]:.3f}' for route in ROUTES) + ' |')
    lines += ['', f'Within the learner hybrid\'s model stage, exact ANS costing accounts for '
              f'{totals["routes"]["gpu-learn"]["exact_ans_seconds"]:.3f} summed seconds. '
              'That interval includes the GPU adapter, upload and wait; the model stage is '
              'mixed CPU/GPU work, not an isolated remaining CPU cost.', '',
              'GPU statistics evaluate the same CPU-selected thresholds and exact Q12 integer '
              'costs. Leaves below 16,384 samples use the original scalar scan. Greedy growth, '
              'strict ties, parent histograms, property sorting and checkpoints stay on the CPU. '
              'Preparation/evaluation counters include host validation, upload and completion; '
              'they are nested within split scanning. The fallback counter explicitly includes '
              'small leaves and does not mean the richer learning policy was disabled. '
              'The measured change also includes CPU threshold preflight, which skips histogram '
              'work for eligible properties with no valid threshold; it is not a kernel-only speedup.', '',
              '## Rate and storage', '',
              '| Image | GJXL bytes | Default libjxl e7 bytes | Default libjxl e9 bytes |',
              '|---|---:|---:|---:|']
    for row in result['images']:
        lines.append(f'| {row["image"]} | {row["encoded_bytes"]:,} | '
                     f'{row["libjxl_bytes"]["e7"]:,} | {row["libjxl_bytes"]["e9"]:,} |')
    lines += ['', f'Summed GJXL size is {totals["encoded_bytes"]:,} bytes: '
              f'{totals["size_increase_percent"]["e7"]:.2f}% above default libjxl e7 and '
              f'{totals["size_increase_percent"]["e9"]:.2f}% above e9. These are verified saved '
              'reference files. No fresh libjxl timings enter this comparison. GPU execution '
              'preserves the richer CPU search\'s compression exactly.', '',
              '| Image | CPU / ANS hybrid / learner hybrid peak MiB | Respective admission MiB | Learner backend setup seconds |',
              '|---|---:|---:|---:|']
    for row in result['images']:
        values = [' / '.join(f'{row["routes"][route][key]["maximum"]/2**20:.1f}' for route in ROUTES)
                  for key in ('peak_backing', 'planned_capacity')]
        lines.append(f'| {row["image"]} | {values[0]} | {values[1]} | '
                     f'{row["routes"]["gpu-learn"]["backend_setup_seconds"]["median"]:.3f} |')
    lines += ['', 'Backing and admission are managed-memory measurements, not process RSS or '
              'driver memory. Every call fits its conservative bound and releases its resource '
              'domain. Learner preparation is scoped to one learning call and reuses immutable '
              'codes/count costs across eligible properties and leaves.', '',
              '## Small inputs and limits', '',
              '| Smoke image | Bytes | Complete seconds | GPU learner submissions | CPU small-leaf properties |',
              '|---|---:|---:|---:|---:|']
    for row in result['smoke']:
        r = row['result']
        g = r['adaptive_gpu']
        lines.append(f'| {row["image"]} | {r["encoded_bytes"]:,} | {r["seconds"]:.3f} | '
                     f'{g["tree_gpu_groups"]} | {g["tree_cpu_small_properties"]} |')
    lines += ['', 'These are single four-CPU-participant smoke runs, not repeated small-image '
              'performance measurements. All use GPU tokens and the explicit CPU ANS crossover '
              'below 64 nonempty sections. Tree checkpoints and all candidate decisions remain '
              'enabled. Other devices, intermediate shapes and multi-CPU full-image performance '
              'remain unqualified. Public encoding defaults are unchanged; this is a private '
              'Metal qualification route, not public API rollout or hosted CI evidence.', '',
              'See the [learner operation contract](gpu-modular-learner-statistics.md) for '
              'exact arithmetic, storage, fault checks and crossover evidence.', '',
              'Reproduce from saved artifacts:', '', '```sh',
              'python3 tools/modular_training_study/analyze_adaptive_learning.py \\',
              '  --capture build/adaptive-learning-qualified-20261011 \\',
              '  --baseline build/adaptive-baseline-20261011 \\',
              '  --json docs/gpu-adaptive-modular-learning-results.json \\',
              '  --markdown docs/gpu-adaptive-modular-learning.md', '```', '']
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
