#!/usr/bin/env python3
"""Validate and summarize a frozen complete-search CPU/hybrid pilot."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(capture, baseline):
    manifest = json.loads((capture / 'manifest.json').read_text())
    control = json.loads((baseline / 'manifest.json').read_text())
    require(manifest.get('finished_utc'), 'Capture is incomplete')
    require(not manifest['tracked_status'], 'Capture started from a dirty revision')
    hashes = json.loads((capture / 'sha256-manifest.json').read_text())
    for name, entry in hashes.items():
        path = capture / name
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                f'Artifact changed: {name}')
    require(sha(Path(manifest['binary']['path'])) == manifest['binary']['sha256'], 'Binary changed')
    records = {r['tag']: r for r in manifest['commands']}
    require(len(records) == len(manifest['commands']) == 9, 'Expected nine unique pilot captures')
    smoke = []
    for name in ('photo-macan', 'palette-pattern', 'alpha16-ramp'):
        record = records[name + '-hybrid']
        r = record['result']
        require(record['returncode'] == 0 and record['prior_bytes_equal'] and record['prior_policy_equal'],
                f'Smoke parity failed: {name}')
        require(r['decoded_exact'] and r['gpu_submissions'] == r['candidates'] and
                r['adaptive_gpu']['cpu_fallback_candidates'] == 0, 'Smoke route mismatch')
        smoke.append(dict(name=name, encoded_bytes=r['encoded_bytes'], candidates=r['candidates'],
                          adaptive_gpu=r['adaptive_gpu'], decoded_exact=True, prior_bytes_equal=True))
    pairs = []
    for name in ('1029', '1039', '1049'):
        c, h = records[name + '-cpu'], records[name + '-hybrid']
        cpu, hybrid = c['result'], h['result']
        for record in (c, h):
            r = record['result']
            require(record['returncode'] == 0 and record['paired_bytes_policy_training_equal'] and
                    record['prior_bytes_equal'] and record['prior_policy_equal'] and r['decoded_exact'],
                    f'Paired parity failed: {record["tag"]}')
            require((r['width'], r['height'], r['channels'], r['bits'], r['threads']) == (4000, 3000, 3, 8, 1),
                    'Wrong input or participant count')
            require((r['maximum_samples'], r['maximum_leaves'], r['checkpoints']) == (262144, 256, True),
                    'Wrong search budget')
            require(record['input_sha256'] == manifest['inputs'][name]['sha256'] ==
                    control['inputs'][name]['sha256'], 'Input mismatch')
        require(c['output_sha256'] == h['output_sha256'] and cpu['policy'] == hybrid['policy'],
                'CPU/hybrid output or policy differs')
        for key in ('learning_calls', 'population', 'samples', 'split_evaluations',
                    'sorted_properties', 'sorted_samples'):
            require(cpu['training'][key] == hybrid['training'][key], 'Training work differs')
        require(cpu['candidates'] == hybrid['candidates'], 'Candidate counts differ')
        for record in (c, h):
            require(sha(capture / (record['tag'] + '.jxl')) == record['output_sha256'],
                    'Output hash differs from the paired record')
        g = hybrid['adaptive_gpu']
        require(g['feature_preparations'] + g['feature_reuses'] == g['encoded_candidates'] ==
                hybrid['candidates'] and g['evictions'] + 1 == g['feature_preparations'],
                'Feature cache accounting differs')
        require(cpu['route'] == 'cpu-complete-search' and hybrid['route'] == 'hybrid-complete-search',
                'Incorrect paired route')
        require(cpu['gpu_submissions'] == 0 and hybrid['gpu_submissions'] == hybrid['candidates'] and
                hybrid['adaptive_gpu']['cpu_fallback_candidates'] == 0, 'Incorrect GPU participation')
        reference_bytes = {}
        for effort in (7, 9):
            ref = next(r for r in control['commands'] if r['tag'] == f'{name}-libjxl-default-e{effort}')
            path = baseline / f'{name}-libjxl-default-e{effort}.jxl'
            require(ref['decoded_exact'] and sha(path) == ref['output_sha256'], 'Reference output changed')
            reference_bytes[f'e{effort}'] = ref['output_bytes']
        pairs.append(dict(image=name, encoded_bytes=cpu['encoded_bytes'], libjxl_bytes=reference_bytes,
                          output_sha256=c['output_sha256'], cpu_seconds=cpu['seconds'],
                          hybrid_seconds=hybrid['seconds'], observed_time_ratio=cpu['seconds']/hybrid['seconds'],
                          candidates=cpu['candidates'], adaptive_gpu=hybrid['adaptive_gpu'],
                          cpu_stages=cpu['stages'], hybrid_stages=hybrid['stages'],
                          cpu_peak_backing=cpu['peak_backing'], hybrid_peak_backing=hybrid['peak_backing'],
                          cpu_planned_capacity=cpu['planned_capacity'], hybrid_planned_capacity=hybrid['planned_capacity'],
                          hybrid_backend_setup_seconds=hybrid['backend_setup_seconds'],
                          training=hybrid['training'], entropy_work=hybrid['entropy_work'],
                          cpu_training=cpu['training'], cpu_entropy_work=cpu['entropy_work']))
    totals = {key: sum(row[key] for row in pairs) for key in ('encoded_bytes', 'cpu_seconds', 'hybrid_seconds')}
    totals['observed_time_ratio'] = totals['cpu_seconds'] / totals['hybrid_seconds']
    totals['libjxl_bytes'] = {key: sum(row['libjxl_bytes'][key] for row in pairs) for key in ('e7', 'e9')}
    totals['size_increase_percent'] = {key: 100 * (totals['encoded_bytes']/n - 1)
                                       for key, n in totals['libjxl_bytes'].items()}
    totals['stages'] = {stage: dict(cpu=sum(row['cpu_stages'][stage] for row in pairs),
                                  hybrid=sum(row['hybrid_stages'][stage] for row in pairs))
                       for stage in pairs[0]['cpu_stages']}
    totals['hybrid_exact_ans_seconds'] = sum(r['entropy_work']['ans_token_cost_ns'] for r in pairs) / 1e9
    return dict(schema=1, revision=manifest['revision'], artifact_root=str(capture.resolve()),
                started_utc=manifest['started_utc'], finished_utc=manifest['finished_utc'],
                qualification='Diagnostic single-pair pilot on a busy desktop; no general throughput claim',
                smoke=smoke, pairs=pairs, totals=totals)


def markdown(result):
    t = result['totals']
    lines = ['# Adaptive Modular complete-search hybrid pilot', '',
        'The hybrid search preserves the complete CPU files and winning policies on all three '
        'full-resolution inputs. Every candidate used Metal token evaluation; no CPU fallback occurred. '
        'The photographic, palette and 16-bit-alpha smoke captures also match the saved CPU files.', '',
        '**Timing scope:** one fresh CPU/hybrid pair per image on a busy desktop, with alternating '
        'pair order. These are diagnostic observations, not repeated throughput qualification. '
        'Both routes use one CPU participant, the original 262144-sample/256-leaf learner, '
        'retained baseline candidates and checkpoint selection. Backend initialization is outside '
        'the encoder-call timer and is reported separately. CPU transforms, weighted-feature '
        'preparation, cache creation, every candidate encode and output publication are inside it.', '',
        f'Frozen implementation: `{result["revision"]}`. Artifacts: `{result["artifact_root"]}`. '
        'Inputs are the original 4000×3000 RGB8 PAMs used by the CPU baseline.', '',
        '| Image | Encoded bytes | CPU seconds | Hybrid seconds | Observed CPU/hybrid ratio | Feature builds / reuses |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['pairs']:
        g = r['adaptive_gpu']
        lines.append(f'| {r["image"]} | {r["encoded_bytes"]:,} | {r["cpu_seconds"]:.3f} | '
                     f'{r["hybrid_seconds"]:.3f} | {r["observed_time_ratio"]:.2f}× | '
                     f'{g["feature_preparations"]} / {g["feature_reuses"]} |')
    lines += ['', f'The aggregate observed time ratio is {t["observed_time_ratio"]:.2f}×. '
              'The paired outputs, full selected policies and training work counters match exactly. '
              'Each output is independently decoded by pinned libjxl before capture succeeds.', '',
              '## Attribution', '', '| Caller stage | CPU total seconds | Hybrid total seconds |',
              '|---|---:|---:|']
    for stage, times in t['stages'].items():
        if stage != 'gpu_input_tokens':
            lines.append(f'| {stage} | {times["cpu"]:.3f} | {times["hybrid"]:.3f} |')
    features = sum(r['adaptive_gpu']['feature_seconds'] for r in result['pairs'])
    tokens = sum(r['adaptive_gpu']['token_seconds'] for r in result['pairs'])
    lines += ['', f'Within hybrid `tokens`, feature preparation accounts for {features:.3f} seconds '
              f'and token generation/handoff for {tokens:.3f} seconds. These are nested intervals; '
              'do not add them again to the table. Feature preparation currently computes weighted '
              'history on CPU. The token interval includes dispatch setup, GPU wait and host handoff, '
              'not just device execution. Training and entropy modeling remain CPU work.', '',
              f'Exact ANS token-cost simulation alone takes {t["hybrid_exact_ans_seconds"]:.3f} '
              f'seconds ({100*t["hybrid_exact_ans_seconds"]/t["hybrid_seconds"]:.1f}% of hybrid '
              'complete-call time), nested inside `model`. It is the next implementation target: '
              'preserve each section\'s reverse-order rANS recurrence while processing independent '
              'sections and candidate models in parallel. Leave model selection and ties unchanged.', '',
              '## Resource tradeoff', '',
              '| Image | CPU peak backing MiB | Hybrid peak backing MiB | CPU admission MiB | Hybrid admission MiB | Backend setup seconds |',
              '|---|---:|---:|---:|---:|---:|']
    for r in result['pairs']:
        lines.append(f'| {r["image"]} | {r["cpu_peak_backing"]/2**20:.1f} | '
                     f'{r["hybrid_peak_backing"]/2**20:.1f} | {r["cpu_planned_capacity"]/2**20:.1f} | '
                     f'{r["hybrid_planned_capacity"]/2**20:.1f} | {r["hybrid_backend_setup_seconds"]:.3f} |')
    lines += ['', 'These are managed backing charges and conservative admitted capacities, not process '
              'RSS or GPU-only memory. Retained transformed planes and weighted maps increase the '
              'hybrid peak; each call fits its plan and releases its domain before returning.', '',
              '## Compression and limits', '',
              f'Summed output is {t["encoded_bytes"]:,} bytes: '
              f'{t["size_increase_percent"]["e7"]:.2f}% above saved default libjxl e7 and '
              f'{t["size_increase_percent"]["e9"]:.2f}% above e9. These are the unchanged CPU-baseline '
              'rates; GPU evaluation neither improves nor weakens the coding policy. The reference '
              'sizes come from verified saved files, not new libjxl timings. This pilot does not '
              'establish an end-to-end speedup over libjxl.', '',
              'The captured process snapshots show background desktop activity. A single pair cannot '
              'separate scheduling variation from smaller stage changes. Use the large phase trends '
              'to prioritize work, then repeat paired captures before claiming throughput. Resource '
              'peaks, conservative planned capacities, setup times and complete nested counters are '
              'retained in the companion JSON. Public encoder defaults are unchanged.', '',
              'Reproduce the summary from the retained capture and CPU baseline:', '', '```sh',
              'python3 tools/modular_training_study/analyze_adaptive_hybrid.py \\',
              '  --capture build/adaptive-hybrid-pilot-20261011 \\',
              '  --baseline build/adaptive-baseline-20261011 \\',
              '  --json docs/gpu-adaptive-modular-hybrid-results.json \\',
              '  --markdown docs/gpu-adaptive-modular-hybrid.md', '```', '']
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
