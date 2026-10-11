#!/usr/bin/env python3
"""Summarize a retained complete-search diagnostic capture; never infer speedup."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest.get('finished_utc'), 'Capture is incomplete'
    assert not manifest['tracked_status'], 'Capture source was dirty'
    commands = {c['tag']: c for c in manifest['commands']}
    rows = []
    for photo, image in manifest['inputs'].items():
        native = commands[photo + '-gjxl-adaptive']
        result = native['result']
        assert native['returncode'] == 0 and result['decoded_exact']
        assert result['route'] == 'cpu-complete-search' and result['profile_enabled']
        assert result['threads'] == result['peak_cpu_participants'] == 1
        assert result['maximum_samples'] == 262144 and result['maximum_leaves'] == 256
        assert result['checkpoints'] and result['stages']['gpu_input_tokens'] == 0
        assert sum(result['stages'].values()) <= result['seconds'] + 1e-6
        assert sum(result['training']['stages'].values()) <= result['stages']['training'] + 1e-6
        assert native['output_sha256'] == sha(root / (native['tag'] + '.jxl'))
        policy = result['policy']
        row = dict(image=photo, width=image['width'], height=image['height'],
                   input_sha256=image['sha256'], gjxl_bytes=result['encoded_bytes'],
                   gjxl_sha256=native['output_sha256'], complete_search_seconds=result['seconds'],
                   candidates=result['candidates'], stages_seconds=result['stages'],
                   training=result['training'], entropy_work=result['entropy_work'],
                   planned_capacity=result['planned_capacity'], peak_backing=result['peak_backing'],
                   selected_policy=dict(rct=policy['rct'], transforms=policy['transforms'],
                       nodes=len(policy['nodes']), leaves=(len(policy['nodes']) + 1) // 2,
                       weighted_leaves=sum(n[0] == -1 and n[4] == 6 for n in policy['nodes']),
                       weighted_properties=sum(n[0] == 15 for n in policy['nodes']),
                       sha256=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()),
                   references={})
        for effort in (7, 9):
            reference = commands[f'{photo}-libjxl-default-e{effort}']
            assert reference['returncode'] == 0 and reference['decoded_exact']
            assert reference['equals_prior_capture']
            assert reference['output_sha256'] == sha(root / (reference['tag'] + '.jxl'))
            stderr = (root / (reference['tag'] + '.stderr')).read_text()
            match = re.search(r'geomean: ([0-9.]+) MP/s .*?, 3 reps, 0 threads\.', stderr)
            assert match, 'Missing serial reference timing'
            mp_s = float(match[1])
            row['references'][str(effort)] = dict(
                bytes=reference['output_bytes'], sha256=reference['output_sha256'],
                gjxl_size_delta_percent=100 * (row['gjxl_bytes'] / reference['output_bytes'] - 1),
                reported_geomean_mpix_per_second=mp_s,
                derived_seconds=image['width'] * image['height'] / 1e6 / mp_s)
        rows.append(row)
    totals = dict(gjxl_bytes=sum(r['gjxl_bytes'] for r in rows), references={})
    for effort in ('7', '9'):
        total = sum(r['references'][effort]['bytes'] for r in rows)
        totals['references'][effort] = dict(bytes=total,
            gjxl_size_delta_percent=100 * (totals['gjxl_bytes'] / total - 1))
    totals['stages_seconds'] = {k: sum(r['stages_seconds'][k] for r in rows)
                                for k in rows[0]['stages_seconds']}
    totals['complete_search_seconds'] = sum(r['complete_search_seconds'] for r in rows)
    totals['training_stages_seconds'] = {
        k: sum(r['training']['stages'][k] for r in rows) for k in rows[0]['training']['stages']}
    return dict(schema=1, source_revision=manifest['revision'], artifact_root=str(root),
                started_utc=manifest['started_utc'], finished_utc=manifest['finished_utc'],
                scope=manifest['note'], manifest_sha256=sha(root / 'manifest.json'),
                sampler_analysis_sha256=sha(root / 'samply-analysis.json'),
                binary_hashes=manifest['binary_hashes'], images=rows, totals=totals,
                samply=json.loads((root / 'samply-analysis.json').read_text()))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('capture', type=Path)
    args.add_argument('output', type=Path)
    options = args.parse_args()
    result = summarize(options.capture.resolve())
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['totals'], indent=2))
