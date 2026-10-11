#!/usr/bin/env python3
"""Aggregate measured factorial cells; verify all referenced file hashes."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--results', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = [json.loads(line) for line in a.results.read_text().splitlines()]
reference = {r['input']: r for r in map(json.loads, a.reference.read_text().splitlines()) if r['config'] == 'l9'}
for r in rows + list(reference.values()):
    data = Path(r['output']).read_bytes()
    assert len(data) == r['encoded_bytes'] and hashlib.sha256(data).hexdigest() == r['output_sha256']
    assert r['decoded_exact']
inputs = sorted({r['input'] for r in rows})
configs = list(dict.fromkeys(r['config'] for r in rows))
assert len(rows) == len(inputs) * len(configs) == len(inputs) * 11
cells = {(r['input'], r['config']): r for r in rows}
assert len(cells) == len(rows)
summary = {'schema': 1, 'measured_files': len(rows), 'inputs': len(inputs),
           'reference_ledger': str(a.reference.resolve()), 'groups': {}, 'per_image': []}
for group in ('png', 'synthetic', 'all'):
    names = [n for n in inputs if group == 'all' or
             (cells[n, 'baseline']['category'] != 'synthetic') == (group == 'png')]
    lib = sum(reference[n]['encoded_bytes'] for n in names)
    baseline = sum(cells[n, 'baseline']['encoded_bytes'] for n in names)
    values = {}
    for c in configs:
        raw = sum(cells[n, c]['encoded_bytes'] for n in names)
        retained = sum(cells[n, c]['retained_bytes'] for n in names)
        values[c] = {'raw_bytes': raw, 'retained_bytes': retained,
                     'raw_gap_percent': 100 * (raw / lib - 1),
                     'retained_gap_percent': 100 * (retained / lib - 1),
                     'saved_vs_baseline': baseline - retained,
                     'baseline_excess_recovered_percent': (100 * (baseline - retained) / (baseline - lib)
                                                          if baseline > lib else None)}
    interactions = {}
    for mode in ('legacy', 'quantized'):
        for key in ('raw_bytes', 'retained_bytes'):
            v = lambda samples, leaves: values[f'{mode}-{samples}-{leaves}'][key]
            interactions[f'{mode}-{key}'] = v(65536, 16) + v(4096, 64) - v(4096, 16) - v(65536, 64)
    summary['groups'][group] = {'input': names, 'reference_bytes': lib, 'baseline_bytes': baseline,
                               'cells': values, 'positive_factorial_interaction_bytes': interactions}
for n in inputs:
    summary['per_image'].append({'input': n, 'reference_bytes': reference[n]['encoded_bytes'],
        'bytes': {c: cells[n, c]['encoded_bytes'] for c in configs},
        'leaves': {c: (len(cells[n, c]['policy']['nodes']) + 1) // 2 for c in configs},
        'quantized_65536_coverage': cells[n, 'quantized-65536-64']['coverage']})
a.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['groups']['png'], indent=2))
