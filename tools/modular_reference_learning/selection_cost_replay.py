#!/usr/bin/env python3
"""Run and verify native fixed-gradient selection costs on captured e9 planes."""
import argparse
import json
import math
import os
import struct
import subprocess
from pathlib import Path

from analyze import sha
from analyze_coding_replay import analyze as analyze_coding
from analyze_policy_study import numbered
from coding_replay import project
from study_provenance import freeze_build

RCTS = [0, 6, 5, 10, 26, 40, 12, 19, 8, 4, 9, 15, 16, 17, 32, 33, 2, 1, 3]


def validate_result(capture, source, result):
    project(capture)  # Validate/bind learning and coding scope semantics first.
    if (result['schema'] != 1 or result['costs_exact'] is not True or
            type(result['managed_allocations']) is not int or result['managed_allocations'] != 0 or
            result['timing_qualified'] is not False):
        raise ValueError('Invalid selection-cost evidence scope')
    images = {}
    for p in numbered(capture / 'streams', 'learning'):
        trace = json.loads(p.read_text())
        for im in trace['images']:
            if im['channels']:
                images[trace['local_stream_id'] if trace['local_stream_id'] >= 0 else im['index']] = im
    w, h = source['width'], source['height']
    dc, groups = ((w + 2047) // 2048) * ((h + 2047) // 2048), ((w + 255) // 256) * ((h + 255) // 256)
    ids = [0] + list(range(1 + dc, 1 + 2 * dc)) + list(range(18 + 3 * dc, 18 + 3 * dc + groups))
    if [r['id'] for r in result['streams']] != ids:
        raise ValueError('Changed canonical cost scopes')
    estimates, rankings, pixels, populated = 0, 0, 0, 0
    for row in result['streams']:
        im = images.get(row['id'], {'channels': [], 'metadata_channels': 0})
        channels = [c for c in im['channels'] if c['included']]
        metadata = im['metadata_channels']
        count = sum(c['bytes'] // 4 for c in channels)
        pixels += count
        populated += bool(channels)
        if (row['channels'], row['metadata_channels']) != (len(channels), metadata):
            raise ValueError('Changed selection-cost channel scope')
        color = channels[metadata:metadata + 3]
        eligible = len(color) == 3 and len({tuple(c[k] for k in ('width', 'height', 'hshift', 'vshift')) for c in color}) == 1
        candidates = row['rct_candidates']
        if [c['rct'] for c in candidates] != (RCTS if eligible else []):
            raise ValueError('Missing/reordered RCT selection candidates')
        for cost in [row['base'], *(c['cost'] for c in candidates)]:
            if (cost['pixels'] != count or any(type(cost[k]) is not int or cost[k] < 0
                    for k in ('pixels', 'histogram_bits', 'extra_bits')) or cost['extra_bits'] >= 1 << 32 or
                    any(type(cost[k]) not in (int, float) or not math.isfinite(cost[k]) or cost[k] < 0
                        for k in ('native_bits', 'reference_bits', 'histogram_fraction'))):
                raise ValueError('Invalid transform-cost decomposition')
            expected = struct.pack('<f', cost['histogram_bits'] + cost['extra_bits'] + int(cost['histogram_fraction']))
            if expected != struct.pack('<f', cost['native_bits']) or expected != struct.pack('<f', cost['reference_bits']):
                raise ValueError('Native/reference selection cost differs')
            estimates += 1
        if eligible:
            if candidates[0]['cost'] != row['base']:
                raise ValueError('Identity RCT changed cost')
            winner = min(candidates, key=lambda c: c['cost']['reference_bits'])['rct']
            rankings += 1
        else:
            winner = 0
        if row['selected_rct'] != winner:
            raise ValueError('Changed strict-improvement RCT winner/tie order')
    return {'canonical_streams': len(ids), 'populated_streams': populated, 'pixels': pixels,
            'exact_estimates': estimates, 'exact_rct_rankings': rankings}


def run(binary, coding, output, requested):
    checked = analyze_coding(coding)
    original = json.loads((coding / 'manifest.json').read_text())
    cases = original['cases']
    if requested:
        wanted = set(requested)
        cases = [c for c in cases if c['input'] + ':' + c['variant'] in wanted]
        if len(cases) != len(wanted): raise ValueError('Unknown selection-cost case')
    output.mkdir(parents=True, exist_ok=False)
    executable, build = freeze_build(binary, output)
    build['artifacts'].update(checked['artifacts'])
    manifest = {'schema': 1, **build, 'coding_study': str(coding), 'cases': cases,
                'sanitizer_environment': {k: os.environ[k] for k in ('ASAN_OPTIONS', 'UBSAN_OPTIONS') if k in os.environ},
                'scope': 'Exact fixed-gradient cost and 19-RCT comparisons on captured transformed planes. '
                         'These are additional hypothetical RCTs, not a replay of the original preselection inputs. '
                         'No learning, WP/palette selection, encoding, native policy or timing qualification.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (output / 'results.jsonl').open('x', buffering=1) as ledger:
        for c in cases:
            capture = Path(c['capture']); directory = output / (c['input'] + '-' + c['variant'])
            directory.mkdir()
            before = {str(p): sha(p) for p in (capture / 'streams').iterdir() if p.is_file()}
            before[c['source']['path']] = c['source']['sha256']
            description = directory / 'input.txt'; description.write_text(project(capture))
            command = [str(executable), 'stream-cost-check', c['source']['path'], str(description)]
            with (directory / 'stdout.json').open('x') as out, (directory / 'stderr.log').open('x') as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
            result = json.loads((directory / 'stdout.json').read_text())
            summary = validate_result(capture, c['source'], result)
            for p, digest in before.items():
                if sha(p) != digest: raise ValueError('Captured selection input changed')
            ledger.write(json.dumps({'input': c['input'], 'policy': c['variant'], 'command': command,
                                      'summary': summary, 'result': result,
                                      'artifacts': {**before, **{str(p): sha(p) for p in directory.iterdir()}}}) + '\n')
            print(c['input'], c['variant'], summary, flush=True)


def analyze(root):
    m = json.loads((root / 'manifest.json').read_text())
    if m['schema'] != 1 or m['source_snapshot_is_authoritative'] is not True:
        raise ValueError('Invalid selection-cost study')
    for p, h in m['artifacts'].items():
        if sha(p) != h: raise ValueError('Changed selection-cost build/input')
    coding = Path(m['coding_study']); analyze_coding(coding)
    original = json.loads((coding / 'manifest.json').read_text())
    if any(c not in original['cases'] for c in m['cases']):
        raise ValueError('Selection-cost case differs from frozen capture')
    rows = [json.loads(l) for l in (root / 'results.jsonl').read_text().splitlines()]
    if [(r['input'], r['policy']) for r in rows] != [(c['input'], c['variant']) for c in m['cases']]:
        raise ValueError('Missing/reordered selection-cost cases')
    totals = {}
    for row, case in zip(rows, m['cases']):
        capture = Path(case['capture']); directory = root / (row['input'] + '-' + row['policy'])
        outputs = {str(directory / n) for n in ('input.txt', 'stdout.json', 'stderr.log')}
        inputs = {str(p) for p in (capture / 'streams').iterdir() if p.is_file()} | {case['source']['path']}
        command = [str(root / 'reference_learning'), 'stream-cost-check', case['source']['path'], str(directory / 'input.txt')]
        if set(row['artifacts']) != inputs | outputs or {str(p) for p in directory.iterdir()} != outputs or row['command'] != command:
            raise ValueError('Unbound selection-cost command/output')
        for p, h in row['artifacts'].items():
            if sha(p) != h: raise ValueError('Changed selection-cost output/input')
        if ((directory / 'input.txt').read_text() != project(capture) or (directory / 'stderr.log').read_text() or
                json.loads((directory / 'stdout.json').read_text()) != row['result']):
            raise ValueError('Selection-cost projection/result differs')
        summary = validate_result(capture, case['source'], row['result'])
        if summary != row['summary']: raise ValueError('Changed selection-cost summary')
        for k, v in summary.items(): totals[k] = totals.get(k, 0) + v
    return {'schema': 1, 'cases': len(rows), 'scope': m['scope'], 'totals': totals, 'results': rows,
            'timing_qualified': False, 'native_policy_qualified': False,
            'artifacts': {str(root / n): sha(root / n) for n in ('manifest.json', 'results.jsonl')},
            'analyzer_sha256': sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--coding-study', type=Path)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    if args.mode == 'run':
        if args.binary is None or args.coding_study is None: parser.error('Run requires --binary and --coding-study')
        run(args.binary.resolve(), args.coding_study.resolve(), args.output.resolve(), args.cases)
    else:
        if args.study is None: parser.error('Analyze requires --study')
        result = analyze(args.study.resolve())
        with args.output.open('x') as out: out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
