#!/usr/bin/env python3
"""Serial paired timing of native sample-model construction on fixed tokens.

Input preparation, tree coding, final token emission, assembly and independent
decode are outside the measured data-model intervals. Detailed profiling is off.
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

from analyze import sha
from analyze_entropy_replay import analyze as analyze_rate
from entropy_replay import validate_controls, validate_result
from study_provenance import freeze_build

VARIANTS = ('native', 'native_direct_rate64', 'native_direct_estimated64')


def run(study, output, repeats):
    checked = analyze_rate(study)
    source = json.loads((study / 'manifest.json').read_text())
    if (source.get('model_timings', False) or set(VARIANTS) - set(source['variants']) or
            '-fsanitize=' in (study / 'CMakeCache.txt').read_text() or
            'CMAKE_BUILD_TYPE:STRING=Release' not in (study / 'CMakeCache.txt').read_text()):
        raise ValueError('Timing requires the uninstrumented qualified rate study')
    output.mkdir(parents=True, exist_ok=False)
    binary, build = freeze_build(study / 'reference_learning', output)
    build['artifacts'].update(checked['artifacts'])
    manifest = {'schema': 1, **build, 'rate_study': str(study), 'cases': source['cases'],
                'variants': list(VARIANTS), 'warmups': 1, 'repeats': repeats,
                'serial_processes': True, 'detailed_profiling': False,
                'scope': __doc__}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (output / 'results.jsonl').open('x', buffering=1) as ledger:
        for iteration in range(-1, repeats):
            shift = (iteration + 1) % len(VARIANTS)
            variants = list(VARIANTS[shift:] + VARIANTS[:shift])
            # Rotate case order as well as paired variant order across rounds.
            shift_cases = (iteration + 1) % len(source['cases'])
            cases = source['cases'][shift_cases:] + source['cases'][:shift_cases]
            for case in cases:
                name = f"{case['input']}-{case['variant']}"
                directory = output / f'round-{iteration + 1}' / name
                directory.mkdir(parents=True)
                description = study / name / 'input.txt'
                command = [str(binary), 'stream-entropy-time', case['source']['path'], str(description),
                           str(directory / 'encoded'), ','.join(variants)]
                with (directory / 'stdout.jsonl').open('x') as out, (directory / 'stderr.log').open('x') as err:
                    subprocess.run(command, stdout=out, stderr=err, check=True)
                results = [json.loads(l) for l in (directory / 'stdout.jsonl').read_text().splitlines()]
                if [r['variant'] for r in results] != variants:
                    raise ValueError('Incomplete timing variants')
                for r in results:
                    validate_result(Path(case['capture']), r)
                    if not r['model_timing_recorded'] or sha(directory / 'encoded' / (r['variant'] + '.jxl')) != sha(
                            study / name / 'encoded' / (r['variant'] + '.jxl')):
                        raise ValueError('Timed output differs from the rate control')
                validate_controls(results)
                ledger.write(json.dumps({'iteration': iteration, 'input': case['input'], 'policy': case['variant'],
                                          'command': command, 'results': results,
                                          'artifacts': {str(p): sha(p) for p in directory.rglob('*') if p.is_file()}}) + '\n')
                print(iteration, name, {r['variant']: round(sum(s['model_cpu_seconds'] for s in r['scopes']
                                                               if s['kind'] == 'data'), 6) for r in results}, flush=True)
    analyze_rate(study)  # Fail if any bound capture/build input changed during the run.


def analyze(root):
    manifest_path, ledger = root / 'manifest.json', root / 'results.jsonl'
    m = json.loads(manifest_path.read_text())
    if (m['schema'] != 1 or m['source_snapshot_is_authoritative'] is not True or m['warmups'] != 1 or
            m['repeats'] < 5 or m['serial_processes'] is not True or m['detailed_profiling'] is not False or
            m['variants'] != list(VARIANTS)):
        raise ValueError('Invalid timing protocol')
    for p, h in m['artifacts'].items():
        if sha(p) != h: raise ValueError('Changed timing build/input artifact')
    study = Path(m['rate_study']); rate = analyze_rate(study)
    original = json.loads((study / 'manifest.json').read_text())
    if (m['cases'] != original['cases'] or sha(root / 'reference_learning') != sha(study / 'reference_learning') or
            '-fsanitize=' in (root / 'CMakeCache.txt').read_text() or
            'CMAKE_BUILD_TYPE:STRING=Release' not in (root / 'CMakeCache.txt').read_text()):
        raise ValueError('Timing cohort or executable differs')
    cases = {(c['input'], c['variant']): c for c in m['cases']}
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    expected_order = []
    for iteration in range(-1, m['repeats']):
        shift = (iteration + 1) % len(m['cases'])
        for c in m['cases'][shift:] + m['cases'][:shift]:
            expected_order.append((iteration, c['input'], c['variant']))
    if [(r['iteration'], r['input'], r['policy']) for r in rows] != expected_order:
        raise ValueError('Timing rounds are incomplete or out of order')
    measurements = {}
    for row in rows:
        case = cases[row['input'], row['policy']]
        name = f"{row['input']}-{row['policy']}"
        directory = root / f"round-{row['iteration'] + 1}" / name
        shift = (row['iteration'] + 1) % len(VARIANTS)
        variants = list(VARIANTS[shift:] + VARIANTS[:shift])
        command = [str(root / 'reference_learning'), 'stream-entropy-time', case['source']['path'],
                   str(study / name / 'input.txt'), str(directory / 'encoded'), ','.join(variants)]
        outputs = {str(directory / p) for p in ('stdout.jsonl', 'stderr.log')}
        outputs |= {str(directory / 'encoded' / (v + '.jxl')) for v in variants}
        if (row['command'] != command or set(row['artifacts']) != outputs or
                {str(p) for p in directory.rglob('*') if p.is_file()} != outputs):
            raise ValueError('Unbound timing command/output')
        for p, h in row['artifacts'].items():
            if sha(p) != h: raise ValueError('Changed timing output')
        results = [json.loads(l) for l in (directory / 'stdout.jsonl').read_text().splitlines()]
        if results != row['results'] or [r['variant'] for r in results] != variants or (directory / 'stderr.log').read_text():
            raise ValueError('Timing ledger/stdout/diagnostics differ')
        for r in results:
            validate_result(Path(case['capture']), r)
            if not r['model_timing_recorded'] or sha(directory / 'encoded' / (r['variant'] + '.jxl')) != sha(
                    study / name / 'encoded' / (r['variant'] + '.jxl')):
                raise ValueError('Timing did not preserve qualified rate bytes')
            if row['iteration'] >= 0:
                values = measurements.setdefault((row['input'], row['policy'], r['variant']), {'cpu': [], 'wall': []})
                for field in ('cpu', 'wall'):
                    values[field].append(sum(s[f'model_{field}_seconds'] for s in r['scopes'] if s['kind'] == 'data'))
        validate_controls(results)
    summary, totals = [], {v: {'cpu': 0.0, 'wall': 0.0} for v in VARIANTS}
    for name, policy in cases:
        medians = {v: {f: statistics.median(measurements[name, policy, v][f]) for f in ('cpu', 'wall')}
                   for v in VARIANTS}
        for v in VARIANTS:
            for field in ('cpu', 'wall'): totals[v][field] += medians[v][field]
        summary.append({'input': name, 'policy': policy, 'median_seconds': medians,
                        'estimated_over_exact': {f: medians[VARIANTS[2]][f] / medians[VARIANTS[1]][f]
                                                  for f in ('cpu', 'wall')},
                        'raw_seconds': {v: measurements[name, policy, v] for v in VARIANTS}})
    return {'schema': 1, 'cases': len(cases), 'processes': len(rows), 'files': len(rows) * len(VARIANTS),
            'warmups_per_case': 1, 'measured_rounds': m['repeats'], 'scope': m['scope'],
            'all_files_match_rate_control': True, 'data_model_timing_qualified': True,
            'whole_encoder_timing_qualified': False,
            'results': summary, 'sum_of_medians_seconds': totals,
            'sum_estimated_over_exact': {f: totals[VARIANTS[2]][f] / totals[VARIANTS[1]][f] for f in ('cpu', 'wall')},
            'artifacts': {**rate['artifacts'], str(manifest_path): sha(manifest_path), str(ledger): sha(ledger)},
            'analyzer_sha256': sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'analyze'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 5: parser.error('At least five measured rounds are required')
    if args.mode == 'run': run(args.study.resolve(), args.output.resolve(), args.repeats)
    else:
        result = analyze(args.study.resolve())
        with args.output.open('x') as out: out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
