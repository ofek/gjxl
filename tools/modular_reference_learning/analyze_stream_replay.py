#!/usr/bin/env python3
"""Verify all captured-stream replay artifacts and summarize qualified coverage."""
import argparse
import json
from pathlib import Path

from analyze import sha
from analyze_native_preparation import analyze as analyze_preparation
from run_stream_replay import capture_cases
from stream_replay import project, validate_result


def analyze(root):
    root = root.resolve()
    manifest_path, ledger = root / 'manifest.json', root / 'results.jsonl'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 1: raise ValueError('Unknown replay manifest schema')
    for path, digest in manifest['artifacts'].items():
        if sha(path) != digest: raise ValueError(f'Changed replay artifact: {path}')
    cases, _ = capture_cases(Path(manifest['policy_study']), Path(manifest['policy_validation']), manifest['selected_cases'])
    if cases != manifest['calls']: raise ValueError('Changed captured learning inputs')
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if len(rows) != len(cases): raise ValueError('Incomplete stream replay')
    groups = {}
    for case, row in zip(cases, rows):
        if any(case[k] != row[k] for k in ('input', 'variant', 'event')):
            raise ValueError('Duplicate, reordered or missing replay case')
        directory = root / f"{case['input']}-{case['variant']}-{case['event']}"
        source = Path(case['trace']); pixels = source.with_suffix('.i32le')
        description = directory / 'input.txt'
        command = [str(root / 'reference_learning'), 'stream-prepare-check', str(description), str(pixels)]
        if row['command'] != command: raise ValueError('Changed replay command')
        expected = {str(directory / n) for n in ('input.txt', 'stdout.json', 'stderr.log')}
        if row['artifacts'].keys() != expected or set(map(str, directory.iterdir())) != expected:
            raise ValueError('Missing or unbound replay output file')
        for path, digest in row['artifacts'].items():
            if sha(path) != digest: raise ValueError('Changed replay result artifact')
        trace = json.loads(source.read_text())
        if description.read_text() != project(trace, pixels.stat().st_size):
            raise ValueError('Replay projection differs from captured metadata')
        if row['result'] != json.loads((directory / 'stdout.json').read_text()):
            raise ValueError('Replay result disagrees with raw stdout')
        validate_result(trace, row['result'], case['fit'])
        key = case['input'], case['variant']
        if key not in groups:
            groups[key] = {'input': key[0], 'variant': key[1], 'learning_calls': 0,
                           'fits': 0, 'samples': 0, 'nodes': 0, 'local_learning_calls': 0,
                           'maximum_preparation_peak_bytes': 0, 'maximum_fitting_peak_bytes': 0}
        group, result = groups[key], row['result']
        group['learning_calls'] += 1
        group['local_learning_calls'] += result['local_stream_id'] >= 0
        for k in ('fits', 'samples', 'nodes'): group[k] += result[k]
        for stage in ('preparation', 'fitting'):
            k = f'maximum_{stage}_peak_bytes'
            group[k] = max(group[k], result[f'{stage}_storage_peak_bytes'])
    return {'schema': 1, 'learning_calls': len(rows), 'fits': sum(g['fits'] for g in groups.values()),
            'samples_exact': True, 'reference_trees_match_original_captures': True, 'native_trees_exact': True,
            'managed_bounds_passed': True, 'timing_qualified': False, 'groups': list(groups.values()),
            'scope': manifest['scope'], 'artifacts': {str(p): sha(p) for p in (manifest_path, ledger)}}


def qualify(result, release, scalar, sanitizer, default_wp, scalar_default_wp, corpus, validation):
    scalar_result, sanitizer_result = analyze(scalar), analyze(sanitizer)
    if scalar_result['groups'] != result['groups']:
        raise ValueError('Scalar replay coverage or results differ from Release')
    expected = [g for g in result['groups'] if (g['input'], g['variant']) in
                {('photo-macan', 'auto'), ('chessboard', 'auto'), ('alpha16-ramp', 'auto')}]
    if sanitizer_result['groups'] != expected:
        raise ValueError('Missing photographic, palette/local or true 16-bit sanitizer replay')
    manifest = json.loads((sanitizer / 'manifest.json').read_text())
    if manifest.get('sanitizer_environment', {}).get('UBSAN_OPTIONS') != 'halt_on_error=1':
        raise ValueError('Sanitizer replay did not request UB failure')
    controls = [analyze_preparation(root, corpus) for root in (default_wp, scalar_default_wp)]
    cases = {(name, v) for name in ('photo-macan', 'chessboard', 'splines', 'alpha16-ramp') for v in ('best', 'all')}
    for control, root, replay in zip(controls, (default_wp, scalar_default_wp), (release, scalar)):
        if {(r['input'], r['variant']) for r in control['results']} != cases or control['cases'] != len(cases):
            raise ValueError('Missing default-WP regression controls')
        if sha(root / 'reference_learning') != sha(replay / 'reference_learning'):
            raise ValueError('Default-WP controls used a different executable')
    path = validation / 'manifest.json'
    recorded = json.loads(path.read_text())
    for artifact, digest in recorded['artifacts'].items():
        if sha(artifact) != digest: raise ValueError('Changed final validation artifact')
    for name, study in (('release', release), ('scalar', scalar), ('asan', sanitizer)):
        root = validation / name
        if sha(root / 'reference_learning') != sha(study / 'reference_learning'):
            raise ValueError('Final validated executable differs from replay executable')
        log = (root / 'LastTest.log').read_text()
        if log.count('Test Passed.') != 8 or 'Test Failed.' in log:
            raise ValueError('Missing complete final test suite')
        json_rows = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
        if {'preparation_differential_cases': 119, 'samples_exact': True, 'tree_exact': True,
                'reference_weight_overflow_rejected': True} not in json_rows or {
                    'native_differential_cases': 129, 'tree_exact': True} not in json_rows:
            raise ValueError('Missing preparation/fitter differential suite')
    if '-DGJXL_HISTOGRAM_SCALAR=1' not in (validation / 'scalar/CMakeCache.txt').read_text():
        raise ValueError('Scalar build did not select the fallback')
    if '-fsanitize=address,undefined -fno-sanitize=vptr' not in (validation / 'asan/CMakeCache.txt').read_text():
        raise ValueError('Sanitizer build is not instrumented')
    failure = recorded['scalar_rounding_regression']
    old = Path(failure['before_source']).read_text()
    new = Path(failure['after_source']).read_text()
    change = ('  // Match the reference\'s separate multiply and subtract. Implicit contraction\n'
              '  // changes rounding in close split/predictor decisions; explicit std::fma in\n'
              '  // FastLog2 remains intentional. Scope this to the scalar fallback.\n'
              '#pragma STDC FP_CONTRACT OFF\n')
    if new.replace(change, '', 1) != old or new.count(change) != 1:
        raise ValueError('Scalar regression is not isolated to contraction control')
    if Path(failure['before_stderr']).read_text().strip() != 'Prepared tree mismatch at node 639':
        raise ValueError('Missing original scalar regression')
    trace = json.loads(Path(failure['trace']).read_text())
    validate_result(trace, json.loads(Path(failure['after_stdout']).read_text()))
    result.update({'scalar': scalar_result, 'sanitizer': sanitizer_result,
                   'default_wp_regressions': controls,
                   'build_validation': {'tests_per_suite': 8, 'preparation_differential_cases': 119,
                                        'fitting_differential_cases': 129,
                                        'release_binary_unchanged_by_scalar_fix': True,
                                        'scalar_rounding_regression_fixed': True,
                                        'scope': recorded['scope'], 'artifacts': {str(path): sha(path)}}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('scalar', 'sanitizer', 'default-wp', 'scalar-default-wp', 'corpus', 'validation'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    result = analyze(args.study)
    extra = [args.scalar, args.sanitizer, args.default_wp, args.scalar_default_wp, args.corpus, args.validation]
    if any(extra):
        if not all(extra): parser.error('Combined qualification requires all six validation arguments')
        result = qualify(result, args.study.resolve(), *(p.resolve() for p in extra))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['learning_calls'], 'learning calls;', result['fits'], 'fits; samples/trees exact')


if __name__ == '__main__':
    main()
