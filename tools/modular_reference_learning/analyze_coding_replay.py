#!/usr/bin/env python3
"""Verify captured coding artifacts and report conditional rate, never timing."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from coding_replay import project, validate_result
from run_stream_replay import capture_cases


def analyze(root):
    manifest_path, ledger_path = root / 'manifest.json', root / 'results.jsonl'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 1 or manifest['source_snapshot_is_authoritative'] is not True:
        raise ValueError('Unknown coding manifest')
    for path, digest in manifest['artifacts'].items():
        if sha(path) != digest:
            raise ValueError(f'Changed coding build artifact: {path}')
    cases = {(c['input'], c['variant']): c for c in manifest['cases']}
    if len(cases) != len(manifest['cases']):
        raise ValueError('Duplicate coding case')
    captured, _ = capture_cases(Path(manifest['policy_study']), Path(manifest['policy_validation']),
                                 [f'{name}:{variant}' for name, variant in cases])
    originals = {(c['input'], c['variant']): str(Path(c['trace']).parent.parent) for c in captured}
    if {key: c['capture'] for key, c in cases.items()} != originals:
        raise ValueError('Coding input is not the qualified original capture')
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    seen, results = set(), []
    for row in rows:
        key = row['input'], row['variant']
        if key not in cases or key in seen:
            raise ValueError('Duplicate or unknown coding result')
        seen.add(key)
        for path, digest in row['artifacts'].items():
            if sha(path) != digest:
                raise ValueError(f'Changed coding result artifact: {path}')
        case = cases[key]
        directory = root / f'{key[0]}-{key[1]}'
        outputs = {str(directory / p) for p in
                   ('input.txt', 'stdout.json', 'stderr.log', 'encoded/prefix.jxl', 'encoded/ans.jxl')}
        inputs = {str(p) for p in (Path(case['capture']) / 'streams').iterdir() if p.is_file()}
        inputs |= {case['source']['path'], str(Path(case['capture']) / 'output.jxl')}
        if (set(row['artifacts']) != inputs | outputs or
                {str(p) for p in directory.rglob('*') if p.is_file()} != outputs):
            raise ValueError('Missing or unbound coding input/output')
        if (directory / 'input.txt').read_text() != project(Path(case['capture'])):
            raise ValueError('Coding projection differs from original capture')
        command = [str(root / 'reference_learning'), 'stream-encode-check', case['source']['path'],
                   str(directory / 'input.txt'), str(directory / 'encoded')]
        result = row['result']
        if row['command'] != command or json.loads((directory / 'stdout.json').read_text()) != result:
            raise ValueError('Coding command/result differs from artifact')
        if sha(case['source']['path']) != case['source']['sha256']:
            raise ValueError('Changed coding source input')
        validate_result(Path(case['capture']), result)
        source = case['source']
        groups = ((source['width'] + 255) // 256) * ((source['height'] + 255) // 256)
        dc_groups = ((source['width'] + 2047) // 2048) * ((source['height'] + 2047) // 2048)
        if result['canonical_slots'] != 1 + groups + dc_groups:
            raise ValueError('Canonical coding geometry changed')
        reference_bytes = (Path(case['capture']) / 'output.jxl').stat().st_size
        if row['reference_bytes'] != reference_bytes:
            raise ValueError('Reference byte count differs from artifact')
        sizes = {}
        for encode in result['encodes']:
            path = directory / 'encoded' / (encode['mode'] + '.jxl')
            if path.stat().st_size != encode['bytes']:
                raise ValueError('Native byte count differs from artifact')
            sizes[encode['mode']] = encode['bytes']
        results.append({'input': key[0], 'variant': key[1], 'reference_bytes': reference_bytes,
                        'prefix_bytes': sizes['prefix'], 'ans_bytes': sizes['ans'],
                        'ans_delta_percent': 100 * (sizes['ans'] / reference_bytes - 1),
                        **{k: result[k] for k in ('captured_streams', 'canonical_slots', 'coded_streams',
                                                 'local_trees', 'global_trees', 'transforms', 'tokens')},
                        'maximum_storage_peak_bytes': max(e['storage_peak_bytes'] for e in result['encodes']),
                        'maximum_storage_bound_bytes': max(e['storage_bound_bytes'] for e in result['encodes'])})
    if seen != set(cases):
        raise ValueError('Incomplete coding ledger')
    totals = {}
    # The alpha16 sanitizer-derived capture is a separate correctness control;
    # keep the original five-image policy cohort's aggregate denominator.
    for variant in ('auto', 'buffered'):
        selected = [r for r in results if r['variant'] == variant and r['input'] != 'alpha16-ramp']
        if selected:
            native = sum(r['ans_bytes'] for r in selected)
            reference = sum(r['reference_bytes'] for r in selected)
            totals[variant] = {'inputs': len(selected), 'ans_bytes': native, 'reference_bytes': reference,
                               'ans_delta_percent': 100 * (native / reference - 1)}
    return {'schema': 1, 'cases': len(results), 'files': 2 * len(results),
            'captured_streams': sum(r['captured_streams'] for r in results),
            'coded_streams': sum(r['coded_streams'] for r in results),
            'local_trees': sum(r['local_trees'] for r in results),
            'global_trees': sum(r['global_trees'] for r in results),
            'tokens': sum(r['tokens'] for r in results),
            'tokens_exact': True, 'trees_exact': True, 'decoded_exact': True, 'managed_bounds_passed': True,
            'timing_qualified': False, 'native_search_policy_qualified': False,
            'results': results, 'conditional_rate_totals': totals,
            'scope': manifest['scope'],
            'artifacts': {str(p): sha(p) for p in (manifest_path, ledger_path)},
            'analyzer_sha256': sha(Path(__file__))}


def qualify(result, release, sanitizers, validation):
    required = {(name, variant) for name in
                ('photo-macan', 'photo-flower', 'alpha-flower', 'chessboard', 'splines')
                for variant in ('auto', 'buffered')} | {('alpha16-ramp', 'auto')}
    if {(r['input'], r['variant']) for r in result['results']} != required:
        raise ValueError('Incomplete Release coding cohort')
    checked = [analyze(root) for root in sanitizers]
    selected = {(r['input'], r['variant']) for s in checked for r in s['results']}
    expected = {('photo-macan', 'auto'), ('chessboard', 'auto'),
                ('alpha16-ramp', 'auto'), ('alpha-flower', 'buffered')}
    if selected != expected or sum(s['cases'] for s in checked) != len(expected):
        raise ValueError('Missing or repeated sanitizer capture control')
    comparisons = 0
    for study in sanitizers:
        recorded = json.loads((study / 'manifest.json').read_text())
        if '-fsanitize=address,undefined' not in (study / 'CMakeCache.txt').read_text():
            raise ValueError('Capture replay was not instrumented')
        for case in recorded['cases']:
            directory = f"{case['input']}-{case['variant']}"
            for mode in ('prefix', 'ans'):
                name = directory + '/encoded/' + mode + '.jxl'
                if sha(release / name) != sha(study / name):
                    raise ValueError('Sanitizer encoded bytes differ')
                comparisons += 1
            if (study / directory / 'stderr.log').read_text():
                raise ValueError('Unexpected sanitizer replay diagnostic')
    path = validation / 'manifest.json'
    record = json.loads(path.read_text())
    for artifact, digest in record['artifacts'].items():
        if sha(artifact) != digest:
            raise ValueError('Changed final coding validation artifact')
    if record['sanitizer_environment'] != {'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'}:
        raise ValueError('Missing sanitizer execution environment')
    if (sha(validation / 'release/research/reference_learning') != sha(release / 'reference_learning') or
            any(sha(validation / 'asan/research/reference_learning') != sha(s / 'reference_learning')
                for s in sanitizers)):
        raise ValueError('Validated binary differs from replay binary')
    for build in ('release', 'asan'):
        for layer, count in (('native', 8 if build == 'release' else 2), ('research', 9)):
            log = (validation / build / layer / 'LastTest.log').read_text()
            if log.count('Test Passed.') != count or 'Test Failed.' in log:
                raise ValueError('Missing final coding suite')
            if layer == 'native' and 'Prepared stream reference: 108 cases, 216 exact files' not in log:
                raise ValueError('Missing prepared-stream reference matrix')
            if layer == 'native':
                failures = re.findall(r'^(?:global|local) (?:prefix|ANS) (\d+) allocation failures qualified$',
                                      log, re.MULTILINE)
                if sorted(map(int, failures)) != [299, 300, 435, 436]:
                    raise ValueError('Missing allocation-failure qualification')
                if build == 'release' and (log.count('286 frozen stage/file artifacts match') != 2 or
                        'Transforms qualified: 453 stages, 906 complete files' not in log or
                        'Ordered transforms qualified: 175 stages, 350 complete files, 4520 headers' not in log):
                    raise ValueError('Missing frozen/transform regressions')
    hashes = [{p.name: sha(p) for p in (validation / build / 'generated').glob('*.jxl')}
              for build in ('release', 'asan')]
    if len(hashes[0]) != 216 or hashes[0] != hashes[1]:
        raise ValueError('Generated Release/sanitizer file identity failed')
    result['qualification'] = {'generated_cases': 108, 'generated_files_per_build': 216,
                               'generated_release_sanitizer_identical': 216,
                               'sanitizer_capture_cases': len(selected),
                               'capture_release_sanitizer_identical_files': comparisons,
                               'native_ctest_release': 8, 'native_ctest_asan': 2,
                               'research_ctest_per_build': 9,
                               'allocation_failures_per_build': 1470,
                               'frozen_native_artifacts': 286, 'frozen_reference_artifacts': 286,
                               'legacy_transform_cases': 453, 'legacy_transform_files': 906,
                               'ordered_transform_cases': 175, 'ordered_transform_files': 350,
                               'sanitizer_limits': record['sanitizer_limits'],
                               'validation_manifest': str(path), 'validation_manifest_sha256': sha(path),
                               'bound_validation_files': len(record['artifacts'])}
    result['sanitizer_studies'] = [{k: s[k] for k in ('cases', 'files', 'tokens', 'artifacts')} for s in checked]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sanitizers', type=Path, nargs='+')
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    result = analyze(args.study.resolve())
    if args.sanitizers or args.validation:
        if not (args.sanitizers and args.validation):
            parser.error('Combined qualification requires sanitizers and validation')
        result = qualify(result, args.study.resolve(), [p.resolve() for p in args.sanitizers],
                         args.validation.resolve())
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
