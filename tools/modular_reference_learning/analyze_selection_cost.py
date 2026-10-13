#!/usr/bin/env python3
"""Qualify native transform-cost values and RCT rankings against pinned libjxl."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from selection_cost_replay import analyze

PIN = 'e8ff09762481785938d8e4e01333ed3917571161'


def qualify(release, sanitizer, validation):
    results = [analyze(p) for p in (release, sanitizer)]
    full = {(name, policy) for name in
            ('photo-macan', 'photo-flower', 'alpha-flower', 'chessboard', 'splines')
            for policy in ('auto', 'buffered')} | {('alpha16-ramp', 'auto')}
    selected = {('photo-macan', 'auto'), ('chessboard', 'auto'),
                ('alpha16-ramp', 'auto'), ('alpha-flower', 'buffered')}
    for result, cases in zip(results, (full, selected)):
        if {(r['input'], r['policy']) for r in result['results']} != cases:
            raise ValueError('Incomplete transform-cost qualification cohort')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (release, sanitizer)]
    if manifests[0]['coding_study'] != manifests[1]['coding_study']:
        raise ValueError('Different transform-cost captures')
    if (manifests[0]['sanitizer_environment'] or manifests[1]['sanitizer_environment'] !=
            {'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'}):
        raise ValueError('Unexpected transform-cost sanitizer environment')
    by_case = {(r['input'], r['policy']): r for r in results[0]['results']}
    for row in results[1]['results']:
        if row['result'] != by_case[row['input'], row['policy']]['result']:
            raise ValueError('Sanitizer transform costs/rankings differ')

    manifest_path = validation / 'manifest.json'
    frozen = json.loads(manifest_path.read_text())
    if (frozen['schema'] != 1 or frozen['source_snapshot_is_authoritative'] is not True or
            frozen['reference_revision'] != PIN):
        raise ValueError('Invalid transform-cost validation freeze')
    for p, digest in frozen['artifacts'].items():
        if sha(p) != digest:
            raise ValueError('Changed transform-cost validation artifact')

    def bound(path):
        if str(path) not in frozen['artifacts']:
            raise ValueError(f'Unbound transform-cost validation artifact: {path}')
        return path

    generated = None
    for build, study in (('release', release), ('asan', sanitizer)):
        root = validation / build
        if sha(bound(root / 'reference_learning')) != sha(study / 'reference_learning'):
            raise ValueError('Tested transform-cost binary differs from replay')
        for log_name, count in (('native-ctest.log', 3), ('research-ctest.log', 10)):
            log = bound(root / log_name).read_text()
            if (len(re.findall(rf'\d+/{count} Test\s+#\d+:.*Passed', log)) != count or
                    not re.search(rf'^100% tests passed(?:, 0 tests failed)? out of {count}$', log, re.M)):
                raise ValueError('Incomplete transform-cost checks')
        native_log = bound(root / 'native-LastTest.log').read_text()
        if ('Modular transform-cost allocation and validation contracts passed' not in native_log or
                native_log.count('Test Passed.') != 3):
            raise ValueError('Missing transform-cost allocation/validation controls')
        counts = re.findall(r'^Transform-cost reference: (\d+) cases, (\d+) exact estimates, '
                            r'(\d+) exact RCT rankings$', native_log, re.M)
        if len(counts) != 2 or counts[0] != counts[1] or tuple(map(int, counts[0])) != (396, 2372, 104):
            raise ValueError('Missing NEON/portable reference cost controls')
        if generated is not None and generated != counts[0]:
            raise ValueError('Release/sanitizer generated controls differ')
        generated = counts[0]
        for name in ('gjxl_modular_selection_cost_test', 'gjxl_modular_selection_cost_reference_test',
                     'gjxl_modular_selection_cost_scalar_reference_test'):
            bound(root / 'native' / name)
        for name in ('CMakeCache.txt', 'compile_commands.json'):
            bound(root / name)
            bound(root / 'native' / name)
        commands = json.loads((root / 'native/compile_commands.json').read_text())
        estimator = [c['command'] for c in commands if c['file'].endswith('/modular/selection_cost.cpp')]
        if (len(estimator) != 2 or sum('GJXL_SELECTION_COST_SCALAR=1' in c for c in estimator) != 1 or
                any('-fno-vectorize' in c or '-ffast-math' in c for c in estimator)):
            raise ValueError('Native transform estimator math/configuration changed')
        if any(('-fsanitize=address,undefined' in c) != (build == 'asan') for c in estimator):
            raise ValueError('Missing/unexpected native sanitizer instrumentation')
        if ('-fsanitize=address,undefined' in (root / 'CMakeCache.txt').read_text()) != (build == 'asan'):
            raise ValueError('Missing/unexpected research sanitizer instrumentation')
        # Bind the sources and native archives used by each replay, rather than
        # accepting a matching log alongside unrelated native build products.
        for p in (study / 'source').rglob('*'):
            if p.is_file() and sha(bound(root / 'source' / p.relative_to(study / 'source'))) != sha(p):
                raise ValueError('Transform-cost source snapshot differs')
        study_manifest = json.loads((study / 'manifest.json').read_text())
        for name in ('libgjxl_codec.a', 'libgjxl_codestream.a', 'libgjxl_gpu_ops.a', 'libgjxl_gpu.a'):
            original = [p for p in study_manifest['artifacts'] if Path(p).name == name]
            if len(original) != 1 or sha(bound(root / 'native' / name)) != study_manifest['artifacts'][original[0]]:
                raise ValueError('Transform-cost native archive differs')

    cases, estimates, rankings = map(int, generated)
    rows = [{k: r[k] for k in ('input', 'policy', 'summary')} for r in results[0]['results']]
    return {
        'schema': 1, 'reference_revision': PIN,
        'scope': ('Exact fixed-gradient transform costs on Apple AArch64, plus identical RCT candidates '
                  'and strict-order winners on generated/captured planes. Captured planes already '
                  'have e9 transforms; these are additional hypothetical RCTs, not original policy decisions.'),
        'qualification': {
            'native_ctest_per_build': 3, 'research_ctest_per_build': 10,
            'generated_cases_per_backend': cases, 'exact_estimates_per_backend': estimates,
            'exact_rct_rankings_per_backend': rankings,
            'backends': ['NEON', 'portable four-lane arithmetic'], 'builds': ['Release', 'ASan/UBSan'],
            'capture_cases': len(full), 'capture_totals': results[0]['totals'],
            'sanitizer_cases': len(selected), 'sanitizer_totals': results[1]['totals'],
            'sanitizer_results_identical': len(selected), 'managed_allocations_per_estimate': 0,
            'histogram_stack_bytes': 8772,
            'sanitizer_limits': frozen['sanitizer_limits'],
            'timing_qualified': False, 'palette_selection_qualified': False,
            'wp_selection_qualified': False, 'native_policy_qualified': False,
            'default_promotion_qualified': False,
        },
        'results': rows,
        'validation_manifest': str(manifest_path), 'validation_manifest_sha256': sha(manifest_path),
        'bound_validation_files': len(frozen['artifacts']),
        'artifacts': {**results[0]['artifacts'], **results[1]['artifacts']},
        'analyzer_sha256': sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'sanitizer-study', 'validation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(*(p.resolve() for p in (args.study, args.sanitizer_study, args.validation)))
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
