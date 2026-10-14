#!/usr/bin/env python3
"""Qualify existing native direct-ANS modes on the captured-token cohort."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from analyze_entropy_replay import analyze
from entropy_replay import DIRECT_VARIANTS


def qualify(release, reference, sanitizer, validation):
    result, baseline, checked = (analyze(p) for p in (release, reference, sanitizer))
    full_cases = {(name, policy) for name in
                  ('photo-macan', 'photo-flower', 'alpha-flower', 'chessboard', 'splines')
                  for policy in ('auto', 'buffered')} | {('alpha16-ramp', 'auto')}
    selected = {('photo-macan', 'auto'), ('chessboard', 'auto'),
                ('alpha16-ramp', 'auto'), ('alpha-flower', 'buffered')}
    for study, cases in ((result, full_cases), (baseline, full_cases), (checked, selected)):
        if {(r['input'], r['policy']) for r in study['results']} != cases:
            raise ValueError('Incomplete direct-ANS qualification cohort')
    variants = result['variants']
    if (variants[0] != 'native' or len(variants) < 2 or set(variants[1:]) - set(DIRECT_VARIANTS) or
            checked['variants'] != variants or not {'reference', 'no_lz77', 'native'} <= set(baseline['variants'])):
        raise ValueError('Incomplete direct-ANS controls')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (release, reference, sanitizer)]
    if len({m['coding_study'] for m in manifests}) != 1:
        raise ValueError('Direct-ANS studies use different token captures')
    if (manifests[2]['sanitizer_environment'] !=
            {'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'} or
            '-fsanitize=address,undefined' not in (sanitizer / 'CMakeCache.txt').read_text()):
        raise ValueError('Missing direct-ANS sanitizer instrumentation/environment')
    for name, policy in full_cases:
        path = f'{name}-{policy}/encoded/native.jxl'
        if sha(release / path) != sha(reference / path):
            raise ValueError('Original native control changed')
    for name, policy in selected:
        for variant in variants:
            path = f'{name}-{policy}/encoded/{variant}.jxl'
            if sha(release / path) != sha(sanitizer / path):
                raise ValueError('Direct-ANS sanitizer output differs')
    validation_path = validation / 'manifest.json'
    frozen = json.loads(validation_path.read_text())
    if frozen['schema'] != 1 or frozen['source_snapshot_is_authoritative'] is not True:
        raise ValueError('Invalid direct-ANS validation freeze')
    for artifact, digest in frozen['artifacts'].items():
        if sha(artifact) != digest:
            raise ValueError('Changed direct-ANS validation artifact')
    for build, study in (('release', release), ('asan', sanitizer)):
        if sha(validation / build / 'reference_learning') != sha(study / 'reference_learning'):
            raise ValueError('Direct-ANS tested binary differs from replay')
        log = (validation / build / 'ctest.log').read_text()
        if (len(re.findall(r'\d+/10 Test\s+#\d+:.*Passed', log)) != 10 or
                not re.search(r'^100% tests passed(?:, 0 tests failed)? out of 10$', log, re.M)):
            raise ValueError('Incomplete direct-ANS research checks')
    originals = {(r['input'], r['policy']): r for r in baseline['results']}
    for row in result['results']:
        original = originals[row['input'], row['policy']]
        sizes = {r['variant']: r['bytes'] for r in original['results']}
        if row['stock_bytes'] != original['stock_bytes']:
            raise ValueError('Direct-ANS stock control differs')
        row['reference_entropy_bytes'] = sizes['reference']
        row['reference_no_lz77_bytes'] = sizes['no_lz77']
        for r in row['results']:
            r['delta_percent_from_stock'] = 100 * (r['bytes'] / row['stock_bytes'] - 1)
            r['delta_bytes_from_reference_no_lz77'] = r['bytes'] - sizes['no_lz77']
    for total in result['totals'].values():
        total['delta_percent_from_stock'] = {v: 100 * (n / total['stock_bytes'] - 1)
                                             for v, n in total['variant_bytes'].items()}
    result['qualification'] = {
        'original_native_files_identical': len(full_cases),
        'sanitizer_cases': len(selected), 'sanitizer_files_identical_to_release': checked['files'],
        'research_ctest_per_build': 10,
        'validation_manifest': str(validation_path), 'validation_manifest_sha256': sha(validation_path),
        'bound_validation_files': len(frozen['artifacts']), 'sanitizer_limits': frozen['sanitizer_limits'],
        'production_routing_qualified': False, 'managed_storage_qualified': False,
    }
    result['scope'] = ('Existing native direct-ANS data modes with unchanged native tree coding and captured '
                       'e9 transformed samples, trees and scope decisions. No learning or selection; '
                       'no production routing, timing, resource or default qualification.')
    for other in (baseline, checked):
        result['artifacts'].update(other['artifacts'])
    result['analyzer_sha256'] = sha(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'reference-study', 'sanitizer-study', 'validation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(*(p.resolve() for p in
                       (args.study, args.reference_study, args.sanitizer_study, args.validation)))
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
