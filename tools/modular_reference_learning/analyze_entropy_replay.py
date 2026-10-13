#!/usr/bin/env python3
"""Verify same-token ablations and report direct measured differences."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from analyze_coding_replay import analyze as analyze_coding
from coding_replay import project
from entropy_replay import VARIANTS, validate_controls, validate_result


def analyze(root):
    manifest_path, ledger = root / 'manifest.json', root / 'results.jsonl'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 1 or manifest['source_snapshot_is_authoritative'] is not True:
        raise ValueError('Unknown entropy study')
    for path, digest in manifest['artifacts'].items():
        if sha(path) != digest:
            raise ValueError(f'Changed entropy build artifact: {path}')
    coding = Path(manifest['coding_study'])
    analyze_coding(coding)
    original = json.loads((coding / 'manifest.json').read_text())
    cases = {(c['input'], c['variant']): c for c in manifest['cases']}
    originals = {(c['input'], c['variant']): c for c in original['cases']}
    variants = manifest['variants']
    if (len(cases) != len(manifest['cases']) or any(originals.get(k) != c for k, c in cases.items()) or
            not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS)):
        raise ValueError('Changed entropy cohort or variants')
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    seen, summary, verified_partitions = set(), [], 0
    for row in rows:
        key = row['input'], row['policy']
        if key not in cases or key in seen:
            raise ValueError('Repeated or unexpected entropy case')
        seen.add(key)
        case = cases[key]
        directory = root / f'{key[0]}-{key[1]}'
        capture = Path(case['capture'])
        outputs = {str(directory / p) for p in ('input.txt', 'stdout.jsonl', 'stderr.log')}
        outputs |= {str(directory / 'encoded' / (v + '.jxl')) for v in variants}
        inputs = {str(p) for p in (capture / 'streams').iterdir() if p.is_file()}
        native_control = coding / directory.name / 'encoded/ans.jxl'
        inputs |= {case['source']['path'], str(capture / 'output.jxl'), str(native_control)}
        if set(row['artifacts']) != inputs | outputs or {str(p) for p in directory.rglob('*') if p.is_file()} != outputs:
            raise ValueError('Unbound or missing entropy artifact')
        for path, digest in row['artifacts'].items():
            if sha(path) != digest:
                raise ValueError(f'Changed entropy result artifact: {path}')
        command = [str(root / 'reference_learning'), 'stream-entropy-check', case['source']['path'],
                   str(directory / 'input.txt'), str(directory / 'encoded'), ','.join(variants)]
        if row['command'] != command or (directory / 'input.txt').read_text() != project(capture, entropy=True):
            raise ValueError('Entropy command/projection differs from capture')
        results = [json.loads(line) for line in (directory / 'stdout.jsonl').read_text().splitlines()]
        if row['results'] != results or [r['variant'] for r in results] != variants:
            raise ValueError('Incomplete/reordered entropy controls')
        if (directory / 'stderr.log').read_text():
            raise ValueError('Unexpected entropy replay diagnostic')
        stock_bytes = (capture / 'output.jxl').stat().st_size
        compact = []
        for r in results:
            validate_result(capture, r)
            output = directory / 'encoded' / (r['variant'] + '.jxl')
            if output.stat().st_size != r['bytes']:
                raise ValueError('Entropy result bytes differ from file')
            if r['variant'] == 'native' and sha(output) != sha(native_control):
                raise ValueError('Native control differs from frozen coding output')
            compact.append({k: r[k] for k in ('variant', 'bytes', 'tree_bits', 'data_model_bits', 'data_token_bits',
                                               'header_bits', 'toc_bits', 'stream_header_bits', 'section_padding_bits')})
            compact[-1]['lz77_tree_scopes'] = sum(s['lz77'] for s in r['scopes'] if s['kind'] == 'tree')
            compact[-1]['lz77_data_scopes'] = sum(s['lz77'] for s in r['scopes'] if s['kind'] == 'data')
            compact[-1]['data_histograms'] = sum(s['clusters'] for s in r['scopes'] if s['kind'] == 'data')
        baseline = next((r['bytes'] for r in results if r['variant'] == 'reference'), None)
        verified_partitions += validate_controls(results)
        if baseline is not None:
            for r in compact:
                r['delta_bytes_from_reference_entropy'] = r['bytes'] - baseline
                r['delta_percent_from_reference_entropy'] = 100 * (r['bytes'] / baseline - 1)
        summary.append({'input': key[0], 'policy': key[1], 'stock_bytes': stock_bytes,
                        'reference_entropy_minus_stock_bytes': None if baseline is None else baseline - stock_bytes,
                        'results': compact})
    if seen != set(cases):
        raise ValueError('Missing entropy case')
    totals = {}
    for policy in ('auto', 'buffered'):
        selected = [r for r in summary if r['policy'] == policy and r['input'] != 'alpha16-ramp']
        if selected:
            sizes = {v: sum(next(r['bytes'] for r in c['results'] if r['variant'] == v) for c in selected) for v in variants}
            baseline = sizes.get('reference')
            totals[policy] = {'inputs': len(selected), 'stock_bytes': sum(r['stock_bytes'] for r in selected),
                              'variant_bytes': sizes, 'delta_bytes_from_reference_entropy': None if baseline is None else
                              {v: n - baseline for v, n in sizes.items()}}
    return {'schema': 1, 'cases': len(rows), 'files': len(rows) * len(variants),
            'variants': variants, 'tokens_and_trees_exact': True, 'decodes_exact': True,
            'native_control_matches_frozen': 'native' in variants,
            'transplanted_data_scopes_match_full_reference': verified_partitions,
            'timing_qualified': False, 'native_policy_qualified': False,
            'scope': manifest['scope'], 'results': summary, 'totals': totals,
            'artifacts': {str(p): sha(p) for p in (manifest_path, ledger)},
            'analyzer_sha256': sha(Path(__file__))}


def qualify(release, partition, sanitizer, validation):
    """Join only directly measured cells; verify the two studies' common control."""
    studies = [analyze(p) for p in (release, partition, sanitizer)]
    base, maps, sanitized = studies
    full_cases = {(name, policy) for name in
                  ('photo-macan', 'photo-flower', 'alpha-flower', 'chessboard', 'splines')
                  for policy in ('auto', 'buffered')} | {('alpha16-ramp', 'auto')}
    sanitizer_cases = {('photo-macan', 'auto'), ('chessboard', 'auto'),
                       ('alpha16-ramp', 'auto'), ('alpha-flower', 'buffered')}
    for result, expected in zip(studies, (full_cases, full_cases, sanitizer_cases)):
        if {(r['input'], r['policy']) for r in result['results']} != expected:
            raise ValueError('Incomplete entropy qualification cohort')
    if (base['variants'] != list(VARIANTS[:13]) or
            maps['variants'] != ['no_lz77_clusters32', *VARIANTS[13:]] or
            sanitized['variants'] != list(VARIANTS)):
        raise ValueError('Incomplete entropy qualification variants')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (release, partition, sanitizer)]
    if len({m['coding_study'] for m in manifests}) != 1:
        raise ValueError('Entropy studies do not share the captured-token control')
    if (manifests[2]['sanitizer_environment'] !=
            {'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'} or
            '-fsanitize=address,undefined' not in (sanitizer / 'CMakeCache.txt').read_text()):
        raise ValueError('Missing entropy sanitizer instrumentation/environment')
    for name, policy in full_cases:
        path = f'{name}-{policy}/encoded/no_lz77_clusters32.jxl'
        if sha(release / path) != sha(partition / path):
            raise ValueError('Common reference control changed between studies')
    for name, policy in sanitizer_cases:
        for variant in VARIANTS:
            source = partition if variant.startswith('reference_map32') else release
            path = f'{name}-{policy}/encoded/{variant}.jxl'
            if sha(source / path) != sha(sanitizer / path):
                raise ValueError('Sanitizer entropy bytes differ from Release')
    path = validation / 'manifest.json'
    frozen = json.loads(path.read_text())
    if frozen['schema'] != 1 or frozen['source_snapshot_is_authoritative'] is not True:
        raise ValueError('Invalid entropy validation freeze')
    for artifact, digest in frozen['artifacts'].items():
        if sha(artifact) != digest:
            raise ValueError('Changed entropy validation artifact')
    checks = (('release', partition), ('asan', sanitizer))
    for build, study in checks:
        if sha(validation / build / 'reference_learning') != sha(study / 'reference_learning'):
            raise ValueError('Tested entropy executable differs from study')
        log = (validation / build / 'ctest.log').read_text()
        if (len(re.findall(r'\d+/10 Test\s+#\d+:.*Passed', log)) != 10 or
                not re.search(r'^100% tests passed(?:, 0 tests failed)? out of 10$', log, re.M)):
            raise ValueError('Incomplete entropy replay test qualification')
    # Keep all measurements, including the separate ramp; aggregate only the
    # original five-image policy cohort. Never construct unmeasured factorial cells.
    map_results = {(r['input'], r['policy']): r for r in maps['results']}
    for row in base['results']:
        extra = map_results[row['input'], row['policy']]
        baseline = next(r['bytes'] for r in row['results'] if r['variant'] == 'reference')
        row['results'] += [r for r in extra['results'] if r['variant'].startswith('reference_map32')]
        for r in row['results']:
            r['delta_bytes_from_reference_entropy'] = r['bytes'] - baseline
            r['delta_percent_from_reference_entropy'] = 100 * (r['bytes'] / baseline - 1)
    for policy, total in base['totals'].items():
        total['variant_bytes'].update({k: v for k, v in maps['totals'][policy]['variant_bytes'].items()
                                       if k.startswith('reference_map32')})
        baseline = total['variant_bytes']['reference']
        total['delta_bytes_from_reference_entropy'] = {k: v - baseline for k, v in total['variant_bytes'].items()}
        total['delta_percent_from_stock'] = {k: 100 * (v / total['stock_bytes'] - 1)
                                             for k, v in total['variant_bytes'].items()}
    base['variants'] = list(VARIANTS)
    base['files'] = len(full_cases) * len(VARIANTS)  # Unique measured case/variant cells.
    base['transplanted_data_scopes_match_full_reference'] = maps['transplanted_data_scopes_match_full_reference']
    base['qualification'] = {
        'release_generated_files_in_two_studies': studies[1]['files'] + 13 * len(full_cases),
        'common_reference_files_identical': len(full_cases),
        'sanitizer_cases': len(sanitizer_cases),
        'sanitizer_files_identical_to_release': sanitized['files'],
        'research_ctest_per_build': 10,
        'validation_manifest': str(path), 'validation_manifest_sha256': sha(path),
        'bound_validation_files': len(frozen['artifacts']),
        'sanitizer_limits': frozen['sanitizer_limits'],
    }
    for study in studies[1:]:
        base['artifacts'].update(study['artifacts'])
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition-study', type=Path)
    parser.add_argument('--sanitizer-study', type=Path)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    optional = (args.partition_study, args.sanitizer_study, args.validation)
    if any(optional) and not all(optional):
        parser.error('Pass partition, sanitizer and validation inputs together')
    result = (qualify(args.study.resolve(), *(p.resolve() for p in optional)) if all(optional)
              else analyze(args.study.resolve()))
    with args.output.open('x') as out: out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
