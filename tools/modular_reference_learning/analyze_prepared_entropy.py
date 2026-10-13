#!/usr/bin/env python3
"""Qualify prepared-stream direct ANS against frozen standalone entropy controls."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from analyze_direct_entropy import qualify as qualify_direct
from analyze_entropy_replay import analyze as analyze_rate


def qualify(study, reference, prior, sanitizer, validation, baseline_validation):
    result = qualify_direct(study, reference, sanitizer, validation)
    previous = analyze_rate(prior)
    expected = ['native', 'native_direct_rate', 'native_direct_rate64',
                'native_direct_estimated', 'native_direct_estimated64']
    if result['variants'] != expected or previous['variants'] != expected:
        raise ValueError('Incomplete prepared-stream entropy controls')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (study, prior)]
    if manifests[0]['cases'] != manifests[1]['cases'] or any(m.get('model_timings', False) for m in manifests):
        raise ValueError('Changed preparation cohort or correctness-run protocol')
    measurements, unchanged, sanitizer_files, wide_scopes = [], 0, 0, 0
    for source in (study, sanitizer):
        for line in (source / 'results.jsonl').read_text().splitlines():
            row = json.loads(line)
            for encoded in row['results']:
                name = encoded['variant']
                control = encoded.get('prepared_stream_control')
                if not control or control['exact'] is not True:
                    raise ValueError('Missing integrated encoder comparison')
                path = f"{row['input']}-{row['policy']}/encoded/{name}.jxl"
                if sha(source / path) != sha(prior / path):
                    raise ValueError('Integrated encoder changed previously qualified bytes')
                if source == sanitizer:
                    sanitizer_files += 1
                    continue
                unchanged += 1
                wide_scopes += sum(s['clusters'] > 32 for s in encoded['scopes'] if s['kind'] == 'data')
                measurements.append({'input': row['input'], 'policy': row['policy'], 'variant': name, **control})
    if not wide_scopes:
        raise ValueError('The enlarged model bound was not exercised')

    baseline_manifest = baseline_validation / 'manifest.json'
    bound_baseline = json.loads(baseline_manifest.read_text())['artifacts']
    failures = None
    variants = ('direct32', 'direct64', 'estimated32', 'estimated64')
    names = {f'case-{i}.jxl' for i in range(216)}
    names |= {f'{v}-case-{i}.jxl' for v in variants for i in range(108)}
    for build in ('release', 'asan'):
        log = (validation / build / 'native-ctest.log').read_text()
        if (len(re.findall(r'\d+/5 Test\s+#\d+:.*Passed', log)) != 5 or
                not re.search(r'^100% tests passed(?:, 0 tests failed)? out of 5$', log, re.M)):
            raise ValueError('Incomplete native prepared-stream qualification')
        log = (validation / build / 'native-LastTest.log').read_text()
        current = {(scope, mode): int(count) for scope, mode, count in re.findall(
            r'^(global|local) (\w+) (\d+) allocation failures qualified$', log, re.M)}
        if set(current) != {(s, m) for s in ('global', 'local') for m in ('prefix', 'ans', *variants)}:
            raise ValueError('Incomplete prepared-stream allocation-failure sweep')
        if any(n <= 0 for n in current.values()) or (failures is not None and failures != current):
            raise ValueError('Release/sanitizer allocation paths differ')
        failures = current
        generated = validation / build / 'generated'
        if {p.name for p in generated.iterdir()} != names:
            raise ValueError('Missing generated entropy controls')
        if (validation / build / 'generated.log').read_text().strip() != (
                'Prepared stream reference: 108 cases, 648 exact files'):
            raise ValueError('Generated reference reconstruction incomplete')
        for name in names:
            if sha(generated / name) != sha(validation / 'release/generated' / name):
                raise ValueError('Generated Release/sanitizer bytes differ')
            if name.startswith('case-'):
                old = baseline_validation / build / 'generated' / name
                if sha(old) != bound_baseline[str(old)] or sha(generated / name) != sha(old):
                    raise ValueError('Generated legacy control changed')
    totals = {v: max(r['peak_bytes'] for r in measurements if r['variant'] == v) for v in expected}
    result['qualification'].update({
        'managed_storage_qualified': True, 'prepared_stream_routing_qualified': True,
        'unchanged_standalone_control_files': unchanged, 'native_ctest_per_build': 5,
        'sanitizer_integrated_control_files': sanitizer_files, 'data_scopes_above_32_histograms': wide_scopes,
        'generated_cases': 108, 'generated_files_per_build': len(names),
        'generated_release_sanitizer_identical': len(names), 'unchanged_generated_legacy_files_per_build': 216,
        'allocation_failures_per_build': sum(failures.values()),
        'direct_allocation_failures_per_build': sum(n for (s, m), n in failures.items() if m in variants),
        'allocation_failures_by_scope': {s: {m: failures[s, m] for m in ('prefix', 'ans', *variants)}
                                         for s in ('global', 'local')},
        'baseline_validation_manifest': str(baseline_manifest),
        'baseline_validation_manifest_sha256': sha(baseline_manifest),
    })
    result['prepared_stream_storage'] = {'maximum_measured_peak_bytes_by_variant': totals, 'results': measurements}
    result['scope'] = ('Private prepared-stream direct rate ANS, exact/estimated selection at caps 32/64. '
                       'Byte-identical to frozen standalone controls on captured e9 decisions. '
                       'Managed storage and atomic failure qualified; borrowed inputs excluded. '
                       'No autonomous transform/training policy, whole-encoder timing or public-default qualification.')
    result['artifacts'].update(previous['artifacts'])
    result['analyzer_sha256'] = sha(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    keys = ('study', 'reference-study', 'prior-study', 'sanitizer-study', 'validation', 'baseline-validation', 'output')
    for key in keys:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(*(getattr(args, k.replace('-', '_')).resolve() for k in keys[:-1]))
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
