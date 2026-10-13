#!/usr/bin/env python3
"""Join rate, counter, storage-test and serial-timing evidence for direct ANS estimates."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from analyze_direct_entropy import qualify as qualify_direct
from analyze_entropy_replay import analyze as analyze_rate
from entropy_model_timing import analyze as analyze_timing


def qualify(study, reference, exact, sanitizer, timing, validation):
    result = qualify_direct(study, reference, sanitizer, validation)
    old = analyze_rate(exact)
    times = analyze_timing(timing)
    if Path(json.loads((timing / 'manifest.json').read_text())['rate_study']) != study:
        raise ValueError('Timing used a different estimated-rate study')
    expected = ['native', 'native_direct_rate', 'native_direct_rate64',
                'native_direct_estimated', 'native_direct_estimated64']
    if result['variants'] != expected:
        raise ValueError('Incomplete estimated-width controls')
    if {(r['input'], r['policy']) for r in result['results']} != {
            (r['input'], r['policy']) for r in old['results']}:
        raise ValueError('Original exact control cohort differs')
    original = json.loads((exact / 'manifest.json').read_text())
    current = json.loads((study / 'manifest.json').read_text())
    if original['coding_study'] != current['coding_study'] or current.get('model_timings', False):
        raise ValueError('Changed input captures or profiled-rate protocol')
    counters = {cap: {'data_scopes': 0, 'estimated_width_models': 0, 'estimated_cost_evaluations': 0,
                      'exact_width_models': 0, 'exact_cost_evaluations': 0} for cap in ('32', '64')}
    deltas, identical = [], 0
    for line in (study / 'results.jsonl').read_text().splitlines():
        row = json.loads(line); name = f"{row['input']}-{row['policy']}"
        for variant in expected[:3]:
            path = name + '/encoded/' + variant + '.jxl'
            if sha(study / path) != sha(exact / path):
                raise ValueError('The API change altered an exact/default control')
            identical += 1
        variants = {r['variant']: r for r in row['results']}
        for suffix in ('', '64'):
            cap = suffix or '32'
            a, b = variants['native_direct_rate' + suffix], variants['native_direct_estimated' + suffix]
            scopes = {(s['kind'], s['id']): s for s in a['scopes']}
            for s in b['scopes']:
                if s['kind'] != 'data': continue
                before = scopes['data', s['id']]
                if s['context_map'] != before['context_map']:
                    raise ValueError('Estimated selection changed grouping')
                if s['ans_work']['token_cost_candidates'] != 0 or s['ans_work']['width_candidates'] != 1:
                    raise ValueError('Estimated selection measured candidates')
                counters[cap]['data_scopes'] += 1
                for side, scope in (('estimated', s), ('exact', before)):
                    counters[cap][side + '_width_models'] += scope['ans_work']['width_candidates']
                    counters[cap][side + '_cost_evaluations'] += scope['ans_work']['token_cost_candidates']
            delta = b['bytes'] - a['bytes']
            deltas.append({'input': row['input'], 'policy': row['policy'], 'cluster_cap': int(cap),
                           'exact_bytes': a['bytes'], 'estimated_bytes': b['bytes'], 'delta_bytes': delta,
                           'delta_percent': 100 * delta / a['bytes']})
    for build in ('release', 'asan'):
        log = (validation / build / 'native-ctest.log').read_text()
        if (len(re.findall(r'\d+/6 Test\s+#\d+:.*Passed', log)) != 6 or
                not re.search(r'^100% tests passed(?:, 0 tests failed)? out of 6$', log, re.M)):
            raise ValueError('Incomplete native optimizer/storage qualification')
        detail = (validation / build / 'native-LastTest.log').read_text()
        for variant, count in ((14, 88), (15, 91)):
            if f'Variant {variant}: {count} failures checked' not in detail:
                raise ValueError('Missing estimated direct-ANS allocation-failure sweep')
    result['estimated_width'] = {'unchanged_exact_default_files': identical, 'counters_by_cap': counters,
                                 'rate_differences': deltas,
                                 'maximum_absolute_delta_bytes': max(abs(r['delta_bytes']) for r in deltas),
                                 'maximum_increase_percent': max(r['delta_percent'] for r in deltas)}
    result['qualification'].update({'native_ctest_per_build': 6, 'new_mode_allocation_failures_per_build': 179,
                                     'component_storage_bound_qualified': True,
                                     'prepared_stream_routing_qualified': False})
    result['scope'] = ('Direct rate ANS with estimated width selection. Exact outputs and upstream grouping/search '
                       'controls retained; no ordered cost evaluation without a requested cost. '
                       'Captured e9 decisions, not native transform/training policy. Private optimizer storage '
                       'qualified; stream integration and whole-encoder timing remain unqualified.')
    result['model_timing'] = times
    result['artifacts'].update(old['artifacts'])
    result['analyzer_sha256'] = sha(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'reference-study', 'exact-study', 'sanitizer-study', 'timing-study', 'validation', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(*(p.resolve() for p in (args.study, args.reference_study, args.exact_study,
                                             args.sanitizer_study, args.timing_study, args.validation)))
    with args.output.open('x') as out: out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
