#!/usr/bin/env python3
"""Validate retained paired trials and summarize their rate/timing trade-off."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--prior-report', type=Path, required=True)
p.add_argument('--reference-ledger', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read_rows = lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()]
artifacts = {}

def retain(path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None:
        assert actual == expected, str(path)
    artifacts[str(path)] = actual
    return path

prior = json.loads(a.prior_report.read_text())
retain(a.prior_report)
retain(a.reference_ledger, prior['artifact_sha256'][str(a.reference_ledger.resolve())])
reference = {r['input']: r for r in read_rows(a.reference_ledger) if r['config'] == 'l9'}
# Accept only the known saved reference configuration; do not infer missing cells.
assert len(reference) == 16, sorted(set(r['config'] for r in read_rows(a.reference_ledger)))
for row in reference.values():
    path = retain(row['output'], row['output_sha256'])
    assert path.stat().st_size == row['encoded_bytes'] and row['decoded_exact']
by_input = {r['input']: r for r in prior['rate']['per_image']}
for name, r in reference.items():
    if by_input[name]['reference_bytes'] is not None:
        assert r['encoded_bytes'] == by_input[name]['reference_bytes']
png = set(prior['rate']['groups']['png']['input'])
qual_path = Path('docs/modular-ans-selection-qualification.json')
qual = json.loads(qual_path.read_text())
retain(qual_path)
for path, digest in qual['source_sha256'].items():
    assert sha(path) == digest, path
for path, digest in qual['artifact_sha256'].items():
    retain(path, digest)
phases = {}
counts = dict(saved_files=0, complete_calls=0, isolated_model_calls=0, saved_exact_controls=0)
for phase, expected_count in [('fixed', 35), ('search', 16), ('timing', 3)]:
    directory = a.root / phase
    manifest_path = retain(directory / 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    retain(manifest['runner'], manifest['runner_sha256'])
    for source, digest in manifest['source_sha256'].items():
        frozen = a.root / 'measurement-source' / Path(source).name
        retain(frozen if frozen.exists() else source, digest)
    retain(directory / 'results.jsonl')
    rows = read_rows(directory / 'results.jsonl')
    assert len(rows) == expected_count
    cases = {c['name']: c for c in manifest['cases']}
    assert set(cases) == {r['name'] for r in rows} and len(cases) == len(rows)
    summaries = []
    for row in sorted(rows, key=lambda r: r['name']):
        case = cases[row['name']]
        retain(case['input']['path'], case['input']['sha256'])
        trace = read_rows(retain(row['trace'], row['trace_sha256']))
        retain(Path(row['trace']).with_name('stderr.log'))
        policy_file = Path(row['trace']).with_name('policy.txt')
        if policy_file.exists():
            retain(policy_file)
        summary = {k: row[k] for k in ['name', 'input', 'category', 'saved_exact_control']}
        summary['bytes'] = {mode: row[mode]['bytes'] for mode in ['exact', 'estimated']}
        summary['delta_bytes'] = row['estimated']['bytes'] - row['exact']['bytes']
        summary['delta_percent'] = 100 * summary['delta_bytes'] / row['exact']['bytes']
        if 'expected' in case:
            retain(case['expected'], case['expected_sha256'])
            assert row['exact']['sha256'] == case['expected_sha256']
            counts['saved_exact_controls'] += 1
        for mode in ['exact', 'estimated']:
            output = retain(row[mode]['path'], row[mode]['sha256'])
            assert output.stat().st_size == row[mode]['bytes']
            encodes = [r for r in trace if r['kind'] == 'encode' and r['mode'] == mode]
            assert len(encodes) == manifest['repeats'] + manifest['warmups'] + (phase == 'fixed')
            assert all(r['decoded_exact'] and r['bytes'] == output.stat().st_size for r in encodes)
            assert all(r['policy'] == encodes[0]['policy'] for r in encodes)
            if phase != 'fixed':
                assert all(r['candidates'] == encodes[0]['candidates'] for r in encodes)
            counts['saved_files'] += 1
            counts['complete_calls'] += len(encodes)
        summary['policy_changed'] = next(r['policy'] for r in trace if r['kind']=='encode' and r['mode']=='exact') != next(r['policy'] for r in trace if r['kind']=='encode' and r['mode']=='estimated')
        if phase != 'fixed':
            summary['candidate_count_changed'] = next(r['candidates'] for r in trace if r['mode']=='exact') != next(r['candidates'] for r in trace if r['mode']=='estimated')
        if phase == 'fixed':
            assert not summary['policy_changed']
            diag = {r['mode']: r for r in trace if r['kind']=='replay' and r['diagnostic']}
            x,e = diag['exact'], diag['estimated']
            assert e['exact_simulation_ns'] == 0 and e['width_candidates'] == 1
            for key in ['tokens','streams','token_hash','config_candidates','histogram_candidates','clusters']:
                assert x[key] == e[key]
            summary['replay_diagnostic'] = diag
        for kind in ['encode'] + (['replay'] if phase == 'fixed' else []):
            timing = {}
            for mode in ['exact', 'estimated']:
                calls = [r for r in trace if r['kind']==kind and r['mode']==mode]
                if kind == 'replay':
                    assert len(calls) == manifest['repeats'] + manifest['warmups'] + 1
                    counts['isolated_model_calls'] += len(calls)
                measured = [r for r in calls if not r['warmup'] and not r['diagnostic']]
                assert len(measured) == manifest['repeats']
                values = [r['seconds'] for r in measured]
                timing[mode] = dict(seconds=values,median_seconds=median(values),min_seconds=min(values),max_seconds=max(values))
                if kind == 'encode':
                    profiled = [r for r in calls if r['profiled'] and not r['warmup']]
                    timing[mode]['profiled_stages_median_seconds'] = [median(r['stages'][i] for r in profiled) for i in range(7)]
                    timing[mode]['candidates'] = [r['candidates'] for r in profiled]
            timing['ratio_of_medians'] = timing['exact']['median_seconds'] / timing['estimated']['median_seconds']
            timing['qualified_scope'] = 'Concurrent rate-only diagnostic; not a speed measurement' if phase=='search' else 'Serial pairs in one uncontrolled desktop session; adaptive timings are instrumented' if phase=='timing' else 'Serial pairs in one uncontrolled desktop session; profiling disabled'
            summary[kind + '_timing'] = timing
        summaries.append(summary)
    phases[phase] = dict(manifest=str(manifest_path), cases=summaries)

counts['fixed_same_token_partition_pairs'] = len(phases['fixed']['cases'])
counts['estimated_replays_without_exact_simulation'] = len(phases['fixed']['cases'])
counts['unchanged_adaptive_winning_policies'] = sum(not r['policy_changed'] for r in phases['search']['cases'])

fixed_learned = {r['input']: r for r in read_rows(a.root/'fixed/results.jsonl') if r['name'].endswith('-learned')}
# Record agreement with the separately frozen learned policies, without treating
# unchanged adaptive winners as a general contract of estimated selection.
counts['adaptive_files_matching_fixed_learned'] = sum(
    r[mode]['sha256'] == fixed_learned[r['input']][mode]['sha256']
    for r in read_rows(a.root/'search/results.jsonl') for mode in ['exact','estimated'])
# The separately measured adaptive pilot must reproduce the full rate sweep.
search_rows = {r['input']:r for r in read_rows(a.root/'search/results.jsonl')}
for row in read_rows(a.root/'timing/results.jsonl'):
    for mode in ['exact','estimated']:
        assert row[mode]['sha256'] == search_rows[row['input']][mode]['sha256']

def aggregate(rows, timing=True, with_reference=False):
    totals = {mode: sum(r['bytes'][mode] for r in rows) for mode in ['exact','estimated']}
    result = dict(inputs=[r['name'] for r in rows],bytes=totals,
                  delta_bytes=totals['estimated']-totals['exact'],
                  delta_percent=100*(totals['estimated']-totals['exact'])/totals['exact'])
    if timing:
        for kind in ['encode', 'replay']:
            if all(kind+'_timing' in r for r in rows):
                sums = {m: sum(r[kind+'_timing'][m]['median_seconds'] for r in rows) for m in ['exact','estimated']}
                result[kind+'_timing'] = dict(sum_per_case_median_seconds=sums,ratio=sums['exact']/sums['estimated'])
    if with_reference:
        ref = sum(reference[r['input']]['encoded_bytes'] for r in rows)
        result['libjxl_e9_bytes'] = ref
        result['gap_percent_vs_e9'] = {m: 100*(totals[m]-ref)/ref for m in totals}
    return result

fixed=phases['fixed']['cases'];search=phases['search']['cases']
aggregates = {
    'fixed_gradient_16': aggregate([r for r in fixed if r['name'].endswith('-gradient') and not r['name'].startswith('photo-large-')]),
    'fixed_gradient_large_3': aggregate([r for r in fixed if r['name'].startswith('photo-large-')]),
    'fixed_learned_16': aggregate([r for r in fixed if r['name'].endswith('-learned')],with_reference=True),
    'adaptive_all_16': aggregate(search,timing=False,with_reference=True),
    'adaptive_png_11': aggregate([r for r in search if r['input'] in png],timing=False,with_reference=True),
    'adaptive_timing_pilot_3': aggregate(phases['timing']['cases'])}
for phase in ['fixed','search']:
    cases=phases[phase]['cases']
    phases[phase]['worst_delta_bytes'] = max(cases, key=lambda r:r['delta_bytes'])['name']
    phases[phase]['worst_delta_percent'] = max(cases, key=lambda r:r['delta_percent'])['name']
    phases[phase]['changed_policies'] = [r['name'] for r in cases if r['policy_changed']]
for path in a.root.glob('host-*-snapshot.json'):
    retain(path)
for name in ['fixed.log','search.log','timing.log','study-build-final.log','study/CMakeCache.txt','study/CMakeFiles/ans_runner.dir/link.txt']:
    path=a.root/name
    if path.exists(): retain(path)
for token in shlex.split((a.root/'study/CMakeFiles/ans_runner.dir/link.txt').read_text()):
    if token.endswith(('.a','.dylib')) and Path(token).is_file(): retain(token)
retain(a.root/'study/CMakeFiles/ans_runner.dir/flags.make')
dependency_record = a.root/'linked-dependencies.json'
if dependency_record.exists():
    for path, digest in json.loads(retain(dependency_record).read_text())['sha256'].items():
        retain(path, digest)
host_path = a.root/'host-search-snapshot.json'
host = json.loads(host_path.read_text()) if host_path.exists() else dict(platform='not recorded',hardware='not recorded')
result = dict(schema=1,platform=host['platform'],hardware=host['hardware'],implementation_revision='a64510e4270ccda6fd367e57301ca3a84f1cddb2',
    base_revision=qual['base_revision'],worktree=str(Path.cwd()),branch=qual['branch'],
    libjxl_revision=prior['libjxl_revision'],profile_stage_order=['input','transforms','training','tokens','model','emission','assembly'],qualification=counts,
    test_qualification={k:qual[k] for k in ['release_tests','sanitizer_tests','workflow_failure_positions']},
    aggregates=aggregates,phases=phases,
    limitations=[
        'Small related-image corpus; no global rate guarantee or public default promotion.',
        'Complete encodes allow four CPU participants; isolated ANS calls use the calling thread. Fixed timings use four measured pairs; adaptive pilot uses three. Warmups and fixed diagnostic calls excluded.',
        'Timing ratios divide sums of per-case medians; they are single-session observations, not confidence-bounded throughput claims.',
        'An unrelated build, another Modular study, browsers and OS activity were observed during this study. No controlled-machine or thermal-stability claim.',
        'Concurrent adaptive rate-sweep timings are diagnostic only. Adaptive pilot timings have Modular profiling enabled.',
        'Each first output per case/mode is independently decoded; repeated outputs must match it byte-for-byte.',
        'Reference e9 bytes are verified saved files, not new reference encodes or speed measurements.',
        'CPU-only. No hardware GPU execution or timing qualification.'],
    implementation_source_sha256=qual['source_sha256'],
    analysis_source_sha256={str(Path(__file__)):sha(__file__)},
    final_tool_source_sha256={str(f):sha(f) for f in Path('tools/modular_ans_study').iterdir() if f.is_file()},artifact_sha256=dict(sorted(artifacts.items())))
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(qualification=counts,aggregates=aggregates),indent=2))
