#!/usr/bin/env python3
"""Validate retained artifacts and summarize the learning-cost experiments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
from statistics import median

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--ans-study', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
artifacts = {}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]

def retain(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None: assert digest == expected, str(path)
    artifacts[str(path)] = digest
    return path

def stats(values):
    return dict(values=values, median=median(values), minimum=min(values), maximum=max(values))

def policy_summary(policy):
    return dict(rct=policy['rct'], transforms=policy['transforms'],
                leaves=sum(n[0] < 0 for n in policy['nodes']),
                leaf_predictors=dict(Counter(n[4] for n in policy['nodes'] if n[0] < 0)),
                sha256=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest())

prior = {r['input']:r for r in rows(retain(a.ans_study/'search/results.jsonl'))}
for r in prior.values(): retain(r['estimated']['path'],r['estimated']['sha256'])
phases = {}
for phase, count in [('profile-cpu',3),('pilot-cpu',15),('rate',64)]:
    root = a.root/phase
    manifest = json.loads(retain(root/'manifest.json').read_text())
    for path, digest in manifest['binary_sha256'].items(): retain(path,digest)
    for item in manifest['inputs']: retain(item['path'],item['sha256'])
    records = rows(retain(root/'results.jsonl'))
    assert len(records) == count, (phase,len(records))
    cases = []
    for r in records:
        output = retain(r['output'],r['output_sha256'])
        trace = rows(retain(r['trace'],r['trace_sha256']))
        retain(Path(r['trace']).with_name('stderr.log'))
        retain(r['command'][4],r['policy_sha256'])
        assert len(trace) == manifest['repeats']+manifest['warmups']
        assert all(t['decoded_exact'] and t['bytes']==output.stat().st_size for t in trace)
        assert all(t['policy']==trace[0]['policy'] for t in trace)
        measured = [t for t in trace if not t['warmup']]
        case = {k:r[k] for k in ('input','variant','phase','bytes','output_sha256','saved_control')}
        case['policy'] = policy_summary(trace[0]['policy'])
        case['matches_saved_search'] = r['output_sha256'] == prior[r['input']]['estimated']['sha256']
        case['complete_wall'] = stats([t['seconds'] for t in measured])
        case['learning_cpu'] = stats([sum(s['cpu'] for s in t['passes']) for t in measured])
        case['learning_wall'] = stats([sum(s['total'] for s in t['passes']) for t in measured])
        case['fit_cpu'] = stats([sum(s['fit_cpu'] for s in t['passes']) for t in measured])
        case['fit_wall'] = stats([sum(s['fit'] for s in t['passes']) for t in measured])
        case['wall_to_cpu'] = [sum(s['total'] for s in t['passes']) /
                               sum(s['cpu'] for s in t['passes']) for t in measured]
        if phase == 'profile-cpu':
            assert case['matches_saved_search'] and r['saved_control']
            t = measured[0]
            case['stages'] = t['stages']
            case['candidates'] = t['candidates']
            case['passes'] = [{**s, 'policy':policy_summary(s['policy'])} for s in t['passes']]
            case['budgets'] = []
            for cap in sorted(set(s['sample_cap'] for s in t['passes'])):
                passes = [s for s in t['passes'] if s['sample_cap']==cap]
                keys = ('samples','total','cpu','small_tree','fit','fit_cpu','setup','parent',
                        'sort','sweep','partition','nodes','node_samples','sorts','sort_items',
                        'constant_properties','constant_items','thresholds','sweep_items')
                case['budgets'].append(dict(sample_cap=cap,passes=len(passes),
                    sum={k:sum(s[k] for s in passes) for k in keys}))
            sums = {k:sum(s[k] for s in t['passes']) for k in ('total','fit','small_tree','sort','sweep')}
            case['fit_fraction_of_encode_wall'] = sums['fit']/t['seconds']
            case['sort_fraction_of_fit_wall'] = sums['sort']/sums['fit']
            case['sweep_fraction_of_fit_wall'] = sums['sweep']/sums['fit']
            case['preparation_wall'] = sums['total']-sums['fit']-sums['small_tree']
        cases.append(case)
    by_key = {(c['input'],c['variant']):c for c in cases}
    if phase != 'profile-cpu':
        for c in cases:
            base = by_key[(c['input'],'baseline')]
            c['byte_delta_from_baseline'] = c['bytes']-base['bytes']
            c['byte_percent_from_baseline'] = 100*(c['bytes']/base['bytes']-1)
            c['learning_cpu_ratio_to_baseline'] = base['learning_cpu']['median']/c['learning_cpu']['median']
            c['matches_baseline_bytes'] = c['output_sha256']==base['output_sha256']
            if c['variant'] in ('keys','skip_constant'):
                assert c['matches_baseline_bytes'] and c['policy']==base['policy']
            if c['variant']=='two_keys':
                two=by_key[(c['input'],'two')]
                assert c['output_sha256']==two['output_sha256'] and c['policy']==two['policy']
    totals = {}
    for variant in manifest['variants']:
        subset = [c for c in cases if c['variant']==variant]
        totals[variant] = dict(bytes=sum(c['bytes'] for c in subset),
                              sum_learning_cpu_medians=sum(c['learning_cpu']['median'] for c in subset))
    phases[phase] = dict(cases=cases,totals=totals)

root = a.root/'reference'
manifest = json.loads(retain(root/'manifest.json').read_text())
retain(manifest['binary'],manifest['binary_sha256'])
provenance = json.loads(retain(a.root/'reference-source/provenance.json').read_text())
for path, digest in provenance.items(): retain(path,digest)
previous_root = Path(next(path for path in provenance if path.endswith('/reference_effort.cpp'))).parent
retain(previous_root/'manifest.json',manifest['prior_manifest_sha256'])
previous_rows = rows(retain(previous_root/'results.jsonl',manifest['prior_results_sha256']))
reference_controls = {r['input']:r for r in previous_rows if r['config']=='base'}
for item in manifest['inputs']: retain(item['path'],item['sha256'])
ref_rows = rows(retain(root/'results.jsonl'))
assert len(ref_rows)==36
groups = defaultdict(list)
for r in ref_rows:
    retain(r['output'],r['output_sha256']);retain(r['trace'],r['trace_sha256'])
    assert r['decoded_exact']
    if r['config']=='base':
        control=reference_controls[r['input']]
        retain(control['output'],control['output_sha256'])
        assert r['output_sha256']==control['output_sha256']
    groups[(r['input'],r['config'])].append(r)
reference = []
for (name, config), rs in groups.items():
    assert len(rs)==4 and len(set(r['output_sha256'] for r in rs))==1
    measured = [r for r in rs if not r['warmup']]
    assert len(measured)==3
    case = dict(input=name,config=config,bytes=rs[0]['encoded_bytes'],
                output_sha256=rs[0]['output_sha256'],leaves=rs[0]['leaves'],
                training=[e for e in rs[0]['events'] if e['event']=='training'],
                complete_wall=stats([r['seconds_diagnostic_only'] for r in measured]))
    for event in ('fit_time','learning_time'):
        for clock in ('seconds','cpu'):
            case[event+'_'+clock] = stats([sum(e[clock] for e in r['events']
                                             if e['event']==event) for r in measured])
    reference.append(case)

control_root=a.root/'control'
control_manifest=json.loads(retain(control_root/'manifest.json').read_text())
for path,digest in control_manifest['binary_sha256'].items():retain(path,digest)
control_rows=rows(retain(control_root/'results.jsonl'))
assert len(control_rows)==3
control=[]
for r in control_rows:
    retain(r['output'],r['output_sha256'])
    trace=rows(retain(r['trace'],r['trace_sha256']))
    assert len(trace)==3 and all(t['decoded_exact'] and not t['passes'] for t in trace)
    assert r['output_sha256']==prior[r['input']]['estimated']['sha256']
    stderr=retain(Path(r['trace']).with_name('stderr.log')).read_text()
    cpu=[json.loads(s[8:]) for s in stderr.splitlines() if s.startswith('CONTROL ')]
    assert len(cpu)==3
    control.append(dict(input=r['input'],bytes=r['bytes'],output_sha256=r['output_sha256'],
                        learning_cpu=stats([t['learning_cpu'] for t in cpu if not t['warmup']]),
                        learning_wall=stats([t['learning_seconds'] for t in trace if not t['warmup']])))

# Bind link inputs and generated overrides to the committed source and frozen
# reference artifacts. Existing qualification records retain archive provenance.
for path in (a.root/'bin/CMakeFiles').glob('*.dir/link.txt'):
    retain(path)
    flags=path.with_name('flags.make')
    if flags.exists(): retain(flags)
    for token in shlex.split(path.read_text()):
        if token.endswith(('.a','.dylib')) and Path(token).is_file(): retain(token)
for sub in ('overrides','reference-source'):
    for path in (a.root/sub).rglob('*'):
        if path.is_file(): retain(path)
for path in Path('tools/modular_learning_cost').glob('*'):
    if path.is_file(): retain(path)
for path in ('src/codec/modular/tree_learning.cpp','src/codec/modular/quantized_tree_learning.cpp',
             'src/codec/modular/tree_learning.h','src/codestream/modular/search.cpp',
             'src/codestream/modular/workflow.cpp','docs/modular-ans-selection-qualification.json'):
    retain(path)
for path in a.root.glob('*.log'): retain(path)
for name in ('pilot/results.jsonl','pilot/manifest.json','host.json'):
    if (a.root/name).exists(): retain(a.root/name)

report = dict(native_base_revision='f27a797',instrumentation_commit='810c52e',
              libjxl_revision='e8ff09762481785938d8e4e01333ed3917571161',
              native=phases,reference=reference,unmodified_native_control=control,
              limitations=[
                  'Instrumented, uncontrolled desktop diagnostics; no release throughput claim.',
                  'CPU time excludes descheduling, not frequency or cache effects.',
                  'First wall-only pilot excluded after very long pauses; CPU-timed rerun retained.',
                  'Frozen-transform ablations change neither transform selection nor full search.',
                  'Quantized mode changes sampling and objectives as well as binning.',
                  'Reference and native sample sets, transforms, growth, and objectives differ.',
                  'Matching sample caps do not match samples or leaf counts.',
                  'Key-sort scratch is unbudgeted research code; production sources unchanged.'
              ],artifact_sha256=artifacts)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v['totals'] for k,v in phases.items()},indent=2))
