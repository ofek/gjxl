#!/usr/bin/env python3
"""Paired ANS selection controls with frozen inputs, policies and executable."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--runner', type=Path, required=True)
p.add_argument('--checkpoint-data', type=Path, required=True)
p.add_argument('--input-manifest', type=Path, required=True)
p.add_argument('--profile-record', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--phase', choices=('fixed', 'search', 'timing'), required=True)
p.add_argument('--jobs', type=int, default=1)
a = p.parse_args()
assert a.jobs > 0 and (a.phase == 'search' or a.jobs == 1), 'Timed phases must run serially'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
inputs = json.loads(a.input_manifest.read_text())['input']
prior = {r['input']: r for r in map(json.loads, (a.checkpoint_data/'search/results.jsonl').read_text().splitlines())
         if r['config'] == 'search-checkpoint-262144-256'}
profile = json.loads(a.profile_record.read_text())
large = [dict(v, name='photo-large-'+k, category='photos-rgb8-large') for k,v in profile['inputs'].items()]
cases = []
if a.phase == 'fixed':
    for item in inputs + large:
        cases.append(dict(input=item, policy='gradient', name=item['name']+'-gradient'))
    for item in inputs:
        cases.append(dict(input=item, policy=prior[item['name']]['policy'], name=item['name']+'-learned',
                          expected=prior[item['name']]['output'], expected_sha256=prior[item['name']]['output_sha256']))
elif a.phase == 'search':
    for item in inputs:
        cases.append(dict(input=item, policy='search', name=item['name'],
                          expected=prior[item['name']]['output'], expected_sha256=prior[item['name']]['output_sha256']))
else:
    for item in inputs:
        if item['name'] in ('photo-macan', 'alpha-ria', 'gray-patches'):
            cases.append(dict(input=item, policy='search', name=item['name'],
                              expected=prior[item['name']]['output'], expected_sha256=prior[item['name']]['output_sha256']))
if a.phase == 'fixed':
    for case in cases:
        if case['input']['name'].startswith('photo-large-'):
            path = a.profile_record.parent.parent / 'build/modular-token-performance-20261010/expected' / (case['input']['name']+'-gradient-ans-t4.jxl')
            case.update(expected=str(path), expected_sha256=sha(path))
repeats, warmups = (4, 1) if a.phase == 'fixed' else (1, 0) if a.phase == 'search' else (3, 1)
spec = dict(schema=1, phase=a.phase, cases=cases, jobs=a.jobs, repeats=repeats, warmups=warmups,
            cpu_participants=4, runner=str(a.runner.resolve()), runner_sha256=sha(a.runner),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={str(f):sha(f) for f in Path(__file__).parent.iterdir() if f.is_file()},
            input_manifest_sha256=sha(a.input_manifest), profile_record_sha256=sha(a.profile_record),
            prior_results_sha256=sha(a.checkpoint_data/'search/results.jsonl'))
manifest = out/'manifest.json'
if manifest.exists():
    assert json.loads(manifest.read_text()) == spec
else:
    manifest.write_text(json.dumps(spec,indent=2)+'\n')
ledger = out/'results.jsonl'
rows = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
for r in rows:
    for key in ('exact', 'estimated'):
        assert sha(r[key]['path']) == r[key]['sha256']
    assert sha(r['trace']) == r['trace_sha256']
done = {r['name'] for r in rows}

def run(index, case):
    item = case['input']
    assert sha(item['path']) == item['sha256'], item['name']
    directory = out/case['name']
    directory.mkdir(exist_ok=True)
    policy = case['policy']
    if isinstance(policy, dict):
        lines = [f"{policy['rct']} {len(policy['transforms'])} {len(policy['nodes'])}"]
        lines += [' '.join(str(int(v)) for v in row) for row in policy['transforms']+policy['nodes']]
        policy_file = directory/'policy.txt'
        policy_file.write_text('\n'.join(lines)+'\n')
        policy = str(policy_file)
    phase = 'fixed' if a.phase == 'fixed' else 'search'
    cmd = [str(a.runner.resolve()), item['path'], str(directory), phase, policy,
           str(repeats),str(warmups),'4',str(index % 2)]
    start = time.time()
    with (directory/'runner.jsonl').open('w') as stdout, (directory/'stderr.log').open('w') as stderr:
        subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True)
    entries = [json.loads(l) for l in (directory/'runner.jsonl').read_text().splitlines()]
    result = dict(name=case['name'],input=item['name'],category=item['category'],phase=phase,
                  command=cmd,started_unix=start,finished_unix=time.time(),input_sha256=item['sha256'],
                  trace=str(directory/'runner.jsonl'),trace_sha256=sha(directory/'runner.jsonl'))
    for mode in ('exact','estimated'):
        records = [r for r in entries if r['kind']=='encode' and r['mode']==mode]
        assert len(records)==repeats+warmups+(1 if phase=='fixed' else 0)
        path = directory/(mode+'.jxl')
        assert all(r['decoded_exact'] and r['bytes']==path.stat().st_size for r in records)
        result[mode] = dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    if 'expected' in case:
        assert sha(case['expected']) == case['expected_sha256'] == result['exact']['sha256'],case['name']
        result['saved_exact_control'] = True
    else:
        result['saved_exact_control'] = False
    if phase=='fixed':
        assert all(r['policy']==entries[0]['policy'] for r in entries if r['kind']=='encode')
        diagnostic = {r['mode']:r for r in entries if r['kind']=='replay' and r['diagnostic']}
        e,x=diagnostic['estimated'],diagnostic['exact']
        assert e['exact_simulation_ns']==0 and e['width_candidates']==1
        assert (e['tokens'],e['streams'],e['token_hash'],e['config_candidates'],e['histogram_candidates']) == (x['tokens'],x['streams'],x['token_hash'],x['config_candidates'],x['histogram_candidates'])
    print(case['name'],result['exact']['bytes'],result['estimated']['bytes'],flush=True)
    return result

with ledger.open('a',buffering=1) as log, ThreadPoolExecutor(max_workers=a.jobs) as pool:
    futures = [pool.submit(run,i,c) for i,c in enumerate(cases) if c['name'] not in done]
    for f in as_completed(futures):
        log.write(json.dumps(f.result())+'\n')
