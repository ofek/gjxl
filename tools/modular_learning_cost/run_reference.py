#!/usr/bin/env python3
"""Serial reference controls and predictor-width ablation, with exact decode."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--previous-data', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
old = json.loads((a.previous_data / 'manifest.json').read_text())
controls = {(r['input'], r['config']): r for r in map(json.loads,
            (a.previous_data / 'results.jsonl').read_text().splitlines())}
inputs = [i for i in old['inputs'] if i['name'] in {'photo-macan','alpha-ria','gray-patches'}]
configs = {'base': {}, 'all-predictors': {'STUDY_ALL_PREDICTORS':'1'},
           'samples-256k': {'STUDY_SAMPLES':'262144'}}
a.output.mkdir(parents=True, exist_ok=True)
manifest = dict(inputs=inputs, configs=configs, repeats=3, warmups=1,
                binary=str(a.binary.resolve()), binary_sha256=sha(a.binary),
                prior_manifest_sha256=sha(a.previous_data / 'manifest.json'),
                prior_results_sha256=sha(a.previous_data / 'results.jsonl'),
                threading='No libjxl thread-pool installed; caller thread only')
m = a.output / 'manifest.json'
if m.exists(): assert json.loads(m.read_text()) == manifest
else: m.write_text(json.dumps(manifest, indent=2)+'\n')
ledger = a.output / 'results.jsonl'
done = set()
if ledger.exists():
    for r in map(json.loads, ledger.read_text().splitlines()):
        assert sha(r['output']) == r['output_sha256'] and sha(r['trace']) == r['trace_sha256']
        done.add((r['input'],r['config'],r['iteration']))
with ledger.open('a', buffering=1) as log:
    for item in inputs:
        assert sha(item['path']) == item['sha256']
        retained = {}
        # Interleave configurations between repetitions; separate process per run.
        for iteration in range(4):
            names = list(configs)
            names = names[iteration % len(names):] + names[:iteration % len(names)]
            for config in names:
                if (item['name'],config,iteration) in done: continue
                directory = (a.output / f"{item['name']}-{config}-{iteration}").resolve()
                directory.mkdir(exist_ok=True)
                output, trace = directory/'output.jxl', directory/'stderr.log'
                env = {k:v for k,v in os.environ.items() if not k.startswith('STUDY_')}
                env.update(configs[config])
                cmd = [str(a.binary.resolve()),item['path'],'base',str(output)]
                with trace.open('w') as err:
                    result = subprocess.run(cmd,env=env,stderr=err,capture_output=False,
                                            stdout=subprocess.PIPE,text=True,check=True)
                (directory/'stdout.json').write_text(result.stdout)
                row = json.loads(result.stdout)
                assert row['decoded_exact']
                output_hash = sha(output)
                if config in retained: assert retained[config] == output_hash
                retained[config] = output_hash
                if config == 'base': assert output_hash == controls[(item['name'],'base')]['output_sha256']
                events = [json.loads(line[6:]) for line in trace.read_text().splitlines()
                          if line.startswith('STUDY ')]
                row.update(input=item['name'],config=config,iteration=iteration,warmup=iteration==0,
                           output=str(output),output_sha256=output_hash,trace=str(trace),
                           trace_sha256=sha(trace),command=cmd,
                           events=[e for e in events if e['event'] in
                                   ('training','fit_time','learning_time','target_capacity')],
                           leaves=[sum(n[0]<0 for n in e['nodes']) for e in events
                                   if e['event'] in ('tree','local_tree') and e['nodes']])
                log.write(json.dumps(row)+'\n')
                print(item['name'],config,iteration,row['encoded_bytes'],flush=True)
