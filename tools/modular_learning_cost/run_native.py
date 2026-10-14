#!/usr/bin/env python3
"""Serial, hash-bound native learning decomposition and frozen-policy ablations."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binaries',type=Path,required=True);p.add_argument('--ans-study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['profile','pilot','rate','search-ablation','control'],required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a.output.mkdir(parents=True,exist_ok=True)
old=json.loads((a.ans_study/'search/manifest.json').read_text())
old_results={r['input']:r for r in map(json.loads,(a.ans_study/'search/results.jsonl').read_text().splitlines())}
inputs=[c['input'] for c in old['cases']]
pilot={'photo-macan','alpha-ria','gray-patches'}
if a.phase!='rate':inputs=[i for i in inputs if i['name'] in pilot]
variants={'profile':['baseline'],'pilot':['baseline','two','keys','two_keys','skip_constant'], 'rate':['baseline','two','keys','quantized'], 'search-ablation':['keys','two'], 'control':['unprofiled']}[a.phase]
repeats,warmups=(2,1) if a.phase in ('pilot','control') else (1,0)
binaries={v:(a.binaries/('native_'+('baseline' if v=='quantized' else v))).resolve() for v in variants}
manifest=dict(phase=a.phase,inputs=inputs,variants=variants,repeats=repeats,warmups=warmups,cpu_participants=4,prior_manifest_sha256=sha(a.ans_study/'search/manifest.json'),prior_results_sha256=sha(a.ans_study/'search/results.jsonl'),binary_sha256={str(b):sha(b) for b in binaries.values()},source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
m=a.output/'manifest.json'
if m.exists():assert json.loads(m.read_text())==manifest
else:m.write_text(json.dumps(manifest,indent=2)+'\n')
ledger=a.output/'results.jsonl';done=[]
if ledger.exists():
 for row in map(json.loads,ledger.read_text().splitlines()):
  assert sha(row['output'])==row['output_sha256'] and sha(row['trace'])==row['trace_sha256'];done.append((row['input'],row['variant']))
with ledger.open('a',buffering=1) as log:
 for index,item in enumerate(inputs):
  assert sha(item['path'])==item['sha256']
  order=variants[index%len(variants):]+variants[:index%len(variants)]
  for variant in order:
   if (item['name'],variant) in done:continue
   directory=(a.output/(item['name']+'-'+variant)).resolve();directory.mkdir(exist_ok=True)
   policy=a.ans_study/'fixed'/(item['name']+'-learned')/'policy.txt'
   phase='search' if a.phase in ['profile','search-ablation'] else 'quantized' if variant=='quantized' else 'fixed'
   cmd=[str(binaries[variant]),item['path'],str(directory),phase,str(policy.resolve()),'262144','256',str(repeats),str(warmups)]
   start=time.time()
   with (directory/'trace.jsonl').open('w') as stdout,(directory/'stderr.log').open('w') as stderr:subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True)
   trace=list(map(json.loads,(directory/'trace.jsonl').read_text().splitlines()))
   assert len(trace)==repeats+warmups and all(r['decoded_exact'] for r in trace)
   output=directory/'output.jxl'
   assert all(r['bytes']==output.stat().st_size for r in trace)
   control=phase=='search' and variant in ['baseline','keys','skip_constant']
   if control:assert sha(output)==old_results[item['name']]['estimated']['sha256'],item['name']
   row=dict(input=item['name'],variant=variant,phase=phase,output=str(output),output_sha256=sha(output),bytes=output.stat().st_size,trace=str(directory/'trace.jsonl'),trace_sha256=sha(directory/'trace.jsonl'),policy_sha256=sha(policy),command=cmd,started_unix=start,finished_unix=time.time(),saved_control=control)
   log.write(json.dumps(row)+'\n');print(item['name'],variant,row['bytes'],flush=True)
