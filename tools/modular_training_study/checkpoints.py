#!/usr/bin/env python3
"""Record checkpoint curves and complete-search runs with saved byte controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--runner',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--prior',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--phase',choices=('checkpoints','checkpoint-search'),required=True)
p.add_argument('--jobs',type=int,default=2)
a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=json.loads(a.manifest.read_text())['input']
prior={(r['input'],r['config']):r for r in map(json.loads,a.prior.read_text().splitlines())}
spec={'schema':1,'phase':a.phase,'inputs':inputs,'binary_sha256':sha(a.runner),
      'prior':str(a.prior.resolve()),'prior_sha256':sha(a.prior),'input_manifest_sha256':sha(a.manifest)}
mp=out/'manifest.json'
if mp.exists(): assert json.loads(mp.read_text())==spec
else: mp.write_text(json.dumps(spec,indent=2)+'\n')
lp=out/'results.jsonl'
rows=[json.loads(s) for s in lp.read_text().splitlines()] if lp.exists() else []
for r in rows: assert sha(r['output'])==r['output_sha256']
expected=11 if a.phase=='checkpoints' else 5
done={name for name in {r['input'] for r in rows} if sum(r['input']==name for r in rows)==expected}
def run(item):
 name=item['name']; assert sha(item['path'])==item['sha256']
 directory=out/name; directory.mkdir(exist_ok=True)
 with (directory/'runner.jsonl').open('w') as stdout,(directory/'stderr.log').open('w') as stderr:
  subprocess.run([str(a.runner.resolve()),item['path'],str(directory),a.phase],stdout=stdout,stderr=stderr,check=True)
 result=[]
 for r in map(json.loads,(directory/'runner.jsonl').read_text().splitlines()):
  file=directory/(r['config']+'.jxl')
  r.update(input=name,category=item['category'],input_sha256=item['sha256'],output=str(file),output_sha256=sha(file))
  old_config={'baseline':'baseline','search-control-65536-64':'search-legacy',
              'checkpoint-65536-16':'legacy-65536-16','checkpoint-65536-64':'legacy-65536-64'}.get(r['config'])
  if old_config:
   old=prior[name,old_config]
   assert sha(old['output'])==old['output_sha256'] and r['output_sha256']==old['output_sha256'],(name,r['config'])
  if a.phase=='checkpoints' and r['config']!='baseline':
   assert r['policy']['rct']==prior[name,'baseline']['policy']['rct']
   assert r['policy']['transforms']==prior[name,'baseline']['policy']['transforms']
  result.append(r)
 assert len(result)==expected
 print(name,a.phase,expected,'files decoded exactly; saved controls byte-identical',flush=True)
 return result
with lp.open('a',buffering=1) as log,ThreadPoolExecutor(max_workers=a.jobs) as pool:
 for f in as_completed([pool.submit(run,i) for i in inputs if i['name'] not in done]):
  for r in f.result():log.write(json.dumps(r)+'\n')
