#!/usr/bin/env python3
"""Run the 64K/256K x 64/256 bounded-reference factorial on frozen controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--reference-data',type=Path,required=True)
p.add_argument('--native-results',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--jobs',type=int,default=2)
a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
data=a.reference_data.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((data/'manifest.json').read_text())
runner=data/'bin/study_runner'
assert sha(runner)==manifest['binary_sha256']
prior={(r['input'],r['config']):r for r in map(json.loads,(data/'results.jsonl').read_text().splitlines())}
native={r['input']:r for r in map(json.loads,a.native_results.read_text().splitlines()) if r['config']=='baseline'}
inputs=manifest['inputs']
configs={'base':{}}
for samples in (65536,262144):
 for leaves in (64,256):
  configs[f's{samples}-l{leaves}']={'STUDY_GLOBAL':'1','STUDY_SAMPLES':str(samples),'STUDY_TARGET_LEAVES':str(leaves)}
spec={'schema':1,'reference_data':str(data),'binary_sha256':sha(runner),'prior_manifest_sha256':sha(data/'manifest.json'),
      'prior_results_sha256':sha(data/'results.jsonl'),'native_results_sha256':sha(a.native_results),
      'configs':configs,'inputs':inputs,'native_rcts':{n:r['policy']['rct'] for n,r in native.items()}}
mp=out/'manifest.json'
if mp.exists(): assert json.loads(mp.read_text())==spec
else: mp.write_text(json.dumps(spec,indent=2)+'\n')
lp=out/'results.jsonl'
rows=[json.loads(s) for s in lp.read_text().splitlines()] if lp.exists() else []
for r in rows:
 assert sha(r['output'])==r['output_sha256'] and sha(r['trace'])==r['trace_sha256']
done={(r['input'],r['config']) for r in rows}
def run(item,config):
 name=item['name']; assert sha(item['path'])==item['sha256']
 directory=out/'outputs'/f'{name}-{config}'; directory.mkdir(parents=True,exist_ok=True)
 env={k:v for k,v in os.environ.items() if not k.startswith('STUDY_') and k!='GJXL_FEATURE_VARIANT'}
 env.update(configs[config])
 if config!='base': env['STUDY_RCT']=str(native[name]['policy']['rct'])
 result=subprocess.run([str(runner),item['path'],'base',str(directory/'output.jxl')],env=env,text=True,capture_output=True)
 (directory/'stderr.log').write_text(result.stderr)
 if result.returncode: raise RuntimeError((name,config,result.stderr[-2000:]))
 row=json.loads(result.stdout)
 events=[json.loads(s[6:]) for s in result.stderr.splitlines() if s.startswith('STUDY ')]
 trace=directory/'trace.json'; trace.write_text(json.dumps(events,indent=2)+'\n')
 baseline_config='base' if config=='base' else 'g-s65536-target64-native-rct'
 saved=prior[name,baseline_config]
 assert sha(saved['output'])==saved['output_sha256'] and sha(saved['trace'])==saved['trace_sha256']
 if config in ('base','s65536-l64'): assert sha(directory/'output.jxl')==saved['output_sha256'],(name,config)
 if config!='base':
  old_events=json.loads(Path(saved['trace']).read_text())
  assert [e for e in events if e['event']=='stream']==[e for e in old_events if e['event']=='stream'],(name,config,'changed transforms/layout')
  assert all(e['samples']<=int(configs[config]['STUDY_SAMPLES']) for e in events if e['event']=='training')
  assert all((len(e['nodes'])+1)//2<=int(configs[config]['STUDY_TARGET_LEAVES']) for e in events if e['event'] in ('tree','local_tree'))
 row.update(input=name,config=config,output=str(directory/'output.jxl'),output_sha256=sha(directory/'output.jxl'),
            trace=str(trace),trace_sha256=sha(trace),rct=None if config=='base' else native[name]['policy']['rct'],
            leaves=[(len(e['nodes'])+1)//2 for e in events if e['event'] in ('tree','local_tree')])
 return row
with lp.open('a',buffering=1) as log,ThreadPoolExecutor(max_workers=a.jobs) as pool:
 futures=[pool.submit(run,i,c) for c in configs for i in inputs if (i['name'],c) not in done]
 for f in as_completed(futures):
  r=f.result(); log.write(json.dumps(r)+'\n'); print(r['input'],r['config'],r['encoded_bytes'],flush=True)
