#!/usr/bin/env python3
"""Verify measured checkpoint curves, full searches and reference factorial."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--prior',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
 rows=[json.loads(s) for s in path.read_text().splitlines()]
 for r in rows:
  assert r['decoded_exact'] and sha(r['output'])==r['output_sha256']
  assert Path(r['output']).stat().st_size==r['encoded_bytes']
  if 'trace' in r:assert sha(r['trace'])==r['trace_sha256']
 assert len({(r['input'],r['config']) for r in rows})==len(rows)
 return {(r['input'],r['config']):r for r in rows}
fixed=read(a.data/'fixed/results.jsonl'); search=read(a.data/'search/results.jsonl')
reference=read(a.data/'reference/results.jsonl'); prior=read(a.prior)
names=sorted({n for n,c in fixed})
fixed_configs=['baseline']+[f'checkpoint-{s}-{l}' for s in (65536,262144) for l in (16,32,64,128,256)]
search_configs=['baseline','search-control-65536-64','search-checkpoint-65536-64',
                'search-checkpoint-65536-256','search-checkpoint-262144-256']
for cells,configs in ((fixed,fixed_configs),(search,search_configs)):
 assert set(cells)=={(n,c) for n in names for c in configs}
for n in names:
 assert fixed[n,'baseline']['output_sha256']==search[n,'baseline']['output_sha256']==prior[n,'baseline']['output_sha256']
 assert search[n,'search-control-65536-64']['output_sha256']==prior[n,'search-legacy']['output_sha256']
 for c,old in (('checkpoint-65536-16','legacy-65536-16'),('checkpoint-65536-64','legacy-65536-64')):
  assert fixed[n,c]['output_sha256']==prior[n,old]['output_sha256']
 previous=search[n,'search-control-65536-64']['encoded_bytes']
 for c in search_configs[2:4]:
  assert search[n,c]['encoded_bytes']<=previous
  previous=search[n,c]['encoded_bytes']
 for c in search_configs[2:]: assert search[n,c]['encoded_bytes']<=search[n,'baseline']['encoded_bytes']
 for s in (65536,262144):
  for l in (16,32,64,128,256):
   r=fixed[n,f'checkpoint-{s}-{l}']
   assert (len(r['policy']['nodes'])+1)//2<=l and r['coverage']['samples']<=s
png=[n for n in names if fixed[n,'baseline']['category']!='synthetic']
assert set(reference)=={(n,c) for n in png for c in ('base','s65536-l64','s65536-l256','s262144-l64','s262144-l256')}
lib=sum(reference[n,'base']['encoded_bytes'] for n in png)
summary={'schema':1,'files':{'native_fixed':len(fixed),'native_search':len(search),'reference':len(reference)},
         'png_reference_bytes':lib,'groups':{},'reference':{},'per_image':[]}
for group,group_names in (('png',png),('synthetic',[n for n in names if n not in png]),('all',names)):
 values={}
 for s in (65536,262144):
  for l in (16,32,64,128,256):
   c=f'checkpoint-{s}-{l}'
   raw=sum(fixed[n,c]['encoded_bytes'] for n in group_names)
   selected=sum(min([fixed[n,'baseline']['encoded_bytes']]+
                    [fixed[n,f'checkpoint-{s}-{k}']['encoded_bytes'] for k in (16,32,64,128,256) if k<=l]) for n in group_names)
   values[c]={'raw_bytes':raw,'checkpoint_selection_bytes':selected}
 searches={c:sum(search[n,c]['encoded_bytes'] for n in group_names) for c in search_configs}
 interactions={key:values['checkpoint-65536-256'][key]+values['checkpoint-262144-64'][key]
                     -values['checkpoint-65536-64'][key]-values['checkpoint-262144-256'][key]
               for key in ('raw_bytes','checkpoint_selection_bytes')}
 summary['groups'][group]={'input':group_names,'fixed':values,'search':searches,
                           'positive_factorial_interaction_bytes':interactions}
for c in ('base','s65536-l64','s65536-l256','s262144-l64','s262144-l256'):
 total=sum(reference[n,c]['encoded_bytes'] for n in png)
 summary['reference'][c]={'bytes':total,'gap_percent':100*(total/lib-1)}
r=summary['reference']
summary['reference_positive_factorial_interaction_bytes']=r['s65536-l256']['bytes']+r['s262144-l64']['bytes']-r['s65536-l64']['bytes']-r['s262144-l256']['bytes']
for n in names:
 before=search[n,'search-control-65536-64']['encoded_bytes']
 item={'input':n,'category':fixed[n,'baseline']['category'],
       'reference_bytes':reference[n,'base']['encoded_bytes'] if n in png else None,
       'search':{c:{'bytes':search[n,c]['encoded_bytes'],'leaves':(len(search[n,c]['policy']['nodes'])+1)//2,
                    'candidates':search[n,c]['candidates']} for c in search_configs},
       'regressed_vs_previous_64k':search[n,search_configs[-1]]['encoded_bytes']>before,
       'fixed':{c:{'bytes':fixed[n,c]['encoded_bytes'],'leaves':(len(fixed[n,c]['policy']['nodes'])+1)//2}
                for c in fixed_configs[1:]}}
 summary['per_image'].append(item)
a.output.write_text(json.dumps(summary,indent=2)+'\n')
print('PNG search:')
for c,v in summary['groups']['png']['search'].items():print(c,v,f'{100*(v/lib-1):+.6f}%')
print('Regressions of 256K/256 vs previous 64K/64:',[i['input'] for i in summary['per_image'] if i['regressed_vs_previous_64k']])
