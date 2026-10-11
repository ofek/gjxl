#!/usr/bin/env python3
"""Frozen transforms first; native factorial and full-search controls per input."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--runner', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--jobs', type=int, default=2)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
baseline = {r['input']: r for r in map(json.loads, args.baseline.read_text().splitlines()) if r['config'] == 'sa'}
args.output.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(item):
    assert sha(Path(item['path'])) == item['sha256']
    prior = baseline[item['name']]
    assert sha(Path(prior['output'])) == prior['output_sha256']
    directory = args.output / item['name']
    directory.mkdir(exist_ok=True)
    # Retain incremental stdout/stderr even if a later cell fails.
    with (directory / 'runner.jsonl').open('w') as out, (directory / 'stderr.log').open('w') as err:
        subprocess.run([str(args.runner.resolve()), item['path'], str(directory.resolve())],
                       stdout=out, stderr=err, check=True)
    rows = []
    for r in map(json.loads, (directory / 'runner.jsonl').read_text().splitlines()):
        path = directory / (r['config'] + '.jxl')
        r.update(input=item['name'], category=item['category'], input_sha256=item['sha256'],
                 output=str(path.resolve()), output_sha256=sha(path))
        if r['config'] == 'baseline':
            assert r['output_sha256'] == prior['output_sha256'], item['name']
        rows.append(r)
    assert len(rows) == 11
    print(item['name'], '11 complete files decoded exactly; baseline byte-identical', flush=True)
    return rows

with ThreadPoolExecutor(max_workers=args.jobs) as pool, (args.output / 'results.jsonl').open('w') as log:
    for rows in pool.map(run, manifest['input']):
        for r in rows:
            log.write(json.dumps(r) + '\n')
        log.flush()
