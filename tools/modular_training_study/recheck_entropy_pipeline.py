#!/usr/bin/env python3
"""Repeat one whole eight-mode comparison after a recorded power transition."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from qualify_entropy_pipeline import FLAGS, REPO, VARIANTS, basic, require, run, sha, utc, write


def on_ac(snapshot):
    return snapshot['returncode'] == 0 and "Now drawing from 'AC Power'" in snapshot['stdout']


def capture(args):
    source = args.source.resolve()
    prior = json.loads((source / 'manifest.json').read_text())
    require(prior['complete'], 'Original capture is incomplete')
    schedule = [r for r in prior['commands'] if r['image'] == args.image and r['repetition'] == args.repetition]
    require(len(schedule) == 8 and {r['variant'] for r in schedule} == set(VARIANTS), 'Recheck must cover all eight modes')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('JXL_', 'GJXL_'))}
    m = dict(schema=1, complete=False, started_utc=utc(), commands=[],
             revision=run(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
             scripts={str(p.relative_to(REPO)): sha(p) for p in
                      (Path(__file__).resolve(), Path(__file__).with_name('qualify_entropy_pipeline.py').resolve())},
             reason=args.reason, source=dict(path=str(source / 'manifest.json'), sha256=sha(source / 'manifest.json')),
             image=args.image, repetition=args.repetition, environment=run(['pmset', '-g', 'batt']))
    require(on_ac(m['environment']), 'Whole comparison recheck requires AC power')
    guard = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        for control in schedule:
            require(not (root / 'STOP_AFTER_CURRENT').exists(), 'Orderly stop requested')
            binary = prior['binaries'][control['variant']]
            require(sha(binary['path']) == binary['sha256'], 'Frozen binary changed')
            image = prior['inputs'][control['image']]
            require(sha(image['path']) == image['sha256'], 'Input changed')
            row = {k: control[k] for k in ('image', 'repetition', 'variant', 'tag', 'input_sha256')}
            argv = [binary['path'], image['path'], str(root / (row['tag'] + '.jxl')),
                    '262144', '256', '1', *FLAGS[row['variant']]]
            row.update(argv=argv, started_utc=utc(), power_before=run(['pmset', '-g', 'batt']))
            require(on_ac(row['power_before']), 'Power source changed before recheck call')
            m['commands'].append(row)
            write(root / 'manifest.json', m)
            print('START', row['tag'], flush=True)
            start = time.monotonic()
            with (root / (row['tag'] + '.stdout')).open('w') as out, \
                 (root / (row['tag'] + '.stderr')).open('w') as err, \
                 (root / (row['tag'] + '-activity.jsonl')).open('w') as activity:
                process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=out, stderr=err)
                row['pid'] = process.pid
                while process.poll() is None:
                    activity.write(json.dumps(dict(utc=utc(), monotonic=time.monotonic(),
                        power=run(['pmset', '-g', 'batt']),
                        processes=run(['ps', '-Ao', 'pid,pcpu,comm', '-r'])['stdout'].splitlines()[:20])) + '\n')
                    activity.flush()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            row.update(returncode=process.returncode, ended_utc=utc(), process_wall_seconds=time.monotonic() - start,
                       power_after=run(['pmset', '-g', 'batt']))
            require(process.returncode == 0 and on_ac(row['power_after']), 'Recheck call failed or power changed')
            result = json.loads((root / (row['tag'] + '.stdout')).read_text())
            basic(result, row['variant'])
            row['result'] = {k: v for k, v in result.items() if k != 'candidate_ledger'}
            row['files'] = {row['tag'] + suffix: dict(bytes=(root / (row['tag'] + suffix)).stat().st_size,
                               sha256=sha(root / (row['tag'] + suffix)))
                            for suffix in ('.jxl', '.stdout', '.stderr', '-activity.jsonl')}
            require(row['files'][row['tag'] + '.jxl'] == control['files'][row['tag'] + '.jxl'], 'Recheck changed output bytes')
            write(root / 'manifest.json', m)
            print('DONE', row['tag'], f"{result['seconds']:.3f}s", result['encoded_bytes'], flush=True)
        m['complete'] = True
    finally:
        guard.terminate(); guard.wait(timeout=5)
        m['ended_utc'] = utc()
        with (root / 'sleep-events.log').open('w') as log:
            subprocess.run(['pmset', '-g', 'log'], stdout=log, stderr=subprocess.STDOUT, check=False)
        m['sleep_events_sha256'] = sha(root / 'sleep-events.log')
        write(root / 'manifest.json', m)


def audit_recheck(root, controls):
    root = root.resolve()
    m = json.loads((root / 'manifest.json').read_text())
    require(m['complete'] and on_ac(m['environment']) and len(m['commands']) == 8, 'Incomplete AC recheck')
    require(sha(m['source']['path']) == m['source']['sha256'], 'Recheck source provenance changed')
    require(sha(root / 'sleep-events.log') == m['sleep_events_sha256'], 'Recheck sleep evidence changed')
    prior = json.loads(Path(m['source']['path']).read_text())
    expected = [r for r in prior['commands'] if r['image'] == m['image'] and r['repetition'] == m['repetition']]
    require([r['tag'] for r in m['commands']] == [r['tag'] for r in expected] and
            {r['variant'] for r in m['commands']} == set(VARIANTS), 'Recheck order/matrix changed')
    original_root = Path(m['commands'][0]['argv'][2]).parent
    for row in m['commands']:
        control = controls[row['image'], row['repetition'], row['variant']]
        require(row['returncode'] == 0 and on_ac(row['power_before']) and on_ac(row['power_after']), 'Recheck status/power changed')
        require(row['input_sha256'] == control['input_sha256'], 'Recheck input changed')
        argv = list(control['argv']); argv[2] = str(original_root / (row['tag'] + '.jxl'))
        require(row['argv'] == argv, 'Recheck arguments changed')
        require(set(row['files']) == set(control['files']), 'Recheck artifact inventory changed')
        for name, info in row['files'].items():
            p = root / name
            require(p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], 'Recheck artifact changed')
        require(row['files'][row['tag'] + '.jxl'] == control['files'][row['tag'] + '.jxl'], 'Recheck file changed')
        result = json.loads((root / (row['tag'] + '.stdout')).read_text())
        require({k: v for k, v in result.items() if k != 'candidate_ledger'} == row['result'], 'Recheck result changed')
        basic(result, row['variant'])
        for key in ('policy', 'candidates', 'encoded_bytes', 'width', 'height', 'channels', 'bits'):
            require(result[key] == control['result'][key], 'Recheck policy/shape/work changed')
        for key in ('learning_calls', 'population', 'samples', 'split_evaluations', 'sorted_properties', 'sorted_samples'):
            require(result['training'][key] == control['result']['training'][key], 'Recheck training work changed')
        # All ledger content other than measured evaluation time must agree.
        def ledger(r):
            return [{k: v for k, v in item.items() if k != 'seconds'} for item in r.get('candidate_ledger', [])]
        require(ledger(result) == ledger(control['result']), 'Recheck candidate ledger changed')
        activity = [json.loads(line) for line in (root / (row['tag'] + '-activity.jsonl')).read_text().splitlines()]
        require(activity and all(on_ac(a['power']) for a in activity), 'Recheck lost AC power')
        require(all(max(b['monotonic'] - a['monotonic'],
                    (datetime.datetime.fromisoformat(b['utc']) - datetime.datetime.fromisoformat(a['utc'])).total_seconds()) < 20
                    for a, b in zip(activity, activity[1:])), 'Recheck heartbeat interrupted')
        row['result'] = result
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--repetition', type=int, required=True)
    p.add_argument('--reason', required=True)
    capture(p.parse_args())
