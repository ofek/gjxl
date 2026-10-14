#!/usr/bin/env python3
"""Retain a separately audited replacement pair after an external interruption.

Run under `caffeinate -i` on macOS. The original capture is immutable. This does
not select faster repetitions or subtract pauses: both members are repeated.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from entropy_efficiency import FLAGS, analyze, check_result, command, sha, require, utc
from analyze_adaptive_learning import WORK


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    source = Path(manifest['source'])
    analyze(source)
    require(manifest.get('finished_utc') and not manifest.get('interrupted_utc'), 'Supplement is incomplete')
    require(sha(source / 'manifest.json') == manifest['source_manifest_sha256'], 'Source capture changed')
    original = json.loads((source / 'manifest.json').read_text())
    require(manifest['binary']['sha256'] == original['binary']['sha256'] and
            sha(Path(manifest['binary']['path'])) == original['binary']['sha256'], 'Frozen binary differs')
    inventory = json.loads((root / 'sha256-manifest.json').read_text())
    require(set(inventory) == {str(p.relative_to(root)) for p in root.rglob('*')
        if p.is_file() and p.name != 'sha256-manifest.json'}, 'Supplement inventory differs')
    for path, record in inventory.items():
        require(sha(root / path) == record['sha256'] and (root / path).stat().st_size == record['bytes'],
                f'Supplement artifact differs: {path}')
    require([r['route'] for r in manifest['commands']] == ['exact', 'estimated'], 'Supplement pair is incomplete')
    for record in manifest['commands']:
        require(record['returncode'] == 0, 'Supplement encode failed')
        source_image = original['inputs'][manifest['image']]
        require(record['argv'] == [manifest['binary']['path'], source_image['path'],
            str(root / (record['route'] + '.jxl')), '262144', '256', '1', *FLAGS[record['route']]],
            'Supplement command differs')
        result = json.loads((root / (record['route'] + '.stdout')).read_text())
        check_result(result, record['route'])
        require(result == record['result'], 'Supplement result differs from stdout')
        baseline = next(r for r in original['commands'] if r['image'] == manifest['image'] and
                        r['route'] == record['route'])
        require(sha(root / (record['route'] + '.jxl')) == baseline['output_sha256'] and
                result['policy'] == baseline['result']['policy'], 'Repeated pair changed bytes/policy')
        require(result['candidates'] == baseline['result']['candidates'] and
                all(result['training'][k] == baseline['result']['training'][k] for k in WORK),
                'Repeated pair changed candidate count or training work')
        rows = [json.loads(s) for s in (root / (record['route'] + '-activity.jsonl')).read_text().splitlines()]
        gaps = [(datetime.datetime.fromisoformat(b['time']) - datetime.datetime.fromisoformat(a['time'])).total_seconds()
                for a, b in zip(rows, rows[1:])]
        require(gaps and max(gaps) < 15, 'Supplement has another long observation gap')
    return manifest


def capture(args):
    source = args.source.resolve()
    analyze(source)
    original = json.loads((source / 'manifest.json').read_text())
    root = args.capture.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = root / 'gjxl_modular_search_profile'
    shutil.copy2(original['binary']['path'], binary)
    repo = Path(__file__).resolve().parents[2]
    manifest = dict(schema=1, source=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
        revision=original['revision'], binary=dict(path=str(binary), sha256=sha(binary)),
        image=args.image, reason=args.reason, started_utc=utc(), commands=[],
        scope='Both members repeated in exact/estimated order after documented system sleep. Original raw matrix retained. No timing subtraction or fastest-run selection.',
        environment=[command(a, repo) for a in (['pmset', '-g', 'assertions'], ['pmset', '-g', 'batt'], ['pmset', '-g', 'therm'])])
    def save():
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('JXL_', 'GJXL_'))}
    try:
        for route in ('exact', 'estimated'):
            source_image = original['inputs'][args.image]
            require(sha(Path(source_image['path'])) == source_image['sha256'], 'Input changed')
            argv = [str(binary), source_image['path'], str(root / (route + '.jxl')), '262144', '256', '1', *FLAGS[route]]
            record = dict(route=route, argv=argv, started_utc=utc())
            manifest['commands'].append(record); save()
            print('START', route, flush=True)
            with (root / (route + '.stdout')).open('w') as out, (root / (route + '.stderr')).open('w') as err, \
                 (root / (route + '-activity.jsonl')).open('w') as activity:
                begin = time.monotonic()
                p = subprocess.Popen(argv, cwd=repo, env=env, stdout=out, stderr=err)
                record['pid'] = p.pid; save()
                while p.poll() is None:
                    snapshot = command(['ps', '-Ao', 'pid,pcpu,comm', '-r'], repo)
                    activity.write(json.dumps(dict(time=utc(), processes=snapshot['stdout'].splitlines()[:25])) + '\n')
                    activity.flush()
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
                record.update(returncode=p.returncode, finished_utc=utc(), process_seconds=time.monotonic()-begin)
            require(p.returncode == 0, 'Repeat encode failed')
            record['result'] = json.loads((root / (route + '.stdout')).read_text())
            check_result(record['result'], route)
            save(); print('DONE', route, record['result']['seconds'], flush=True)
        manifest['finished_utc'] = utc(); save()
        inventory = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
                     for p in sorted(root.rglob('*')) if p.is_file()}
        (root / 'sha256-manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
        audit(root)
    except BaseException as error:
        manifest.update(error=str(error), interrupted_utc=utc()); save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'audit'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--image', choices=('1029', '1039', '1049'))
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.mode == 'capture':
        require(args.source and args.image and args.reason, 'Supply source, image, and explicit reason')
        capture(args)
    else:
        result = audit(args.capture.resolve())
        print(json.dumps({r['route']: r['result']['seconds'] for r in result['commands']}, indent=2))
