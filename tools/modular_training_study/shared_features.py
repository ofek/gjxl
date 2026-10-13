#!/usr/bin/env python3
"""Paired shared-feature captures, with frozen binary/source and artifact audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from analyze_adaptive_learning import WORK
from qualify_entropy_pipeline import REPO, LARGE, FLAGS, basic, require, run, sha, utc, write

SMOKE = ('photo-macan', 'palette-pattern', 'alpha16-ramp')


def feature_flags(shared, schema):
    # The original frozen pilot predates promotion of sharing to the default.
    require(schema in (1, 2), 'Unknown shared-feature capture schema')
    return ['--shared-features'] if shared else ([] if schema == 1 else ['--no-shared-features'])


def schedule():
    rows = [(name, 1, variant) for name in SMOKE for variant in ('control', 'shared')]
    for repeat in (1, 2):
        for index, name in enumerate(LARGE[repeat - 1:] + LARGE[:repeat - 1]):
            variants = ('control', 'shared') if (index + repeat) % 2 else ('shared', 'control')
            rows.extend((name, repeat, variant) for variant in variants)
    return [dict(image=n, repetition=r, variant=v, tag=f'{n}-r{r}-{v}') for n, r, v in rows]


def check_result(result, shared):
    basic(result, 'population')
    require(result['shared_features'] == shared, 'Feature-sharing switch differs')
    g, t = result['adaptive_gpu'], result['training']
    require(g['cpu_fallback_candidates'] == g['learning_feature_fallbacks'] == 0,
            'Unexpected CPU token/feature fallback')
    require(g['feature_preparations'] + g['feature_reuses'] ==
            g['encoded_candidates'] + g['learning_feature_uses'] and
            g['feature_preparations'] == g['evictions'] + 1, 'One-slot feature accounting differs')
    require(g['learning_feature_uses'] == (t['learning_calls'] if shared else 0) and
            t['weighted_prediction_samples'] == (0 if shared else t['population']) and
            t['reused_prediction_samples'] == (t['population'] if shared else 0),
            'Learning repeated or omitted weighted prediction')
    require(0 <= g['learning_feature_seconds'] <= g['feature_seconds'] and
            g['feature_seconds'] - g['learning_feature_seconds'] + g['token_seconds'] <=
            result['stages']['tokens'] + 1e-6 and
            g['learning_feature_seconds'] + sum(t['stages'].values()) <=
            result['stages']['training'] + 1e-6, 'Feature timers overlap or exceed their parent')
    require(shared or g['learning_feature_seconds'] == 0, 'Control prepared learning features')


def semantic_ledger(result):
    return [{k: v for k, v in row.items() if k != 'seconds'} for row in result['candidate_ledger']]


def check_pair(control, shared):
    require(control['policy'] == shared['policy'] and
            control['encoded_bytes'] == shared['encoded_bytes'] and
            semantic_ledger(control) == semantic_ledger(shared), 'Candidate decisions or exact costs changed')
    require(all(control['training'][k] == shared['training'][k] for k in WORK), 'Training work changed')
    require(control['planned_capacity'] == shared['planned_capacity'], 'Admission requirement changed')
    original = control['adaptive_gpu']['feature_samples'] + control['training']['weighted_prediction_samples']
    reused = shared['adaptive_gpu']['feature_samples'] + shared['training']['weighted_prediction_samples']
    require(0 < reused < original, 'Sharing did not eliminate weighted predictor evaluations')


def environment():
    result = {name: run(argv) for name, argv in (
        ('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm']),
        ('processes', ['ps', '-Ao', 'pid,pcpu,comm', '-r']))}
    result['processes']['stdout'] = '\n'.join(result['processes']['stdout'].splitlines()[:20]) + '\n'
    return result


def capture(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prior = json.loads(args.inputs_manifest.read_text())
    binary = root / 'gjxl_modular_search_profile'
    shutil.copy2(args.binary, binary)
    patch = root / 'source.patch'
    patch.write_text(run(['git', 'diff', '--binary', 'HEAD'])['stdout'])
    sources = [Path(__file__), REPO / 'tools/modular_training_study/qualify_entropy_pipeline.py',
               REPO / 'tools/modular_training_study/analyze_adaptive_learning.py']
    for source in sources:
        shutil.copy2(source, root / source.name)
    shutil.copy2(args.binary.parent / 'CMakeCache.txt', root / 'CMakeCache.txt')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GJXL_')}
    manifest = dict(schema=2, complete=False, started_utc=utc(), revision=run(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
        source_patch_sha256=sha(patch), tracked_status=run(['git', 'status', '--short', '--untracked-files=no'])['stdout'],
        binary=dict(path=str(binary), sha256=sha(binary)),
        provenance_files={p.name: sha(p) for p in [patch, root / 'CMakeCache.txt', *[root / p.name for p in sources]]},
        inputs={name: prior['inputs'][name] for name in (*SMOKE, *LARGE)},
        inputs_manifest=dict(path=str(args.inputs_manifest.resolve()), sha256=sha(args.inputs_manifest)),
        system=[run(['sw_vers']), run(['sysctl', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string'])],
        power_policy=run(['pmset', '-g', 'custom']),
        removed_environment_keys=sorted(set(os.environ) - set(env)), schedule=schedule(), commands=[],
        scope='Same frozen binary; complete calls; three small smoke pairs and two pairs per original 12MP photo. '
              'Alternating order, no excluded warmup; backend setup, file IO and independent decoding outside timer. '
              'Uncommitted implementation identified by base revision plus exact source patch. Active desktop pilot.')
    write(root / 'manifest.json', manifest)
    guard = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        for item in manifest['schedule']:
            require(not (root / 'STOP_AFTER_CURRENT').exists(), 'Orderly stop requested')
            source = manifest['inputs'][item['image']]
            require(sha(source['path']) == source['sha256'] and sha(binary) == manifest['binary']['sha256'],
                    'Input or frozen binary changed')
            tag = item['tag']
            output = root / (tag + '.jxl')
            flags = FLAGS['population'] + feature_flags(item['variant'] == 'shared', manifest['schema'])
            argv = [str(binary), source['path'], str(output), '262144', '256', '1', *flags]
            record = dict(item, argv=argv, started_utc=utc(), before=environment())
            manifest['commands'].append(record)
            write(root / 'manifest.json', manifest)
            print('START', tag, flush=True)
            with (root / (tag + '.stdout')).open('w') as out, (root / (tag + '.stderr')).open('w') as err, \
                 (root / (tag + '-activity.jsonl')).open('w') as activity:
                process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=out, stderr=err)
                record['pid'] = process.pid
                begin = time.monotonic()
                while process.poll() is None:
                    activity.write(json.dumps(dict(utc=utc(), **environment())) + '\n')
                    activity.flush()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            record.update(returncode=process.returncode, ended_utc=utc(),
                          process_wall_seconds=time.monotonic() - begin, after=environment())
            require(process.returncode == 0, f'Encoder failed: {tag}')
            result = json.loads((root / (tag + '.stdout')).read_text())
            check_result(result, item['variant'] == 'shared')
            record['files'] = {p.name: sha(p) for p in
                (output, root / (tag + '.stdout'), root / (tag + '.stderr'), root / (tag + '-activity.jsonl'))}
            write(root / 'manifest.json', manifest)
            print('DONE', tag, f"{result['seconds']:.3f}s", result['encoded_bytes'], flush=True)
        manifest['complete'] = True
    finally:
        guard.terminate()
        guard.wait(timeout=5)
        manifest['ended_utc'] = utc()
        write(root / 'manifest.json', manifest)


def audit(root):
    root = root.resolve()
    m = json.loads((root / 'manifest.json').read_text())
    require(m['complete'] and m['schedule'] == schedule() and len(m['commands']) == len(schedule()),
            'Incomplete or changed capture schedule')
    require(sha(m['binary']['path']) == m['binary']['sha256'], 'Frozen binary changed')
    for name, digest in m['provenance_files'].items():
        require(sha(root / name) == digest, f'Provenance changed: {name}')
    for source in m['inputs'].values():
        require(sha(source['path']) == source['sha256'], 'Input changed')
    results, powers = {}, set()
    for record, item in zip(m['commands'], schedule()):
        require(all(record[k] == v for k, v in item.items()) and record['returncode'] == 0,
                'Capture ordering or completion differs')
        tag, shared = record['tag'], record['variant'] == 'shared'
        expected = [m['binary']['path'], m['inputs'][record['image']]['path'], str(root / (tag + '.jxl')),
                    '262144', '256', '1', *FLAGS['population'], *feature_flags(shared, m['schema'])]
        require(record['argv'] == expected, 'Recorded command differs')
        for name, digest in record['files'].items():
            require(sha(root / name) == digest, f'Capture artifact changed: {name}')
        result = json.loads((root / (tag + '.stdout')).read_text())
        check_result(result, shared)
        require((root / (tag + '.jxl')).stat().st_size == result['encoded_bytes'], 'File size differs')
        results[tag] = result
        snapshots = [record['before'], record['after'],
                     *[json.loads(line) for line in (root / (tag + '-activity.jsonl')).read_text().splitlines()]]
        for snapshot in snapshots:
            require(snapshot['power']['returncode'] == snapshot['thermal']['returncode'] == 0,
                    'Missing power/thermal observation')
            powers.add(snapshot['power']['stdout'].splitlines()[0])
    require(len(powers) == 1, 'Power source changed during the paired capture')
    pairs = []
    for item in schedule():
        if item['variant'] != 'control':
            continue
        prefix = item['tag'].removesuffix('control')
        a, b = results[prefix + 'control'], results[prefix + 'shared']
        check_pair(a, b)
        require(sha(root / (prefix + 'control.jxl')) == sha(root / (prefix + 'shared.jxl')),
                'Complete codestream bytes differ')
        pairs.append(dict(image=item['image'], repetition=item['repetition'], control=a, shared=b))
    images = {}
    for name in (*SMOKE, *LARGE):
        rows = [p for p in pairs if p['image'] == name]
        images[name] = {v: dict(seconds=[r[v]['seconds'] for r in rows],
            median_seconds=statistics.median(r[v]['seconds'] for r in rows),
            encoded_bytes=rows[0][v]['encoded_bytes'],
            peak_backing=max(r[v]['peak_backing'] for r in rows),
            planned_capacity=rows[0][v]['planned_capacity'],
            weighted_evaluations=rows[0][v]['adaptive_gpu']['feature_samples'] +
                                 rows[0][v]['training']['weighted_prediction_samples']) for v in ('control', 'shared')}
    ratio = sum(images[n]['control']['median_seconds'] for n in LARGE) / \
            sum(images[n]['shared']['median_seconds'] for n in LARGE)
    def diagnostic(result):
        # Full trees and ledgers remain in the hashed capture stdout. Avoid
        # repeating hundreds of policy nodes for every call in the report.
        compact = {k: v for k, v in result.items() if k not in ('candidate_ledger', 'policy')}
        policy = json.dumps(result['policy'], sort_keys=True, separators=(',', ':')).encode()
        compact['policy_sha256'] = hashlib.sha256(policy).hexdigest()
        return compact
    return dict(schema=1, manifest=str(root / 'manifest.json'), manifest_sha256=sha(root / 'manifest.json'),
                revision=m['revision'], source_patch_sha256=m['source_patch_sha256'],
                audited_calls=len(results), power=sorted(powers), images=images, large_speedup=ratio,
                pairs=[{**p, **{v: diagnostic(p[v]) for v in ('control', 'shared')}} for p in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('root', type=Path)
    cap.add_argument('--binary', type=Path, required=True)
    cap.add_argument('--inputs-manifest', type=Path, required=True)
    check = sub.add_parser('audit')
    check.add_argument('root', type=Path)
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        capture(args)
    else:
        result = audit(args.root)
        write(args.output, result)
        print(f"{result['audited_calls']} calls audited; large-image paired ratio {result['large_speedup']:.3f}x")


if __name__ == '__main__':
    main()
