#!/usr/bin/env python3
"""Capture and independently audit complete-search estimated ANS qualification."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from analyze_adaptive_learning import WORK, distribution, require, same_policy, sha

IMAGES = ('1029', '1039', '1049')
FLAGS = {'cpu-estimated': ['--estimated-ans'], 'exact': ['--metal-learn'],
         'estimated': ['--metal-learn', '--estimated-ans']}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(argv, cwd):
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    return dict(argv=[str(x) for x in argv], returncode=p.returncode,
                stdout=p.stdout, stderr=p.stderr)


def schedule():
    rows = [(image, 0, 'cpu-estimated') for image in IMAGES]
    for repeat, images in enumerate((IMAGES, IMAGES[1:] + IMAGES[:1]), 1):
        for image in images:
            routes = ('exact', 'estimated')
            if (IMAGES.index(image) + repeat) % 2 == 0:
                routes = tuple(reversed(routes))
            rows.extend((image, repeat, route) for route in routes)
    return [dict(image=i, repetition=n, route=r, tag=f'{i}-r{n}-{r}') for i, n, r in rows]


def check_result(r, route):
    estimated = route != 'exact'
    cpu = route == 'cpu-estimated'
    require((r['maximum_samples'], r['maximum_leaves'], r['checkpoints'], r['threads'],
             r['profile_enabled'], r['decoded_exact'], r['entropy'], r['ans_selection']) ==
            (262144, 256, True, 1, True, True, 'ans', 'estimated' if estimated else 'exact'),
            'Encoder settings or independent decoding differ')
    require((r['width'], r['height'], r['channels'], r['bits']) == (4000, 3000, 3, 8),
            'Original full-resolution input differs')
    require(r['seconds'] > 0 and r['peak_backing'] <= r['planned_capacity'] and
            r['peak_cpu_participants'] <= 1 and r['candidates'] > 0, 'Resources or timing differ')
    require(sum(r['stages'].values()) <= r['seconds'] + 1e-5 and
            sum(r['training']['stages'].values()) <= r['stages']['training'] + 1e-5,
            'Overlapping or invalid phase timing')
    g = r['adaptive_gpu']
    if estimated:
        require(r['entropy_work']['ans_token_cost_ns'] == 0 and
                r['entropy_work']['ans_alphabet_width_candidates'] == r['candidates'] and
                all(g[k] == 0 for k in ('ans_gpu_groups', 'ans_gpu_states',
                    'ans_cpu_fallback_groups', 'ans_cpu_small_groups')),
                'Estimated selection invoked exact costing or constructed extra widths')
    if cpu:
        require(r['route'] == 'cpu-complete-search' and
                r['gpu_submissions'] == r['gpu_allocations'] == 0 and
                all(v == 0 for v in g.values()), 'CPU control invoked the GPU')
        return
    require(r['route'] == 'hybrid-learning-complete-search' and
            g['encoded_candidates'] == r['candidates'] and g['cpu_fallback_candidates'] == 0 and
            r['gpu_submissions'] == r['candidates'] + g['ans_gpu_groups'] + g['tree_gpu_groups'],
            'Hybrid execution or submission counts differ')
    require(g['feature_preparations'] + g['feature_reuses'] == r['candidates'] and
            g['feature_preparations'] == g['evictions'] + 1 and
            g['feature_seconds'] + g['token_seconds'] <= r['stages']['tokens'] + 1e-6,
            'Feature reuse or timing differs')
    require(g['tree_gpu_preparations'] > 0 and g['tree_gpu_groups'] > 0 and
            g['tree_gpu_samples'] > 0 and
            g['tree_cpu_fallback_properties'] == g['tree_cpu_small_properties'] and
            g['tree_cpu_unsupported_learning_calls'] == 0 and
            g['tree_prepare_seconds'] + g['tree_evaluate_seconds'] <=
            r['training']['stages']['split_scanning'] + 1e-6,
            'GPU learning route or timing differs')
    if not estimated:
        require(g['ans_gpu_groups'] >= r['candidates'] and g['ans_gpu_states'] > 0 and
                g['ans_cpu_fallback_groups'] == g['ans_cpu_small_groups'] == 0,
                'Exact full-resolution control omitted GPU ANS costing')


def capture(args):
    repo = Path(__file__).resolve().parents[2]
    root = args.capture.resolve()
    baseline = args.baseline.resolve()
    prior = json.loads((baseline / 'manifest.json').read_text())
    status = command(['git', 'status', '--porcelain'], repo)
    require(status['returncode'] == 0 and not status['stdout'], 'Capture requires a clean checkout')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'bin').mkdir()
    binary = root / 'bin/gjxl_modular_search_profile'
    shutil.copy2(args.binary, binary)
    env = dict(os.environ)
    removed = [k for k in env if k.startswith(('JXL_', 'GJXL_'))]
    for k in removed:
        del env[k]
    manifest = dict(schema=1, started_utc=utc(), revision=command(['git', 'rev-parse', 'HEAD'], repo)['stdout'].strip(),
                    tracked_status=status['stdout'], binary=dict(path=str(binary), sha256=sha(binary)),
                    baseline=str(baseline), baseline_manifest_sha256=sha(baseline / 'manifest.json'),
                    inputs=prior['inputs'], libjxl_source=prior['libjxl_source'],
                    removed_environment_keys=removed, schedule=schedule(), commands=[],
                    environment=[command(a, repo) for a in (['sw_vers'],
                        ['sysctl', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string'],
                        ['pmset', '-g', 'batt'], ['pmset', '-g', 'therm'])],
                    scope='Three CPU estimated correctness controls followed by two paired GPU exact/estimated repetitions per original image. One CPU participant; full retained search and checkpoints; no sampling profiler; backend setup and file I/O outside complete-call timing; no warmup excluded. Active desktop, not isolated throughput qualification.')
    def save():
        temporary = root / 'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(root / 'manifest.json')
    save()
    try:
        for item in prior['inputs'].values():
            require(sha(Path(item['path'])) == item['sha256'], 'Original input hash differs')
        for entry in prior['libjxl_source'].values():
            require(command(['git', '-C', entry['source'], 'rev-parse', 'HEAD'], repo)['stdout'].strip()
                    == entry['revision'] and
                    command(['git', '-C', entry['source'], 'diff', '--quiet', 'HEAD'], repo)['returncode'] == 0,
                    'Pinned reference source differs')
        for item in schedule():
            require(not (root / 'STOP_AFTER_CURRENT').exists(), 'Orderly stop requested')
            source = Path(prior['inputs'][item['image']]['path'])
            output = root / (item['tag'] + '.jxl')
            argv = [str(binary), str(source), str(output), '262144', '256', '1', *FLAGS[item['route']]]
            record = dict(item, argv=argv, started_utc=utc(), input_sha256=sha(source))
            manifest['commands'].append(record)
            save()
            print('START', item['tag'], flush=True)
            with (root / (item['tag'] + '.stdout')).open('w') as out, \
                 (root / (item['tag'] + '.stderr')).open('w') as err, \
                 (root / (item['tag'] + '-activity.jsonl')).open('w') as activity:
                begin = time.monotonic()
                process = subprocess.Popen(argv, cwd=repo, env=env, stdout=out, stderr=err)
                record['pid'] = process.pid
                save()
                while process.poll() is None:
                    snapshot = command(['ps', '-Ao', 'pid,pcpu,comm', '-r'], repo)
                    activity.write(json.dumps(dict(time=utc(), processes=snapshot['stdout'].splitlines()[:25])) + '\n')
                    activity.flush()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                record.update(returncode=process.returncode, process_seconds=time.monotonic() - begin,
                              finished_utc=utc())
            save()
            require(record['returncode'] == 0, f'Encode failed: {item["tag"]}')
            result = json.loads((root / (item['tag'] + '.stdout')).read_text())
            check_result(result, item['route'])
            require(output.stat().st_size == result['encoded_bytes'], 'Output length differs')
            record.update(result=result, output_sha256=sha(output))
            manifest.setdefault('between_calls', []).append(dict(time=utc(), tag=item['tag'],
                thermal=command(['pmset', '-g', 'therm'], repo), power=command(['pmset', '-g', 'batt'], repo)))
            save()
            print('DONE', item['tag'], round(result['seconds'], 3), result['encoded_bytes'], flush=True)
        require(sha(binary) == manifest['binary']['sha256'], 'Frozen binary changed')
        manifest['finished_utc'] = utc()
        save()
        inventory = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
                     for p in sorted(root.rglob('*')) if p.is_file()}
        (root / 'sha256-manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
        analyze(root)
    except BaseException as error:
        manifest['interrupted_utc'] = utc()
        manifest['error'] = str(error)
        save()
        raise


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest.get('finished_utc') and not manifest.get('interrupted_utc') and
            not manifest['tracked_status'], 'Incomplete or dirty capture')
    inventory = json.loads((root / 'sha256-manifest.json').read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and p.name != 'sha256-manifest.json'}
    require(actual == set(inventory), 'Artifact inventory differs')
    for name, expected in inventory.items():
        p = root / name
        require(p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'],
                f'Artifact changed: {name}')
    require(sha(Path(manifest['binary']['path'])) == manifest['binary']['sha256'], 'Binary differs')
    baseline = Path(manifest['baseline'])
    require(sha(baseline / 'manifest.json') == manifest['baseline_manifest_sha256'], 'Baseline changed')
    prior = json.loads((baseline / 'manifest.json').read_text())
    require(manifest['inputs'] == prior['inputs'] and manifest['libjxl_source'] == prior['libjxl_source'],
            'Input or reference provenance differs')
    expected = schedule()
    records = manifest['commands']
    require(manifest['schedule'] == expected and len(records) == len(expected), 'Capture matrix incomplete')
    for row, item in zip(records, expected):
        require(all(row[k] == v for k, v in item.items()), 'Capture order differs')
        source = Path(manifest['inputs'][row['image']]['path'])
        require(row['returncode'] == 0 and row['input_sha256'] == sha(source) ==
                manifest['inputs'][row['image']]['sha256'], 'Input or process differs')
        require(row['argv'] == [manifest['binary']['path'], str(source), str(root / (row['tag'] + '.jxl')),
                               '262144', '256', '1', *FLAGS[row['route']]], 'Command differs')
        r = row['result']
        require(r == json.loads((root / (row['tag'] + '.stdout')).read_text()), 'Trace differs')
        check_result(r, row['route'])
        output = root / (row['tag'] + '.jxl')
        require(sha(output) == row['output_sha256'] and output.stat().st_size == r['encoded_bytes'],
                'Retained output differs')
    images = []
    for image in IMAGES:
        rows = [r for r in records if r['image'] == image]
        control = next(r for r in prior['commands'] if r['tag'] == image + '-gjxl-adaptive')
        require(sha(baseline / (control['tag'] + '.jxl')) == control['output_sha256'], 'Old control changed')
        for r in rows:
            require(r['result']['candidates'] == control['result']['candidates'] and
                    all(r['result']['training'][k] == control['result']['training'][k] for k in WORK),
                    'Candidate count or semantic learning work changed')
        selected = {}
        for route in ('exact', 'estimated'):
            group = [r for r in rows if r['route'] == route]
            require(len(group) == 2 and len({r['output_sha256'] for r in group}) == 1,
                    'Repeated output changed')
            reference = control if route == 'exact' else next(r for r in rows if r['route'] == 'cpu-estimated')
            for r in group:
                require(r['output_sha256'] == reference['output_sha256'], 'CPU or frozen exact file differs')
                same_policy(r['result']['policy'], reference['result']['policy'])
            selected[route] = dict(bytes=group[0]['result']['encoded_bytes'],
                seconds=distribution([r['result']['seconds'] for r in group]),
                stages={k: distribution([r['result']['stages'][k] for r in group])
                        for k in group[0]['result']['stages']},
                peak_backing=max(r['result']['peak_backing'] for r in group),
                planned_capacity=max(r['result']['planned_capacity'] for r in group))
        e, x = selected['estimated'], selected['exact']
        images.append(dict(image=image, routes=selected, extra_bytes=e['bytes']-x['bytes'],
                           ratio=x['seconds']['median']/e['seconds']['median'],
                           same_policy=next(r for r in rows if r['route'] == 'estimated')['result']['policy'] ==
                                       next(r for r in rows if r['route'] == 'exact')['result']['policy']))
    totals = {route: dict(seconds=sum(r['routes'][route]['seconds']['median'] for r in images),
                         bytes=sum(r['routes'][route]['bytes'] for r in images))
              for route in ('exact', 'estimated')}
    result = dict(schema=1, revision=manifest['revision'], binary=manifest['binary'],
                  artifact_root=str(root), artifacts_verified=len(inventory),
                  complete_calls=len(records), images=images, totals=totals,
                  ratio=totals['exact']['seconds']/totals['estimated']['seconds'],
                  extra_bytes=totals['estimated']['bytes']-totals['exact']['bytes'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('capture', 'analyze'))
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--binary', type=Path)
    p.add_argument('--json', type=Path)
    args = p.parse_args()
    if args.mode == 'capture':
        require(args.baseline and args.binary, 'Capture requires baseline and binary')
        capture(args)
    else:
        result = analyze(args.capture.resolve())
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
