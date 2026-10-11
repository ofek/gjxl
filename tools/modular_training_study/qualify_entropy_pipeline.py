#!/usr/bin/env python3
"""Serial frozen-binary complete-search study and independent artifact audit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
LARGE = ('1029', '1039', '1049')
CURRENT = ('emit', 'measure', 'shortlist', 'population', 'combined')
VARIANTS = ('original', 'pre_reuse', 'reuse', *CURRENT)
FLAGS = {
    'original': ['--metal-learn'],
    'pre_reuse': ['--metal-learn', '--estimated-ans'],
    'reuse': ['--metal-learn', '--estimated-ans'],
    **{v: ['--metal-learn', '--estimated-ans', '--candidate-ledger'] for v in CURRENT}}
FLAGS['measure'] += ['--measure-all']
FLAGS['shortlist'] += ['--shortlist']
FLAGS['population'] += ['--measure-all', '--metal-populations']
FLAGS['combined'] += ['--shortlist', '--metal-populations']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(argv):
    p = subprocess.run([str(x) for x in argv], cwd=REPO, capture_output=True, text=True, check=False)
    return dict(argv=[str(x) for x in argv], returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def matrix(inputs):
    schedule = []
    mixed = [n for n in inputs if n not in LARGE]
    for i, name in enumerate(mixed):
        variants = CURRENT[i % len(CURRENT):] + CURRENT[:i % len(CURRENT)]
        schedule.extend(dict(image=name, repetition=1, variant=v) for v in variants)
    for repetition in (1, 2):
        for i, name in enumerate(LARGE[repetition - 1:] + LARGE[:repetition - 1]):
            offset = (i + 3 * (repetition - 1)) % len(VARIANTS)
            variants = VARIANTS[offset:] + VARIANTS[:offset]
            if repetition == 2:
                variants = tuple(reversed(variants))
            schedule.extend(dict(image=name, repetition=repetition, variant=v) for v in variants)
    for row in schedule:
        row['tag'] = f"{row['image']}-r{row['repetition']}-{row['variant']}"
    return schedule


def basic(result, variant):
    require(result['decoded_exact'] and result['profile_enabled'] and result['entropy'] == 'ans',
            'Missing independent decode/profile/ANS')
    require((result['maximum_samples'], result['maximum_leaves'], result['checkpoints'], result['threads']) ==
            (262144, 256, True, 1), 'Training/thread policy changed')
    require(result['ans_selection'] == ('exact' if variant == 'original' else 'estimated'), 'Selector differs')
    require(0 < result['seconds'] and result['peak_backing'] <= result['planned_capacity'], 'Invalid time/storage')
    require(sum(result['stages'].values()) <= result['seconds'] + 1e-5 and
            sum(result['training']['stages'].values()) <= result['stages']['training'] + 1e-5,
            'Overlapping wall or training stages')
    g, e = result['adaptive_gpu'], result['entropy_work']
    require(result['route'] == 'hybrid-learning-complete-search', 'Hybrid route missing')
    if variant not in CURRENT:
        require(g['encoded_candidates'] == result['candidates'] and g['cpu_fallback_candidates'] == 0,
                'Frozen large control lost GPU token route')
        return
    short = variant in ('shortlist', 'combined')
    population = variant in ('population', 'combined')
    measure = variant in ('measure', 'population')
    require(result['adaptive_populations'] == population and result['candidate_ledger_enabled'], 'Private flags differ')
    require(result['search_evaluation'] == ('shortlist' if short else 'measure-all' if measure else 'encode-all'),
            'Candidate evaluation differs')
    n = result['candidates']
    require(result['candidate_emissions'] == (n if variant == 'emit' else 1) and
            result['screened_candidates'] == (n if short else 0) and
            result['exactly_measured_candidates'] == (min(4, n) if short else n if measure else 0),
            'Bounded search work differs')
    actual = n + (min(4, n) + 1 if short else 1 if measure else 0)
    require(g['encoded_candidates'] + g['cpu_fallback_candidates'] == actual, 'Final/shortlist token work missing')
    require(e['ans_value_collection_ns'] == e['ans_value_aggregation_ns'] == e['prefix_section_measurements'] == 0,
            'Duplicate population work or prefix ordered pass returned')
    if population:
        require(e['prefix_population_models'] + g['population_cpu_wide'] == g['population_gpu_calls'] and
                g['population_gpu_calls'] + g['population_cpu_small'] + g['population_cpu_unsupported'] +
                g['population_screen_skips'] == actual, 'Population routes do not cover all requests')
    else:
        require(g['population_gpu_calls'] == e['prefix_population_models'] == 0, 'Disabled population route ran')
    require(result['gpu_submissions'] == g['encoded_candidates'] + g['ans_gpu_groups'] +
            g['tree_gpu_groups'] + g['population_gpu_calls'], 'GPU submissions do not reconcile')
    ledger = result['candidate_ledger']
    require(len(ledger) == n + (min(4, n) if short else 0), 'Ledger is incomplete')
    require([r['index'] for r in ledger[:n]] == list(range(n)), 'Candidate order/index differs')
    incumbent = None
    for r in ledger:
        if r['exact']:
            key = (r['bytes'], r['checkpoint'], r['index'])
            improved = incumbent is None or key < incumbent
            require(r['improved'] == improved, 'Ledger incumbent improvement differs')
            if improved: incumbent = key
            require(sum(r[k] for k in ('payload_bits', 'tree_bits', 'model_bits', 'stream_header_bits',
                    'frame_header_bits', 'toc_bits', 'padding_bits')) == 8 * r['bytes'], 'Cost categories do not cover file')
        else:
            require(short and not r['improved'] and r['bytes'] == 0 and r['estimated_bits'] > 0 and r['sampled_tokens'] > 0,
                    'Screen estimate was presented as exact')
    exact = [r for r in ledger if r['exact']]
    winner = min(exact, key=lambda r: (r['bytes'], r['checkpoint'], r['index']))
    require(result['encoded_bytes'] == winner['bytes'] and result['policy'] == winner['policy'], 'Wrong exact winner')


def capture(args):
    root = args.root.resolve()
    require(not run(['git', 'diff', '--name-only', 'HEAD'])['stdout'], 'Capture requires clean tracked sources')
    prior_root = (REPO / 'build/entropy-estimated-qualified-20261011')
    prior = json.loads((prior_root / 'manifest.json').read_text())
    mixed_path = Path('/Users/yunhocho/GitHub/gjxl-modular-training-budgets/build/training-checkpoints/search/manifest.json')
    mixed = json.loads(mixed_path.read_text())['inputs']
    inputs = {r['name']: r for r in mixed}
    inputs.update(prior['inputs'])
    require(len(inputs) == 19, 'Retained corpus differs')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'bin').mkdir()
    binaries = {}
    for variant in VARIANTS:
        if variant in ('original', 'pre_reuse'):
            source = Path(prior['binary']['path'])
            revision = prior['revision']
            require(sha(source) == prior['binary']['sha256'], 'Original frozen binary differs')
        elif variant == 'reuse':
            source = REPO / 'build/entropy-milestones/step2/gjxl_modular_search_profile'
            milestone = json.loads(source.with_name('manifest.json').read_text())
            revision = milestone['revision']
            require(sha(source) == milestone['files'][source.name]['sha256'], 'Step2 binary differs')
        else:
            source = REPO / 'build/entropy-milestones/step4/gjxl_modular_search_profile'
            milestone = json.loads(source.with_name('manifest.json').read_text())
            revision = milestone['revision']
            require(sha(source) == milestone['files'][source.name]['sha256'], 'Step4 binary differs')
        destination = root / 'bin' / variant
        shutil.copy2(source, destination)
        binaries[variant] = dict(path=str(destination), sha256=sha(destination), revision=revision)
    env = dict(os.environ)
    removed = [k for k in env if k.startswith(('JXL_', 'GJXL_'))]
    for k in removed:
        del env[k]
    manifest = dict(schema=1, started_utc=utc(), complete=False, inputs=inputs, binaries=binaries,
        revision=run(['git', 'rev-parse', 'HEAD'])['stdout'].strip(), schedule=matrix(inputs), commands=[],
        mixed_manifest=dict(path=str(mixed_path), sha256=sha(mixed_path)),
        prior_manifest=dict(path=str(prior_root / 'manifest.json'), sha256=sha(prior_root / 'manifest.json')),
        libjxl_source=prior['libjxl_source'],
        environment=[run(argv) for argv in (['sw_vers'], ['sysctl', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string'],
                                           ['pmset', '-g', 'batt'], ['pmset', '-g', 'therm'])],
        removed_environment_keys=removed,
        scope='Serial complete encoder calls; 16 mixed images once per current mode and 3 original 12MP images twice per mode. Full 262144/256 learner, checkpoints, one CPU participant. No excluded warmup. Backend setup, independent decode, file I/O outside encoder timer. Current modes include candidate ledger serialization; old frozen controls predate that option. Active desktop; no fresh libjxl encoder latency claim.')
    guard = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    manifest['idle_sleep_guard_pid'] = guard.pid
    write(root / 'manifest.json', manifest)
    try:
        for entry in prior['libjxl_source'].values():
            require(run(['git', '-C', entry['source'], 'rev-parse', 'HEAD'])['stdout'].strip() == entry['revision'] and
                    run(['git', '-C', entry['source'], 'diff', '--quiet', 'HEAD'])['returncode'] == 0,
                    'Pinned reference source differs')
        for item in inputs.values():
            require(sha(item['path']) == item['sha256'], 'Input hash differs')
        for item in manifest['schedule']:
            require(not (root / 'STOP_AFTER_CURRENT').exists(), 'Orderly stop requested')
            binary = binaries[item['variant']]
            require(sha(binary['path']) == binary['sha256'], 'Frozen binary changed')
            source = inputs[item['image']]
            output = root / (item['tag'] + '.jxl')
            argv = [binary['path'], source['path'], str(output), '262144', '256', '1', *FLAGS[item['variant']]]
            record = dict(item, argv=argv, started_utc=utc(), input_sha256=sha(source['path']))
            manifest['commands'].append(record)
            write(root / 'manifest.json', manifest)
            print('START', item['tag'], flush=True)
            begin = time.monotonic()
            with (root / (item['tag'] + '.stdout')).open('w') as out, \
                 (root / (item['tag'] + '.stderr')).open('w') as err, \
                 (root / (item['tag'] + '-activity.jsonl')).open('w') as activity:
                process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=out, stderr=err)
                record['pid'] = process.pid
                while process.poll() is None:
                    activity.write(json.dumps(dict(utc=utc(), monotonic=time.monotonic(),
                        processes=run(['ps', '-Ao', 'pid,pcpu,comm', '-r'])['stdout'].splitlines()[:20])) + '\n')
                    activity.flush()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            record.update(returncode=process.returncode, ended_utc=utc(), process_wall_seconds=time.monotonic() - begin)
            require(process.returncode == 0, f"Encoder failed: {item['tag']}")
            result = json.loads((root / (item['tag'] + '.stdout')).read_text())
            basic(result, item['variant'])
            # Keep full ledgers only in hashed stdout; avoid growing manifest I/O.
            record['result'] = {k: v for k, v in result.items() if k != 'candidate_ledger'}
            record['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in
                               (output, root / (item['tag'] + '.stdout'), root / (item['tag'] + '.stderr'),
                                root / (item['tag'] + '-activity.jsonl'))}
            write(root / 'manifest.json', manifest)
            print('DONE', item['tag'], f"{result['seconds']:.3f}s", result['encoded_bytes'], flush=True)
        manifest['complete'] = True
    finally:
        guard.terminate()
        guard.wait(timeout=5)
        manifest['ended_utc'] = utc()
        manifest['environment_after'] = [run(['pmset', '-g', 'batt']), run(['pmset', '-g', 'therm'])]
        with (root / 'sleep-events.log').open('w') as sleep:
            subprocess.run(['pmset', '-g', 'log'], stdout=sleep, stderr=subprocess.STDOUT, check=False)
        manifest['sleep_events_sha256'] = sha(root / 'sleep-events.log')
        write(root / 'manifest.json', manifest)


def audit(root):
    root = root.resolve()
    m = json.loads((root / 'manifest.json').read_text())
    require(m['complete'] and m['schedule'] == matrix(m['inputs']), 'Incomplete or changed capture matrix')
    require(len(m['commands']) == len(m['schedule']) == 128, 'Missing capture calls')
    require(sha(root / 'sleep-events.log') == m['sleep_events_sha256'], 'Sleep evidence differs')
    for b in m['binaries'].values():
        require(sha(root / 'bin' / Path(b['path']).name) == b['sha256'], 'Frozen binary hash differs')
    for entry in m['inputs'].values():
        require(sha(entry['path']) == entry['sha256'], 'Input hash differs')
    lookup = {}
    for row, expected in zip(m['commands'], m['schedule'], strict=True):
        require(all(row[k] == v for k, v in expected.items()) and row['returncode'] == 0, 'Wrong command schedule/status')
        require(row['input_sha256'] == m['inputs'][row['image']]['sha256'], 'Command input hash differs')
        binary = m['binaries'][row['variant']]
        require(row['argv'] == [binary['path'], m['inputs'][row['image']]['path'],
                str(Path(binary['path']).parents[1] / (row['tag'] + '.jxl')), '262144', '256', '1', *FLAGS[row['variant']]],
                'Capture arguments differ')
        required = {row['tag'] + suffix for suffix in ('.jxl', '.stdout', '.stderr', '-activity.jsonl')}
        require(set(row['files']) == required, 'Artifact inventory differs')
        for name, info in row['files'].items():
            p = root / name
            require(p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], 'Artifact size/hash differs')
        result = json.loads((root / (row['tag'] + '.stdout')).read_text())
        require({k: v for k, v in result.items() if k != 'candidate_ledger'} == row['result'] and result['encoded_bytes'] == row['files'][row['tag'] + '.jxl']['bytes'],
                'Result or complete bytes differ from artifact')
        basic(result, row['variant'])
        require(all(result[k] == m['inputs'][row['image']][k] for k in ('width', 'height', 'channels', 'bits')),
                'Profile dimensions differ from frozen input')
        row['result'] = result  # Audited full ledger, in memory only.
        lookup[row['image'], row['repetition'], row['variant']] = row
        activity = [json.loads(line) for line in (root / (row['tag'] + '-activity.jsonl')).read_text().splitlines()]
        gaps = [max(b['monotonic'] - a['monotonic'],
                    (datetime.datetime.fromisoformat(b['utc']) - datetime.datetime.fromisoformat(a['utc'])).total_seconds())
                for a, b in zip(activity, activity[1:])]
        require(not gaps or max(gaps) < 20, 'Capture heartbeat interruption needs explicit whole-pair recheck')
    keys = ('bytes', 'tokens', 'contexts', 'clusters', 'payload_bits', 'tree_bits', 'model_bits',
            'stream_header_bits', 'frame_header_bits', 'toc_bits', 'padding_bits', 'policy', 'checkpoint', 'index')
    cases = []
    for image in m['inputs']:
        repetitions = (1, 2) if image in LARGE else (1,)
        for repetition in repetitions:
            rows = {v: lookup[image, repetition, v] for v in CURRENT}
            results = {v: row['result'] for v, row in rows.items()}
            base = results['emit']
            for v, r in results.items():
                require(r['candidates'] == base['candidates'] and all(r['training'][k] == base['training'][k]
                        for k in ('learning_calls', 'population', 'samples', 'split_evaluations', 'sorted_properties', 'sorted_samples')),
                        'Comparison changed candidate/training work')
                if v in ('measure', 'population'):
                    require(r['policy'] == base['policy'] and rows[v]['files'][rows[v]['tag'] + '.jxl']['sha256'] ==
                            rows['emit']['files'][rows['emit']['tag'] + '.jxl']['sha256'], 'Exact route changed winning file')
                for entry in r['candidate_ledger']:
                    control = base['candidate_ledger'][entry['index']]
                    require(entry['policy'] == control['policy'] and entry['checkpoint'] == control['checkpoint'],
                            'Candidate policy/identity changed')
                    if entry['exact']:
                        require(all(entry[k] == control[k] for k in keys), 'Candidate measured size differs from emission')
            require(results['shortlist']['policy'] == results['combined']['policy'] and
                    rows['shortlist']['files'][rows['shortlist']['tag'] + '.jxl']['sha256'] ==
                    rows['combined']['files'][rows['combined']['tag'] + '.jxl']['sha256'], 'GPU populations changed shortlist file')
            finalist = [r['index'] for r in results['shortlist']['candidate_ledger'] if r['exact']]
            best = min(base['candidate_ledger'], key=lambda r: (r['bytes'], r['checkpoint'], r['index']))
            cases.append(dict(image=image, repetition=repetition, base_bytes=base['encoded_bytes'],
                shortlist_bytes=results['shortlist']['encoded_bytes'], winner_recalled=best['index'] in finalist,
                regret_bytes=results['shortlist']['encoded_bytes'] - base['encoded_bytes'],
                regret_percent=(results['shortlist']['encoded_bytes'] / base['encoded_bytes'] - 1) * 100,
                finalists=finalist, exhaustive_winner=best['index']))
            if image in LARGE:
                for v in ('original', 'pre_reuse', 'reuse'):
                    old = lookup[image, repetition, v]
                    require(old['result']['policy'] == base['policy'] and old['files'][old['tag'] + '.jxl']['sha256'] ==
                            rows['emit']['files'][rows['emit']['tag'] + '.jxl']['sha256'], 'Frozen control output changed')
    return m, lookup, cases


def power_transitions(root, manifest):
    events = []
    previous = None
    begin = datetime.datetime.fromisoformat(manifest['started_utc'])
    end = datetime.datetime.fromisoformat(manifest['ended_utc'])
    for line in (root / 'sleep-events.log').read_text().splitlines():
        source = re.search(r'Using (AC|Batt)\b', line)
        if not source:
            continue
        try:
            stamp = datetime.datetime.strptime(line[:25], '%Y-%m-%d %H:%M:%S %z')
        except ValueError:
            continue
        current = source.group(1)
        if previous is not None and current != previous and begin <= stamp <= end:
            events.append(dict(utc=stamp.astimezone(datetime.timezone.utc).isoformat(),
                               source=current, previous=previous, evidence=line.strip()))
        previous = current
    affected = set()
    for image in LARGE:
        for repetition in (1, 2):
            rows = [r for r in manifest['commands'] if r['image'] == image and r['repetition'] == repetition]
            start = min(datetime.datetime.fromisoformat(r['started_utc']) for r in rows)
            stop = max(datetime.datetime.fromisoformat(r['ended_utc']) for r in rows)
            if any(start <= datetime.datetime.fromisoformat(e['utc']) <= stop for e in events):
                affected.add((image, repetition))
    return events, affected


def analyze(args):
    m, lookup, cases = audit(args.root)
    events, affected = power_transitions(args.root, m)
    raw_sums = {v: sum(statistics.median(lookup[i, n, v]['result']['seconds'] for n in (1, 2))
                       for i in LARGE) for v in VARIANTS}
    rechecks = []
    covered = set()
    for root in args.recheck:
        from recheck_entropy_pipeline import audit_recheck
        recheck = audit_recheck(root, lookup)
        group = (recheck['image'], recheck['repetition'])
        require(group in affected and group not in covered, 'Recheck is not a unique affected comparison')
        require(recheck['source']['sha256'] == sha(args.root / 'manifest.json'), 'Recheck belongs to another capture')
        covered.add(group)
        for row in recheck['commands']:
            lookup[row['image'], row['repetition'], row['variant']] = row
        rechecks.append(dict(manifest=str(root.resolve() / 'manifest.json'), manifest_sha256=sha(root / 'manifest.json'),
            image=group[0], repetition=group[1], reason=recheck['reason'], calls=len(recheck['commands'])))
    table = {}
    for image in LARGE:
        table[image] = {}
        for v in VARIANTS:
            rows = [lookup[image, n, v]['result'] for n in (1, 2)]
            table[image][v] = dict(median_seconds=statistics.median(r['seconds'] for r in rows),
                seconds=[r['seconds'] for r in rows], bytes=rows[0]['encoded_bytes'],
                stages={k: statistics.median(r['stages'][k] for r in rows) for k in rows[0]['stages']},
                entropy_work={k: statistics.median(r['entropy_work'][k] for r in rows) for k in rows[0]['entropy_work']},
                adaptive_gpu={k: statistics.median(r['adaptive_gpu'][k] for r in rows) for k in rows[0]['adaptive_gpu']},
                managed_peak_bytes=[r['peak_backing'] for r in rows],
                planned_capacity=rows[0]['planned_capacity'], candidates=rows[0]['candidates'])
    unique = [r for r in cases if r['repetition'] == 1]
    result = dict(schema=1, manifest=str(args.root.resolve() / 'manifest.json'),
        manifest_sha256=sha(args.root / 'manifest.json'), audited_calls=len(m['commands']) + sum(r['calls'] for r in rechecks),
        binaries=m['binaries'], environment=m['environment'], environment_after=m['environment_after'],
        started_utc=m['started_utc'], ended_utc=m['ended_utc'],
        raw_audited_calls=len(m['commands']), raw_sum_large_medians=raw_sums,
        power_transitions=events, rechecks=rechecks, qualified_time_comparisons=affected == covered,
        unresolved_power_comparisons=[dict(image=i, repetition=n) for i, n in sorted(affected - covered)],
        large=table, candidate_cases=cases,
        aggregate_regret_percent=(sum(r['shortlist_bytes'] for r in unique) / sum(r['base_bytes'] for r in unique) - 1) * 100,
        worst_regret=max(unique, key=lambda r: r['regret_percent']),
        winner_recall=sum(r['winner_recalled'] for r in unique), images=len(unique),
        sum_large_medians={v: sum(table[i][v]['median_seconds'] for i in LARGE) for v in VARIANTS},
        mean_large_stages={v: {k: statistics.mean(table[i][v]['stages'][k] for i in LARGE)
                              for k in table[LARGE[0]][v]['stages']} for v in VARIANTS},
        scope=m['scope'])
    write(args.json, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture'); c.add_argument('--root', type=Path, required=True)
    a = sub.add_parser('analyze'); a.add_argument('--root', type=Path, required=True); a.add_argument('--json', type=Path, required=True)
    a.add_argument('--recheck', type=Path, action='append', default=[])
    args = parser.parse_args()
    (capture if args.command == 'capture' else analyze)(args)
