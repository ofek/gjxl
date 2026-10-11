#!/usr/bin/env python3
"""Frozen identical-token rate study; does not change encoder policy."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
CELLS = ['native-prefix-estimated-32', 'native-direct-balanced-32', 'native-direct-density-32',
         'native-direct-balanced-64', 'native-direct-density-64',
         *[cell for c in (16, 32, 64, 128) for cell in
           ([f'libjxl-e7-streaming-{c}', f'native-on-libjxl-partition-{c}'] if c <= 64 else
            [f'libjxl-e7-streaming-{c}'])],
         'libjxl-e9-streaming-128', 'libjxl-e9-nonstreaming-128']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def policy_text(p):
    rows = [f"POLICY1 {p['rct']} {len(p['transforms'])} {len(p['nodes'])}",
            ' '.join(map(str, p['weighted_coefficients'])), ' '.join(map(str, p['weighted_weights']))]
    rows += [' '.join(str(int(v)) for v in row) for row in p['transforms']]
    rows += [' '.join(map(str, row)) for row in p['nodes']]
    return '\n'.join(rows) + '\n'


def capture(args):
    source = args.pipeline.resolve()
    prior = json.loads((source / 'manifest.json').read_text())
    require(prior['complete'] and len(prior['commands']) == 128, 'Pipeline controls are incomplete')
    controls = {r['image']: r for r in prior['commands'] if r['variant'] == 'emit' and r['repetition'] == 1}
    require(set(controls) == set(prior['inputs']) and len(controls) == 19, 'Rate corpus differs')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = root / 'entropy_replay'
    shutil.copy2(args.binary, binary)
    manifest = dict(schema=1, complete=False, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        pipeline=dict(path=str(source / 'manifest.json'), sha256=sha(source / 'manifest.json')),
        binary=dict(path=str(binary), sha256=sha(binary)), libjxl_source=prior['libjxl_source'],
        inputs=prior['inputs'], cells=CELLS, images=[],
        scope='Identical frozen ordered tokens, contexts, streams, tree, predictor, transforms and image framing. Forced ANS and disabled LZ77 for every model. Pinned libjxl builder cells are not its full default encoder. One invocation per cell; timings are diagnostic API-boundary measurements, not latency qualification. Native-on-reference-partition excludes the cost of discovering the supplied partition.')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('JXL_', 'GJXL_'))}
    manifest['removed_environment_keys'] = sorted(set(os.environ) - set(env))
    reference = Path('/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl')
    reference_files = [reference / 'CMakeCache.txt', reference / 'lib/libjxl-internal.a',
                       *sorted((reference / 'lib').glob('*.dylib')),
                       *sorted((reference / 'third_party/brotli').glob('*.dylib'))]
    manifest['reference_artifacts'] = {str(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in reference_files}
    for entry in manifest['libjxl_source'].values():
        require(subprocess.check_output(['git', '-C', entry['source'], 'rev-parse', 'HEAD'], text=True).strip() == entry['revision'],
                'Pinned reference revision differs')
    guard = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        for name, control in controls.items():
            directory = root / name
            directory.mkdir()
            image = prior['inputs'][name]
            require(sha(image['path']) == image['sha256'], 'Input hash changed')
            policy = directory / 'policy.txt'
            policy.write_text(policy_text(control['result']['policy']))
            archive = directory / 'tokens.bin'
            record = dict(image=name, input_sha256=image['sha256'], commands=[], artifacts={})
            manifest['images'].append(record)
            write(root / 'manifest.json', manifest)
            print('START', name, flush=True)
            for mode in ('freeze', 'replay'):
                argv = [str(binary), mode, image['path'], str(policy), str(archive)]
                if mode == 'replay':
                    argv += [str(directory / 'outputs')]
                start = time.monotonic()
                with (directory / (mode + '.stdout')).open('w') as out, (directory / (mode + '.stderr')).open('w') as err:
                    p = subprocess.run(argv, stdout=out, stderr=err, env=env, check=False)
                record['commands'].append(dict(argv=argv, returncode=p.returncode, wall_seconds=time.monotonic() - start))
                require(p.returncode == 0, f'Replay failed for {name}: {mode}')
                if mode == 'freeze':
                    record['archive_sha256_before_replay'] = sha(archive)
            record['results'] = [json.loads(line) for line in (directory / 'replay.stdout').read_text().splitlines()]
            require([r['cell'] for r in record['results']] == CELLS and all(r['decoded_exact'] for r in record['results']),
                    'Replay cell matrix or decoding differs')
            require(sha(archive) == record['archive_sha256_before_replay'], 'Replay changed the frozen tokens')
            baseline = directory / 'outputs/native-prefix-estimated-32.jxl'
            expected = control['files'][control['tag'] + '.jxl']['sha256']
            require(sha(baseline) == expected == sha(source / (control['tag'] + '.jxl')),
                    'Frozen replay does not reproduce the complete-search winner')
            record['winning_file_sha256'] = expected
            record['artifacts'] = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
                                   for p in directory.rglob('*') if p.is_file()}
            write(root / 'manifest.json', manifest)
            print('DONE', name, len(record['results']), 'independently decoded models', flush=True)
        manifest['complete'] = True
    finally:
        guard.terminate(); guard.wait(timeout=5)
        manifest['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(root / 'manifest.json', manifest)


def audit(root):
    root = root.resolve()
    m = json.loads((root / 'manifest.json').read_text())
    require(m['complete'] and m['cells'] == CELLS and len(m['images']) == len(m['inputs']) == 19,
            'Incomplete rate-study matrix')
    require(sha(root / 'entropy_replay') == m['binary']['sha256'], 'Replay binary differs')
    require(sha(m['pipeline']['path']) == m['pipeline']['sha256'], 'Pipeline provenance changed')
    for path, info in m['reference_artifacts'].items():
        require(Path(path).stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Reference artifact changed')
    require({r['image'] for r in m['images']} == set(m['inputs']), 'Duplicate/missing replay image')
    for row in m['images']:
        directory = root / row['image']
        require(row['input_sha256'] == m['inputs'][row['image']]['sha256'] == sha(m['inputs'][row['image']]['path']),
                'Replay input differs')
        require(len(row['commands']) == 2 and all(r['returncode'] == 0 for r in row['commands']), 'Replay command failed')
        original = Path(m['binary']['path']).parent / row['image']
        for command, mode in zip(row['commands'], ('freeze', 'replay'), strict=True):
            expected_argv = [m['binary']['path'], mode, m['inputs'][row['image']]['path'],
                             str(original / 'policy.txt'), str(original / 'tokens.bin')]
            if mode == 'replay': expected_argv += [str(original / 'outputs')]
            require(command['argv'] == expected_argv, 'Replay command arguments differ')
        expected = {str(p.relative_to(root)) for p in directory.rglob('*') if p.is_file()}
        require(set(row['artifacts']) == expected and len(expected) == 20, 'Replay artifact inventory incomplete')
        for name, info in row['artifacts'].items():
            p = root / name
            require(p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], 'Replay artifact differs')
        results = [json.loads(line) for line in (directory / 'replay.stdout').read_text().splitlines()]
        require(results == row['results'] and [r['cell'] for r in results] == CELLS, 'Replay result differs')
        require(sha(directory / 'tokens.bin') == row['archive_sha256_before_replay'], 'Frozen token archive differs')
        baseline = results[0]
        for r in results:
            require(r['decoded_exact'] and r['tokens'] == baseline['tokens'] and r['contexts'] == baseline['contexts'] and
                    r['tree_bits'] == baseline['tree_bits'] and len(r['context_map']) == r['contexts'], 'Fixed replay conditions differ')
            require(r['model_bits'] == r['context_map_bits'] + r['histogram_config_bits'] + r['lz77_flag_bits'] and
                    r['bytes'] == (directory / 'outputs' / (r['cell'] + '.jxl')).stat().st_size,
                    'Replay exact cost decomposition differs')
            if r['cell'].startswith('native-on-libjxl-partition-'):
                capacity = r['cell'].split('-')[-1]
                control = next(c for c in results if c['cell'] == f'libjxl-e7-streaming-{capacity}')
                require(r['context_map'] == control['context_map'], 'Partition-only control changed assignment')
        require(sha(directory / 'outputs/native-prefix-estimated-32.jxl') == row['winning_file_sha256'],
                'Replay baseline no longer matches actual complete search')
    return m


def analyze(args):
    m = audit(args.root)
    baseline = sum(row['results'][0]['bytes'] for row in m['images'])
    aggregate = {}
    for cell in CELLS:
        rows = [next(r for r in row['results'] if r['cell'] == cell) for row in m['images']]
        aggregate[cell] = {key: sum(r[key] for r in rows) for key in
                           ('bytes', 'payload_bits', 'model_bits', 'context_map_bits', 'histogram_config_bits')}
        aggregate[cell]['byte_change_percent'] = (aggregate[cell]['bytes'] / baseline - 1) * 100
        aggregate[cell]['observed_clusters'] = [r['clusters'] for r in rows]
    result = dict(schema=1, manifest=str(args.root.resolve() / 'manifest.json'), manifest_sha256=sha(args.root / 'manifest.json'),
        images=19, frozen_token_models=19 * len(CELLS), aggregate=aggregate,
        per_image={row['image']: row['results'] for row in m['images']}, scope=m['scope'])
    write(args.json, result)
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture'); c.add_argument('--pipeline', type=Path, required=True)
    c.add_argument('--root', type=Path, required=True); c.add_argument('--binary', type=Path, required=True)
    a = sub.add_parser('analyze'); a.add_argument('--root', type=Path, required=True); a.add_argument('--json', type=Path, required=True)
    args = p.parse_args()
    (capture if args.command == 'capture' else analyze)(args)
