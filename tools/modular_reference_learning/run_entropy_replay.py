#!/usr/bin/env python3
"""Run paired downstream coding controls on previously qualified e9 tokens."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from analyze import sha
from analyze_coding_replay import analyze as analyze_coding
from coding_replay import project
from entropy_replay import DIRECT_VARIANTS, VARIANTS, validate_controls, validate_result
from study_provenance import freeze_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--coding-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    args = parser.parse_args()
    args.binary, args.coding_study, args.output = (p.resolve() for p in
                                                   (args.binary, args.coding_study, args.output))
    if len(set(args.variants)) != len(args.variants):
        parser.error('Repeated entropy variant')
    if any(v.startswith('reference_map32') for v in args.variants) and 'no_lz77_clusters32' not in args.variants:
        parser.error('Partition transplants require no_lz77_clusters32')
    if set(args.variants) & set(DIRECT_VARIANTS) and 'native' not in args.variants:
        parser.error('Direct ANS requires native')
    checked = analyze_coding(args.coding_study)
    source_manifest = json.loads((args.coding_study / 'manifest.json').read_text())
    cases = source_manifest['cases']
    if args.cases:
        if len(set(args.cases)) != len(args.cases) or set(args.cases) - {f"{c['input']}:{c['variant']}" for c in cases}:
            parser.error('Unknown or repeated entropy case')
        cases = [c for c in cases if f"{c['input']}:{c['variant']}" in args.cases]
    args.output.mkdir(parents=True, exist_ok=False)
    binary, build = freeze_build(args.binary, args.output)
    build['artifacts'].update(checked['artifacts'])
    manifest = {'schema': 1, **build, 'coding_study': str(args.coding_study), 'cases': cases,
                'variants': args.variants,
                'sanitizer_environment': {k: os.environ[k] for k in ('ASAN_OPTIONS', 'UBSAN_OPTIONS') if k in os.environ},
                'scope': 'Identical captured logical trees, sample tokens and native frame/stream syntax. '
                         'Reference entropy settings imported exactly; each named variant changes only its declared controls. '
                         'Every file independently decoded; native control must reproduce the frozen native ANS file. '
                         'Empty ignored local payloads omitted in every branch. No learning, native policy or timing qualification.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'results.jsonl').open('x', buffering=1) as ledger:
        for case in cases:
            name, policy = case['input'], case['variant']
            source = Path(case['capture'])
            directory = args.output / f'{name}-{policy}'
            directory.mkdir()
            bindings = {str(p): sha(p) for p in (source / 'streams').iterdir() if p.is_file()}
            bindings[case['source']['path']] = sha(case['source']['path'])
            if bindings[case['source']['path']] != case['source']['sha256']:
                raise ValueError('Changed entropy source image')
            for p in (source / 'output.jxl', args.coding_study / directory.name / 'encoded/ans.jxl'):
                bindings[str(p)] = sha(p)
            description = directory / 'input.txt'
            description.write_text(project(source, entropy=True))
            command = [str(binary), 'stream-entropy-check', case['source']['path'], str(description),
                       str(directory / 'encoded'), ','.join(args.variants)]
            with (directory / 'stdout.jsonl').open('x') as out, (directory / 'stderr.log').open('x') as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
            results = [json.loads(line) for line in (directory / 'stdout.jsonl').read_text().splitlines()]
            if [r['variant'] for r in results] != args.variants:
                raise ValueError('Incomplete entropy controls')
            for result in results:
                validate_result(source, result)
                encoded = directory / 'encoded' / (result['variant'] + '.jxl')
                if encoded.stat().st_size != result['bytes']:
                    raise ValueError('Incorrect entropy file size')
                if result['variant'] == 'native' and sha(encoded) != sha(
                        args.coding_study / directory.name / 'encoded/ans.jxl'):
                    raise ValueError('Native entropy control differs from frozen native file')
            validate_controls(results)
            for path, digest in bindings.items():
                if sha(path) != digest:
                    raise ValueError('Entropy input changed during replay')
            ledger.write(json.dumps({'input': name, 'policy': policy, 'command': command,
                                     'results': results,
                                     'artifacts': {**bindings, **{str(p): sha(p) for p in directory.rglob('*') if p.is_file()}}}) + '\n')
            print(name, policy, {r['variant']: r['bytes'] for r in results}, flush=True)


if __name__ == '__main__':
    main()
