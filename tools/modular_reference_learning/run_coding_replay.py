#!/usr/bin/env python3
"""Verify native encoding of the previously frozen actual e9 stream decisions."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from analyze import sha
from coding_replay import project, validate_result
from run_stream_replay import capture_cases
from study_provenance import freeze_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--policy-study', type=Path, required=True)
    parser.add_argument('--policy-validation', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.binary, args.policy_study, args.policy_validation, args.output = (
        p.resolve() for p in (args.binary, args.policy_study, args.policy_validation, args.output))
    learning, source_artifacts = capture_cases(args.policy_study, args.policy_validation, args.cases)
    cases = {}
    for call in learning:
        cases[call['input'], call['variant']] = Path(call['trace']).parent.parent
    policy = json.loads((args.policy_study / 'manifest.json').read_text())
    corpus = json.loads((Path(policy['corpus']) / 'manifest.json').read_text())
    inputs = {i['name']: i for i in corpus['inputs']}
    args.output.mkdir(parents=True, exist_ok=False)
    binary, build = freeze_build(args.binary, args.output)
    build['artifacts'].update(source_artifacts)
    manifest = {'schema': 1, **build, 'policy_study': str(args.policy_study),
                'policy_validation': str(args.policy_validation),
                'sanitizer_environment': {k: os.environ[k] for k in ('ASAN_OPTIONS', 'UBSAN_OPTIONS') if k in os.environ},
                'cases': [{'input': n, 'variant': v, 'capture': str(d), 'source': inputs[n]}
                          for (n, v), d in cases.items()],
                'scope': 'Borrowed captured transformed pixels, raw trees and WP/transform/scope decisions. '
                         'Independent pinned tree and sample tokens; Prefix and ANS exact decodes. '
                         'No transform search or fitting, no native policy parity claim, no timing qualification.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'results.jsonl').open('x', buffering=1) as ledger:
        for (name, variant), source in cases.items():
            item = inputs[name]
            if sha(item['path']) != item['sha256']:
                raise ValueError('Source PNM changed')
            directory = args.output / f'{name}-{variant}'
            directory.mkdir()
            before = {str(p): sha(p) for p in (source / 'streams').iterdir() if p.is_file()}
            before[item['path']] = item['sha256']
            before[str(source / 'output.jxl')] = sha(source / 'output.jxl')
            description = directory / 'input.txt'
            description.write_text(project(source))
            command = [str(binary), 'stream-encode-check', item['path'], str(description),
                       str(directory / 'encoded')]
            with (directory / 'stdout.json').open('x') as out, (directory / 'stderr.log').open('x') as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
            result = json.loads((directory / 'stdout.json').read_text())
            validate_result(source, result)
            for path, digest in before.items():
                if sha(path) != digest:
                    raise ValueError('Captured coding input changed during replay')
            for encode in result['encodes']:
                file = directory / 'encoded' / (encode['mode'] + '.jxl')
                if file.stat().st_size != encode['bytes']:
                    raise ValueError('Encoded file size mismatch')
            ledger.write(json.dumps({'input': name, 'variant': variant, 'command': command,
                                     'result': result, 'reference_bytes': (source / 'output.jxl').stat().st_size,
                                     'artifacts': {**before, **{str(p): sha(p) for p in directory.rglob('*')
                                                               if p.is_file()}}}) + '\n')
            print(name, variant, 'tokens', result['tokens'], 'bytes',
                  [(e['mode'], e['bytes']) for e in result['encodes']], flush=True)


if __name__ == '__main__':
    main()
