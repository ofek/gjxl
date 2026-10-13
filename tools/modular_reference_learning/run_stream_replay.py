#!/usr/bin/env python3
"""Replay actual e9 learning calls against native preparation and fitting."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from analyze import sha
from analyze_policy_study import analyze as analyze_policy, numbered, validate_builds, validate_trace
from stream_replay import project, validate_result
from study_provenance import freeze_build


def capture_cases(policy, validation, selected=None):
    policy_result = analyze_policy(policy)
    validation_result = validate_builds(validation, policy)
    paths = [(row['input'], variant, policy / f"{row['input']}-{variant}--1")
             for row in policy_result['results'] for variant in ('auto', 'buffered')]
    paths.append(('alpha16-ramp', 'auto', validation / 'asan-alpha16-ramp'))
    if selected:
        if len(set(selected)) != len(selected) or set(selected) - {f'{n}:{v}' for n, v, _ in paths}:
            raise ValueError('Unknown or repeated stream replay case')
        paths = [(n, v, p) for n, v, p in paths if f'{n}:{v}' in selected]
    cases = []
    for name, variant, directory in paths:
        result = json.loads((directory / 'stdout.json').read_text())
        validate_trace(directory, result)
        for path in numbered(directory / 'streams', 'learning'):
            trace = json.loads(path.read_text())
            fit = result['learning']['fits'][trace['fit_begin']] if trace['fit_end'] > trace['fit_begin'] else None
            cases.append({'input': name, 'variant': variant, 'event': trace['event'],
                          'trace': str(path), 'trace_sha256': sha(path),
                          'pixels_sha256': sha(path.with_suffix('.i32le')), 'fit': fit})
    return cases, {**policy_result['artifacts'], **validation_result['artifacts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--policy-study', type=Path, required=True)
    parser.add_argument('--policy-validation', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='Optional INPUT:VARIANT subset for sanitizer controls')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.binary, args.policy_study, args.policy_validation, args.output = (
        p.resolve() for p in (args.binary, args.policy_study, args.policy_validation, args.output))
    cases, source_artifacts = capture_cases(args.policy_study, args.policy_validation, args.cases)
    args.output.mkdir(parents=True, exist_ok=False)
    binary, build = freeze_build(args.binary, args.output)
    build['artifacts'].update(source_artifacts)
    manifest = {'schema': 1, **build, 'policy_study': str(args.policy_study),
                'policy_validation': str(args.policy_validation), 'selected_cases': args.cases,
                'sanitizer_environment': {k: os.environ[k] for k in ('UBSAN_OPTIONS', 'ASAN_OPTIONS') if k in os.environ},
                'calls': cases, 'scope': 'Correctness and managed storage only. Reconstructed aligned planes from captured transformed pixels; full prepared-field equality and raw trees. No encodes, timing conclusion, selection policy or default promotion.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'results.jsonl').open('x', buffering=1) as ledger:
        for case in cases:
            directory = args.output / f"{case['input']}-{case['variant']}-{case['event']}"
            directory.mkdir()
            source = Path(case['trace']); pixels = source.with_suffix('.i32le')
            trace = json.loads(source.read_text())
            description = directory / 'input.txt'
            description.write_text(project(trace, pixels.stat().st_size))
            command = [str(binary), 'stream-prepare-check', str(description), str(pixels)]
            with (directory / 'stdout.json').open('x') as out, (directory / 'stderr.log').open('x') as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
            result = json.loads((directory / 'stdout.json').read_text())
            validate_result(trace, result, case['fit'])
            if sha(source) != case['trace_sha256'] or sha(pixels) != case['pixels_sha256']:
                raise ValueError('Capture changed during replay')
            ledger.write(json.dumps({'input': case['input'], 'variant': case['variant'], 'event': case['event'],
                                     'command': command, 'result': result,
                                     'artifacts': {str(p): sha(p) for p in directory.iterdir()}}) + '\n')
            print(case['input'], case['variant'], case['event'], result['samples'], result['nodes'], flush=True)


if __name__ == '__main__':
    main()
