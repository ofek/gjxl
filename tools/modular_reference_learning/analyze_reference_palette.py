#!/usr/bin/env python3
"""Recheck the frozen reference-palette differential, resource and coding controls."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha

PIN = 'e8ff09762481785938d8e4e01333ed3917571161'
TESTS = {
    'modular_stream', 'modular_foundation', 'modular_transform_contract',
    'modular_reference_palette', 'modular_prepared_stream', 'modular_histogram_preparation',
    'modular_selection_cost', 'modular_transform', 'modular_ordered_transform',
    'modular_reference_palette_oracle', 'modular_prepared_stream_reference',
    'modular_selection_cost_reference', 'modular_selection_cost_scalar_reference',
}


def qualify(validation, baseline):
    frozen = json.loads((validation / 'manifest.json').read_text())
    prior = json.loads((baseline / 'manifest.json').read_text())
    if (frozen['schema'] != 1 or frozen['kind'] != 'native-reference-palette' or
            frozen['reference_revision'] != PIN or not frozen['source_snapshot_is_authoritative'] or
            frozen['sanitizer_environment'] !=
            {'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'}):
        raise ValueError('Invalid palette validation protocol')
    for path, digest in frozen['artifacts'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Changed palette artifact: {path}')

    def bound(path):
        if str(path) not in frozen['artifacts']:
            raise ValueError(f'Unbound palette artifact: {path}')
        return path

    generated = None
    for build in ('release', 'asan'):
        root = validation / build
        log = bound(root / 'LastTest.log').read_text()
        summary = bound(root / 'ctest.log').read_text()
        names = set(re.findall(r'^\d+/\d+ Testing: (\S+)$', log, re.M))
        if (names != TESTS or log.count('Test Passed.') != len(TESTS) or
                not re.search(r'^100% tests passed(?:, 0 tests failed)? out of 13$', summary, re.M)):
            raise ValueError('Incomplete palette CTest controls')
        rows = [tuple(map(int, row)) for row in re.findall(
            r'^palette cases=(\d+) accepted=(\d+) empty=(\d+) enlarged=(\d+) '
            r'allocation_failures=(\d+) max_managed_peak=(\d+) files=(\d+)$', log, re.M)]
        if (len(rows) != 2 or rows[0][:-1] != rows[1][:-1] or
                rows[0] != (876, 588, 116, 8, 70, 27424, 0) or rows[1][-1] != 172):
            raise ValueError('Missing palette construction, resource or decode controls')
        if generated is not None and generated != rows:
            raise ValueError('Release/sanitizer palette results differ')
        generated = rows
        if ('Prepared stream reference: 108 cases, 648 exact files' not in log or
                log.count('Transform-cost reference: 396 cases, 2372 exact estimates, 104 exact RCT rankings') != 2 or
                log.count('Weighted-cost reference: 1980 exact estimates, 396 exact rankings, '
                          '523 rejected candidates, 72463 wrapped history updates, 22 finite wrapped candidates') != 2):
            raise ValueError('Missing prepared-stream or selection-estimator regressions')
        for name in TESTS:
            # Two ordered/unordered tests share the same transform executable.
            executable = 'modular_transform' if name == 'modular_ordered_transform' else name
            bound(root / ('gjxl_' + executable + '_test'))
        commands = json.loads(bound(root / 'compile_commands.json').read_text())
        for source in ('/src/codec/modular/transform/reference_palette.cpp',
                       '/src/codec/modular/image.cpp', '/src/codestream/modular/prepared_streams.cpp',
                       '/tests/modular_reference_palette_test.cpp'):
            selected = [c['command'] for c in commands if c['file'].endswith(source)]
            if not selected or any(('-fsanitize=address,undefined' in c) != (build == 'asan')
                                   for c in selected):
                raise ValueError(f'Unexpected palette instrumentation: {source}')
        for directory, expected_names in (
            ('palette', {f'case-{i}.jxl' for i in range(172)}),
            ('generated', {f'case-{i}.jxl' for i in range(216)} |
             {f'{mode}-case-{i}.jxl' for mode in ('direct32', 'direct64', 'estimated32', 'estimated64')
              for i in range(108)}),
        ):
            folder = root / directory
            if {p.name for p in folder.iterdir()} != expected_names:
                raise ValueError(f'Incomplete palette/coding files: {folder}')
            for name in expected_names:
                current = bound(folder / name)
                if sha(current) != sha(bound(validation / 'release' / directory / name)):
                    raise ValueError('Release/sanitizer files differ')
                if directory == 'generated':
                    old = baseline / build / 'generated' / name
                    if sha(old) != prior['artifacts'].get(str(old)) or sha(current) != sha(old):
                        raise ValueError('Existing prepared-stream bytes changed')
    return {
        'schema': 1, 'reference_revision': PIN,
        'scope': ('Native lossless palette construction and prescribed coding, compared with pinned '
                  'FwdPalette/InvPalette on the qualified Apple Clang/libc++ build. No automatic '
                  'palette/RCT/WP/scope selection, rate, timing or default promotion claim.'),
        'qualification': {
            'builds': ['Release', 'ASan/UBSan'], 'ctest_entries_per_build': len(TESTS),
            'generated_cases': generated[0][0], 'accepted': generated[0][1],
            'rejected': generated[0][0] - generated[0][1], 'empty_explicit_dictionaries': generated[0][2],
            'explicit_counts_exceeding_requested_limit': generated[0][3],
            'injected_allocation_failures_per_build': generated[0][4],
            'maximum_observed_candidate_managed_peak': generated[0][5],
            'dictionary_stack_bytes_less_than': 35 * 1024,
            'new_complete_files_per_build': generated[1][-1],
            'unchanged_existing_files_per_build': 648,
            'generated_release_sanitizer_identical': 820,
            'supported_component_counts': [1, 2, 3, 4], 'supported_bit_depth_range': [1, 24],
            'tested_bit_depths': [1, 2, 3, 4, 7, 8, 12, 16, 24],
            'maximum_requested_colors': 1024, 'maximum_explicit_colors': 1213,
            'managed_storage_qualified': True, 'timing_qualified': False,
            'native_policy_qualified': False, 'default_promotion_qualified': False,
            'sanitizer_limits': frozen['sanitizer_limits'],
        },
        'validation_manifest': str(validation / 'manifest.json'),
        'validation_manifest_sha256': sha(validation / 'manifest.json'),
        'bound_validation_files': len(frozen['artifacts']),
        'baseline_validation_manifest': str(baseline / 'manifest.json'),
        'baseline_validation_manifest_sha256': sha(baseline / 'manifest.json'),
        'analyzer_sha256': sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('validation', 'baseline-validation', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.validation.resolve(), args.baseline_validation.resolve())
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
