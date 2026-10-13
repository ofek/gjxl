#!/usr/bin/env python3
"""Qualify wide single-channel palettes and preserve the prior palette/coding controls."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from analyze_reference_palette import PIN, TESTS


def qualify(validation, baseline):
    manifest = validation / 'manifest.json'
    frozen = json.loads(manifest.read_text())
    previous = json.loads((baseline / 'manifest.json').read_text())
    if (frozen['schema'] != 1 or frozen['kind'] != 'native-reference-channel-palette' or
            frozen['reference_revision'] != PIN or not frozen['source_snapshot_is_authoritative'] or
            frozen['sanitizer_environment'] !=
            {'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'}):
        raise ValueError('Invalid channel palette protocol')
    for path, digest in frozen['artifacts'].items():
        if sha(path) != digest:
            raise ValueError(f'Changed channel palette artifact: {path}')

    def bound(path):
        if str(path) not in frozen['artifacts']:
            raise ValueError(f'Unbound channel palette artifact: {path}')
        return path

    expected_tests = TESTS | {'modular_reference_channel_palette', 'modular_reference_channel_palette_oracle'}
    generated = None
    for build in ('release', 'asan'):
        root = validation / build
        log = bound(root / 'LastTest.log').read_text()
        summary = bound(root / 'ctest.log').read_text()
        names = set(re.findall(r'^\d+/\d+ Testing: (\S+)$', log, re.M))
        if (names != expected_tests or log.count('Test Passed.') != 15 or
                not re.search(r'^100% tests passed(?:, 0 tests failed)? out of 15$', summary, re.M)):
            raise ValueError('Incomplete channel palette CTest controls')
        for name in expected_tests:
            name = 'modular_transform' if name == 'modular_ordered_transform' else name
            name = name.replace('reference_channel_palette', 'reference_palette')
            bound(root / ('gjxl_' + name + '_test'))
        rows = [tuple(map(int, row)) for row in re.findall(
            r'^palette cases=(\d+) accepted=(\d+) empty=(\d+) enlarged=(\d+) '
            r'allocation_failures=(\d+) max_managed_peak=(\d+) files=(\d+)$', log, re.M)]
        expected = [(876, 588, 116, 8, 82, 27424, 0), (34, 22, 0, 0, 44, 1899964, 0),
                    (876, 588, 116, 8, 82, 27424, 172), (34, 22, 0, 0, 44, 1899964, 64)]
        if rows != expected or 'wide palette headers=160' not in log:
            raise ValueError('Missing channel palette differential, resource or header controls')
        if generated is not None and rows != generated:
            raise ValueError('Release/sanitizer channel palette controls differ')
        generated = rows
        if ('Prepared stream reference: 108 cases, 648 exact files' not in log or
                log.count('Transform-cost reference: 396 cases, 2372 exact estimates, 104 exact RCT rankings') != 2 or
                log.count('Weighted-cost reference: 1980 exact estimates, 396 exact rankings, '
                          '523 rejected candidates, 72463 wrapped history updates, 22 finite wrapped candidates') != 2):
            raise ValueError('Missing coding/selection regressions')
        commands = json.loads(bound(root / 'compile_commands.json').read_text())
        for source in ('/src/codec/modular/transform/reference_palette.cpp',
                       '/src/codestream/modular/storage_plan.cpp', '/src/codestream/modular/stream_encoder.cpp',
                       '/tests/modular_reference_palette_test.cpp'):
            selected = [c['command'] for c in commands if c['file'].endswith(source)]
            if not selected or any(('-fsanitize=address,undefined' in c) != (build == 'asan')
                                   for c in selected):
                raise ValueError(f'Unexpected channel palette instrumentation: {source}')
        for directory, expected_names in (
            ('channel-palette', {f'case-{i}.jxl' for i in range(64)}),
            ('palette', {f'case-{i}.jxl' for i in range(172)}),
            ('generated', {f'case-{i}.jxl' for i in range(216)} |
             {f'{mode}-case-{i}.jxl' for mode in ('direct32', 'direct64', 'estimated32', 'estimated64')
              for i in range(108)}),
        ):
            folder = root / directory
            if {p.name for p in folder.iterdir()} != expected_names:
                raise ValueError(f'Incomplete generated files: {folder}')
            for name in expected_names:
                current = bound(folder / name)
                if sha(current) != sha(bound(validation / 'release' / directory / name)):
                    raise ValueError('Release/sanitizer files differ')
                if directory != 'channel-palette':
                    old = baseline / build / directory / name
                    if sha(old) != previous['artifacts'].get(str(old)) or sha(current) != sha(old):
                        raise ValueError('Prior palette/prepared-stream bytes changed')
    return {
        'schema': 1, 'reference_revision': PIN,
        'scope': ('Wide single-channel construction and prescribed coding through the palette wire '
                  'maximum. Does not implement automatic proposal acceptance or RCT/WP/scope selection; '
                  'no rate, timing or default-promotion claim.'),
        'qualification': {
            'ctest_entries_per_build': 15, 'builds': ['Release', 'ASan/UBSan'],
            'wide_generated_cases': 34, 'wide_accepted': 22, 'wide_rejected': 12,
            'wide_injected_allocation_failures': generated[1][4],
            'maximum_observed_wide_candidate_managed_peak': generated[1][5],
            'new_complete_files_per_build': 64, 'exact_wide_headers': 160,
            'maximum_single_channel_colors': 70911,
            'maximum_single_channel_colors_in_complete_files': 65536,
            'dense_value_span_limit': 65536, 'wide_dictionary_stack_bytes_less_than': 13 * 1024,
            'sparse_scratch': '4 * min(limit,pixels) + 4 * bit_ceil(2 * min(limit,pixels)); retained=0',
            'prior_palette_cases_unchanged': 876, 'prior_palette_fault_sweep_expanded_to': 82,
            'prior_complete_files_unchanged': 820,
            'release_sanitizer_identical_complete_files': 884,
            'managed_storage_qualified': True, 'native_policy_qualified': False,
            'timing_qualified': False, 'default_promotion_qualified': False,
            'sanitizer_limits': frozen['sanitizer_limits'],
        },
        'validation_manifest': str(manifest), 'validation_manifest_sha256': sha(manifest),
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
    result = qualify(args.validation.resolve(), args.baseline_validation.resolve())
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
