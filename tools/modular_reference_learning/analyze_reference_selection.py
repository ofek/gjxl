#!/usr/bin/env python3
"""Qualify native reference selection and its unchanged coding controls."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from analyze_reference_palette import PIN, TESTS


def qualify(validation, baseline):
    manifest = validation / 'manifest.json'
    frozen = json.loads(manifest.read_text())
    previous_manifest = baseline / 'manifest.json'
    previous = json.loads(previous_manifest.read_text())
    if (frozen['schema'] != 1 or frozen['kind'] != 'native-reference-selection' or
            frozen['reference_revision'] != PIN or not frozen['source_snapshot_is_authoritative'] or
            frozen['sanitizer_environment'] !=
            {'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'} or
            previous['kind'] != 'native-reference-channel-palette' or previous['reference_revision'] != PIN):
        raise ValueError('Invalid reference selection protocol')
    for path, digest in frozen['artifacts'].items():
        if sha(path) != digest:
            raise ValueError(f'Changed reference selection artifact: {path}')

    def bound(path):
        if str(path) not in frozen['artifacts']:
            raise ValueError(f'Unbound reference selection artifact: {path}')
        return path

    expected_tests = TESTS | {
        'modular_reference_channel_palette', 'modular_reference_channel_palette_oracle',
        'modular_reference_selection', 'modular_reference_selection_oracle',
    }
    expected_summary = {
        'cases': 496, 'faults': 799, 'files': 216, 'managed_peak': 1106256,
        'palette_built': 205, 'palette_accepted': 180, 'palette_ties': 1,
        'palette_rejected_cost': 25, 'borrowed_results': 237, 'wide_palettes': 4,
        'rct_candidates': 3914, 'wp_candidates': 2475, 'shared_baseline_accepts': 1,
        'rct_winners': [342, 0, 0, 0, 0, 18, 88, 0, 0, 18, 12, 0, 0, 0, 0, 2, 8, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'wp_winners': [145, 130, 124, 36, 61],
    }
    source = bound(validation / 'reference/source/lib/jxl/enc_modular.cc')
    if sha(source) != '4bc466a0c7b2e65c03501eff1a81758d63a45568521a22ceb611cdbc67f0fb7a':
        raise ValueError('Changed reference selection source')
    reference_text = source.read_text()
    palette = reference_text[reference_text.index('bool do_transform('):reference_text.index(
        '\n}  // namespace\n\nStatusOr<std::unique_ptr<ModularFrameEncoder>>')]
    stream = reference_text[reference_text.index(
        '  // lossless and no specific color transform specified: try Nothing, YCoCg,'):
        reference_text.index('\nconstexpr float q_deadzone')]
    for build in ('release', 'asan'):
        root = validation / build
        log = bound(root / 'LastTest.log').read_text()
        summary = bound(root / 'ctest.log').read_text()
        names = set(re.findall(r'^\d+/\d+ Testing: (\S+)$', log, re.M))
        if (names != expected_tests or log.count('Test Passed.') != 17 or
                not re.search(r'^100% tests passed(?:, 0 tests failed)? out of 17$', summary, re.M)):
            raise ValueError('Incomplete selection CTest controls')
        for name in expected_tests:
            name = 'modular_transform' if name == 'modular_ordered_transform' else name
            name = name.replace('reference_channel_palette', 'reference_palette')
            bound(root / ('gjxl_' + name + '_test'))
        extracted = bound(root / 'modular_selection_reference.cpp').read_text()
        if extracted.count(palette) != 1 or extracted.count(stream) != 1:
            raise ValueError('Reference decision bodies were not extracted verbatim')
        if ('Prepared stream reference: 108 cases, 648 exact files' not in log or
                log.count('Transform-cost reference: 396 cases, 2372 exact estimates, 104 exact RCT rankings') != 2 or
                log.count('Weighted-cost reference: 1980 exact estimates, 396 exact rankings, '
                          '523 rejected candidates, 72463 wrapped history updates, 22 finite wrapped candidates') != 2):
            raise ValueError('Missing prepared-coding/estimator controls')
        for kind in ('selection', 'selection-native'):
            actual = json.loads(bound(root / kind / 'summary.json').read_text())
            expected = dict(expected_summary, files=216 if kind == 'selection' else 0)
            if actual != expected:
                raise ValueError(f'Incomplete selection coverage: {root / kind}')
            if json.dumps(actual, separators=(',', ':')) not in log:
                raise ValueError('Selection summary is not bound to the CTest run')
            decisions = bound(root / kind / 'decisions.jsonl')
            rows = [json.loads(line) for line in decisions.read_text().splitlines()]
            if [r['case'] for r in rows] != list(range(496)):
                raise ValueError('Incomplete selection decisions ledger')
            if sha(decisions) != sha(bound(validation / 'release/selection/decisions.jsonl')):
                raise ValueError('Native/oracle/Release/sanitizer decisions differ')
        commands = json.loads(bound(root / 'compile_commands.json').read_text())
        for suffix in ('/src/codec/modular/reference_selection.cpp', '/src/codec/modular/selection_cost.cpp',
                       '/src/codec/modular/transform/reference_palette.cpp',
                       '/tests/modular_reference_selection_test.cpp', '/modular_selection_reference.cpp'):
            selected = [c['command'] for c in commands if c['file'].endswith(suffix)]
            if not selected or any(('-fsanitize=address,undefined' in c) != (build == 'asan')
                                   for c in selected):
                raise ValueError(f'Unexpected selection instrumentation: {suffix}')
            if suffix == '/modular_selection_reference.cpp' and any(
                    not all(flag in c for flag in ('-fno-vectorize', '-fno-slp-vectorize', '-fno-sanitize=vptr'))
                    for c in selected):
                raise ValueError('Reference oracle math/sanitizer settings changed')
        for directory, expected_names in (
            ('selection', {f'selected-{i}.jxl' for i in range(216)}),
            ('channel-palette', {f'case-{i}.jxl' for i in range(64)}),
            ('palette', {f'case-{i}.jxl' for i in range(172)}),
            ('generated', {f'case-{i}.jxl' for i in range(216)} |
             {f'{mode}-case-{i}.jxl' for mode in ('direct32', 'direct64', 'estimated32', 'estimated64')
              for i in range(108)}),
        ):
            folder = root / directory
            if {p.name for p in folder.glob('*.jxl')} != expected_names:
                raise ValueError(f'Incomplete generated files: {folder}')
            for name in expected_names:
                current = bound(folder / name)
                if sha(current) != sha(bound(validation / 'release' / directory / name)):
                    raise ValueError('Release/sanitizer files differ')
                if directory != 'selection':
                    old = baseline / build / directory / name
                    if sha(old) != previous['artifacts'].get(str(old)) or sha(current) != sha(old):
                        raise ValueError('Prior palette/prepared-stream bytes changed')
    return {
        'schema': 1, 'reference_revision': PIN,
        'scope': ('Private lossless e9 palette/RCT/WP stage selection on matched input planes. '
                  'Generated complete files use prescribed weighted trees. This does not qualify '
                  'automatic frame/group scope, one-fit orchestration, full rate or whole-encode timing.'),
        'qualification': {
            'ctest_entries_per_build': 17, 'builds': ['Release', 'ASan/UBSan'],
            'selection': expected_summary,
            'exact_cost_sequences_and_transformed_pixels': True,
            'all_five_wp_winners_exercised': True,
            'palette_tie_and_shared_baseline_controls': True,
            'exact_alias_and_atomic_allocation_failure_controls': True,
            'selection_managed_storage_qualified': True,
            'prior_complete_files_unchanged': 884,
            'release_sanitizer_identical_complete_files': 1100,
            'automatic_frame_policy_qualified': False, 'timing_qualified': False,
            'rate_qualified': False, 'default_promotion_qualified': False,
            'sanitizer_limits': frozen['sanitizer_limits'],
        },
        'validation_manifest': str(manifest), 'validation_manifest_sha256': sha(manifest),
        'bound_validation_files': len(frozen['artifacts']),
        'baseline_validation_manifest': str(previous_manifest),
        'baseline_validation_manifest_sha256': sha(previous_manifest),
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
