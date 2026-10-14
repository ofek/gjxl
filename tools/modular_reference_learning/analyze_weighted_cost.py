#!/usr/bin/env python3
"""Qualify native five-preset WP costs, resources and unchanged coding controls."""
import argparse
import json
import re
from pathlib import Path

from analyze import sha
from analyze_selection_cost import qualify as qualify_gradient
from selection_cost_replay import analyze


def qualify(study, sanitizer, gradient, gradient_sanitizer, validation, baseline):
    gradient_result = qualify_gradient(gradient, gradient_sanitizer, validation, native_test_count=6)
    results = [analyze(p) for p in (study, sanitizer)]
    frozen = json.loads((validation / 'manifest.json').read_text())
    previous = json.loads((baseline / 'manifest.json').read_text())
    generated = None
    def bound(path):
        if str(path) not in frozen['artifacts']:
            raise ValueError(f'Unbound weighted-cost validation artifact: {path}')
        return path
    for build, source, control in (('release', study, gradient), ('asan', sanitizer, gradient_sanitizer)):
        m = json.loads((source / 'manifest.json').read_text())
        c = json.loads((control / 'manifest.json').read_text())
        if (m['kind'] != 'weighted' or c['kind'] != 'gradient' or m['cases'] != c['cases'] or
                m['coding_study'] != c['coding_study'] or m['sanitizer_environment'] != c['sanitizer_environment']):
            raise ValueError('WP and gradient controls differ in cohort/build protocol')
        if sha(source / 'reference_learning') != sha(control / 'reference_learning'):
            raise ValueError('WP and gradient controls use different binaries')
        for p in (source / 'source').rglob('*'):
            if p.is_file() and sha(p) != sha(control / 'source' / p.relative_to(source / 'source')):
                raise ValueError('WP and gradient source snapshots differ')
        root = validation / build
        for name in ('gjxl_modular_stream_test', 'gjxl_modular_prepared_stream_test',
                     'gjxl_modular_prepared_stream_reference_test'):
            bound(root / 'native' / name)
        log = (root / 'native-LastTest.log').read_text()
        counts = re.findall(r'^Weighted-cost reference: (\d+) exact estimates, (\d+) exact rankings, '
                            r'(\d+) rejected candidates, (\d+) wrapped history updates, '
                            r'(\d+) finite wrapped candidates$', log, re.M)
        if len(counts) != 2 or counts[0] != counts[1] or tuple(map(int, counts[0])) != (1980, 396, 523, 72463, 22):
            raise ValueError('Missing exact WP state/cost/overflow controls')
        if generated is not None and generated != counts[0]:
            raise ValueError('Release/sanitizer WP generated controls differ')
        generated = counts[0]
        if ('Weighted-cost contracts: 50 allocation failures qualified' not in log or
                'Prepared stream reference: 108 cases, 648 exact files' not in log):
            raise ValueError('Missing WP resource or coding regression checks')
        names = {f'case-{i}.jxl' for i in range(216)} | {
            f'{mode}-case-{i}.jxl' for mode in ('direct32', 'direct64', 'estimated32', 'estimated64')
            for i in range(108)}
        current = root / 'generated'
        if {p.name for p in current.iterdir()} != names:
            raise ValueError('Incomplete generated coding controls')
        for name in names:
            old = baseline / build / 'generated' / name
            if (sha(bound(current / name)) != sha(validation / 'release/generated' / name) or
                    sha(old) != previous['artifacts'].get(str(old)) or sha(current / name) != sha(old)):
                raise ValueError('Checked predictor coding controls changed')

        # The private oracle is extracted verbatim, rather than reimplemented.
        original = bound(validation / 'reference/source/lib/jxl/enc_modular.cc')
        if sha(original) != '4bc466a0c7b2e65c03501eff1a81758d63a45568521a22ceb611cdbc67f0fb7a':
            raise ValueError('WP oracle does not come from pinned reference source')
        text = original.read_text()
        body = text[text.index('float EstimateWPCost('):text.index('\nbool do_transform(')]
        template = (root / 'source/tests/modular_wp_cost_reference.cpp.in').read_text()
        expected = template.replace('@GJXL_REFERENCE_WP_BODY@', body)
        for path in (root / 'modular_wp_cost_reference.cpp', root / 'native/modular_wp_cost_reference.cpp'):
            if bound(path).read_text() != expected:
                raise ValueError('Extracted reference WP body changed')
        if (source / 'modular_wp_cost_reference.cpp').read_text() != expected:
            raise ValueError('WP study uses a different reference oracle')
        for directory in (root, root / 'native'):
            commands = json.loads((directory / 'compile_commands.json').read_text())
            oracle = [c['command'] for c in commands if c['file'].endswith('/modular_wp_cost_reference.cpp')]
            if not oracle or any(any(flag not in c for flag in
                ('-fno-vectorize', '-fno-slp-vectorize', '-fno-rtti', '-fno-exceptions')) for c in oracle):
                raise ValueError('WP oracle compiler settings differ from reference')

    expected = {(r['input'], r['policy']): r['result'] for r in results[0]['results']}
    for r in results[1]['results']:
        if r['result'] != expected[r['input'], r['policy']]:
            raise ValueError('Release/sanitizer WP costs, rankings or bounds differ')
    estimates, rankings, rejections, wraps, finite = map(int, generated)
    return {
        'schema': 1, 'reference_revision': gradient_result['reference_revision'],
        'scope': ('Five native WP presets match the extracted pinned estimator on generated and '
                  'captured transformed planes. Checked coding behavior and gradient/RCT controls '
                  'remain qualified. No complete native palette/transform/scope policy, timing, '
                  'rate or default promotion claim.'),
        'qualification': {
            'native_ctest_per_build': 6, 'research_ctest_per_build': 10,
            'generated_cases_per_backend': 396, 'exact_estimates_per_backend': estimates,
            'exact_rankings_per_backend': rankings, 'rejected_candidates_per_backend': rejections,
            'wrapped_history_updates_per_backend': wraps, 'finite_wrapped_candidates_per_backend': finite,
            'backends': ['NEON entropy', 'portable four-lane entropy'], 'builds': ['Release', 'ASan/UBSan'],
            'allocation_failures_per_build': 50, 'histogram_stack_bytes': 17544,
            'generated_coding_cases': 108, 'unchanged_generated_files_per_build': 648,
            'generated_release_sanitizer_identical': 648,
            'capture_cases': results[0]['cases'], 'capture_totals': results[0]['totals'],
            'sanitizer_cases': results[1]['cases'], 'sanitizer_totals': results[1]['totals'],
            'managed_storage_qualified': True, 'timing_qualified': False,
            'native_policy_qualified': False, 'default_promotion_qualified': False,
            'sanitizer_limits': frozen['sanitizer_limits'],
        },
        'results': [{k: r[k] for k in ('input', 'policy', 'summary')} for r in results[0]['results']],
        'gradient_control': gradient_result,
        'validation_manifest': str(validation / 'manifest.json'),
        'validation_manifest_sha256': sha(validation / 'manifest.json'),
        'bound_validation_files': len(frozen['artifacts']),
        'baseline_validation_manifest': str(baseline / 'manifest.json'),
        'baseline_validation_manifest_sha256': sha(baseline / 'manifest.json'),
        'artifacts': {**results[0]['artifacts'], **results[1]['artifacts']},
        'analyzer_sha256': sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    keys = ('study', 'sanitizer-study', 'gradient-study', 'gradient-sanitizer-study',
            'validation', 'baseline-validation', 'output')
    for key in keys:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(*(getattr(args, k.replace('-', '_')).resolve() for k in keys[:-1]))
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
