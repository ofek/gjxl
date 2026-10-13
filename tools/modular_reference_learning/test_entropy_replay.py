#!/usr/bin/env python3
"""Exercise entropy replay isolation, syntax accounting and rejected controls."""
import argparse
import copy
import json
import subprocess
import unittest
from pathlib import Path

import test_coding_replay
from coding_replay import project
from entropy_replay import VARIANTS, validate_controls, validate_result

BINARY = None


class EntropyTest(test_coding_replay.CodingReplayTest):
    # Reuse only fixture setup. Inherited native replay tests remain useful
    # regressions and run against the same newly linked research executable.
    def setUp(self):
        super().setUp()
        params = {'clustering': 2, 'uint_method': 4, 'lz77_method': 15,
                  'ans_histogram_strategy': 2, 'max_histograms': (1 << 64) - 1,
                  'force_huffman': False, 'initialize_global_state': True,
                  'streaming_mode': False, 'add_missing_symbols': False,
                  'add_fixed_histograms': False, 'image_widths': []}
        self.learning['images'][0]['options']['histogram_params'] = params
        self.coding['options']['histogram_params'] = copy.deepcopy(params)

    def run_entropy(self, text, variants=VARIANTS, timing=False):
        path = self.root / 'entropy.txt'; path.write_text(text)
        return subprocess.run([str(BINARY), 'stream-entropy-time' if timing else 'stream-entropy-check', str(self.source), str(path),
                               str(self.root / 'encoded'), ','.join(variants)], capture_output=True, text=True)

    def test_all_entropy_controls(self):
        self.save()
        run = self.run_entropy(project(self.root, entropy=True))
        self.assertEqual(run.returncode, 0, run.stderr)
        results = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertEqual([r['variant'] for r in results], list(VARIANTS))
        for row in results: validate_result(self.root, row)
        self.assertEqual(validate_controls(results), 2)
        with self.assertRaises(ValueError):
            validate_controls([r for r in results if r['variant'] != 'no_lz77_clusters32'])
        bad = copy.deepcopy(results)
        transplant = next(r for r in bad if r['variant'] == 'reference_map32')
        next(s for s in transplant['scopes'] if s['kind'] == 'data')['context_map'][0] += 1
        with self.assertRaises(ValueError): validate_controls(bad)
        with self.assertRaises(ValueError):
            validate_controls([r for r in results if r['variant'] != 'native'])
        direct = next(r for r in results if r['variant'] == 'native_direct_balanced')
        bad = copy.deepcopy(direct)
        next(s for s in bad['scopes'] if s['kind'] == 'data')['prefix'] = True
        with self.assertRaises(ValueError): validate_result(self.root, bad)
        estimated = next(r for r in results if r['variant'] == 'native_direct_estimated')
        bad = copy.deepcopy(estimated)
        next(s for s in bad['scopes'] if s['kind'] == 'data')['ans_work']['token_cost_candidates'] = 1
        with self.assertRaises(ValueError): validate_result(self.root, bad)
        good = results[0]
        for key, value in [('tokens_exact', 1), ('decoded_exact', False), ('timing_qualified', True),
                           ('native_control_exact', True), ('original_tokens', 10), ('bytes', 0),
                           ('tree_bits', 0), ('scopes', good['scopes'][:-1])]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError): validate_result(self.root, {**good, key: value})
        bad = copy.deepcopy(good); bad['scopes'][0]['implementation'] = 'native'
        with self.assertRaises(ValueError): validate_result(self.root, bad)

    def test_entropy_options_are_not_ignored(self):
        self.save()
        good = project(self.root, entropy=True)
        for text in [good.replace('gjxl-coding-input-v2', 'gjxl-coding-input-v1'),
                     good.replace('entropy 2 4 15 2 -1 0 1', 'entropy 2 4 15 2 -1 0 0'),
                     good.replace('entropy 2 4 15 2 -1', 'entropy 2 4 15 2 0'),
                     good.replace('entropy 2 4 15 2 -1', 'entropy 2 4 16 2 -1')]:
            run = self.run_entropy(text, ['reference'])
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertFalse((self.root / 'encoded').exists())
        for names in [[], ['bad'], ['reference', 'reference'], ['reference', '']]:
            run = self.run_entropy(good, names)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertFalse((self.root / 'encoded').exists())
        self.learning['images'][0]['options']['histogram_params']['image_widths'] = [3]
        self.coding['options'] = copy.deepcopy(self.learning['images'][0]['options'])
        self.save()
        with self.assertRaises(ValueError): project(self.root, entropy=True)

    def test_unprofiled_model_timing(self):
        self.save()
        variants = ['native', 'native_direct_rate', 'native_direct_estimated']
        run = self.run_entropy(project(self.root, entropy=True), variants, timing=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        results = [json.loads(line) for line in run.stdout.splitlines()]
        for row in results:
            self.assertIs(row['model_timing_recorded'], True)
            validate_result(self.root, row)
            bad = copy.deepcopy(row)
            bad['scopes'][0]['model_wall_seconds'] = -1
            with self.assertRaises(ValueError): validate_result(self.root, bad)
        validate_controls(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args, rest = parser.parse_known_args()
    BINARY = test_coding_replay.BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *rest])
