#!/usr/bin/env python3
"""Exercise replay projection, bounded reader and equality-evidence rejection."""
import argparse
import copy
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from stream_replay import project, validate_result

BINARY = None


def empty_trace():
    options = {'max_chan_size': 256, 'group_dim': 256, 'max_properties': 0,
               'max_property_values': 256, 'predictor': 14, 'wp_mode': 0, 'wp_tree_mode': 3,
               'tree_kind': 1, 'skip_encoder_fast_path': False, 'zero_tokens': False,
               'properties': [0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8],
               'wp_coefficients': [16, 10, 7, 7, 7, 0, 0], 'wp_weights': [13, 12, 12, 12]}
    for name, value, bits in [('sampling_fraction', .65, 'sampling_fraction_bits'),
                             ('split_threshold', 89, 'split_threshold_bits'),
                             ('fast_decode_multiplier', 1.001, 'fast_decode_bits')]:
        options[name] = struct.unpack('<f', struct.pack('<f', value))[0]
        options[bits] = struct.unpack('<I', struct.pack('<f', value))[0]
    channel = {'width': 512, 'height': 512, 'hshift': 0, 'vshift': 0, 'component': 0,
               'included': False, 'offset': 0, 'bytes': 0}
    return {'schema': 1, 'event': 0, 'local_stream_id': 0, 'start': 0, 'stop': 1,
            'fit_begin': 0, 'fit_end': 0, 'multipliers': [], 'pixel_bytes': 0,
            'pixels_verified': True, 'tree': [[-1, 0, 0, 0, 6, 0, 1]],
            'images': [{'index': 0, 'width': 512, 'height': 512, 'bitdepth': 8,
                        'metadata_channels': 0, 'options': options, 'channels': [channel], 'transforms': []}]}


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.trace = empty_trace()

    def run_replay(self, text, pixels=b''):
        with tempfile.TemporaryDirectory(prefix='gjxl-stream-replay-') as temporary:
            root = Path(temporary)
            (root / 'input.txt').write_text(text)
            (root / 'pixels.i32le').write_bytes(pixels)
            return subprocess.run([str(BINARY), 'stream-prepare-check', str(root / 'input.txt'),
                                   str(root / 'pixels.i32le')], capture_output=True, text=True)

    def test_empty_learning_with_excluded_channel(self):
        run = self.run_replay(project(self.trace, 0))
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        validate_result(self.trace, result)
        self.assertEqual((result['samples'], result['fits'], result['nodes']), (0, 0, 1))

    def test_projection_rejects_corruption(self):
        mutations = [lambda t: t.update(pixels_verified=False), lambda t: t.update(stop=2),
                     lambda t: t.update(pixel_bytes=4), lambda t: t.update(fit_end=2),
                     lambda t: t['images'][0].update(metadata_channels=2),
                     lambda t: t['images'][0]['channels'][0].update(offset=4),
                     lambda t: t['images'][0]['channels'][0].update(included=True),
                     lambda t: t['images'][0]['options'].update(sampling_fraction_bits=0),
                     lambda t: t['images'][0]['options'].update(wp_weights=[16] * 4),
                     lambda t: t['images'][0]['options'].update(max_properties=1),
                     lambda t: t['tree'][0].__setitem__(2, 1)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                trace = copy.deepcopy(self.trace); mutate(trace)
                with self.assertRaises(ValueError): project(trace, 0)

    def test_reader_rejects_corruption(self):
        good = project(self.trace, 0)
        inputs = [good.replace('gjxl-learning-input-v1', 'wrong-version'), good[:-5], good + 'trailing\n',
                  good.replace('range 0 1 0 0 0', 'range 0 65536 0 0 0'),
                  good.replace('image 0 512', 'image 0 1099511627776'),
                  good.replace('channel 512 512 0 0 0 0 0 0', 'channel 512 512 0 0 0 1 0 0'),
                  good.replace('wp 16', 'wp 32'), good.replace('wp 16', 'wp 15'),
                  good.replace('range 0 1 0 0 0', 'range 0 1 0 1 0'),
                  good.replace('-1 0 0 0 6 0 1', '-1 0 0 0 0 0 1')]
        for text in inputs:
            with self.subTest(text=text):
                run = self.run_replay(text)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertFalse(run.stdout)
        run = self.run_replay(good, b'\0')
        self.assertEqual(run.returncode, 1, run.stderr)

    def test_mixed_sampling_is_not_silently_ignored(self):
        self.trace.update(stop=2, local_stream_id=-1)
        other = copy.deepcopy(self.trace['images'][0]); other['index'] = 1
        self.trace['images'].append(other)
        good = project(self.trace, 0)
        other['options']['max_chan_size'] = 257
        with self.assertRaises(ValueError): project(self.trace, 0)
        prefix, suffix = good.rsplit('options 256', 1)
        run = self.run_replay(prefix + 'options 257' + suffix)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn('shared native profile', run.stderr)

    def test_evidence_rejects_false_claims(self):
        run = self.run_replay(project(self.trace, 0))
        self.assertEqual(run.returncode, 0, run.stderr)
        good = json.loads(run.stdout)
        fields = [('reference_tree_exact', False), ('samples_exact', 1), ('tree_exact', None),
                  ('timing_qualified', True), ('start', 1), ('fits', 1), ('nodes', 2),
                  ('samples', 1), ('population', 0), ('scanned_pixels', 1),
                  ('preparation_storage_bound_bytes', 0), ('fitting_storage_peak_bytes', 0)]
        for name, value in fields:
            with self.subTest(name=name):
                result = dict(good); result[name] = value
                with self.assertRaises(ValueError): validate_result(self.trace, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args, rest = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *rest])
