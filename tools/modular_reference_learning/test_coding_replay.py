#!/usr/bin/env python3
"""Check captured coding scope, bounded-reader and evidence failure paths."""
import argparse
import copy
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_replay import project, validate_result
from test_stream_replay import empty_trace

BINARY = None


class CodingReplayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gjxl-coding-replay-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'streams').mkdir()
        self.source = self.root / 'source.pgm'
        self.source.write_bytes(b'P5\n3 3\n255\n' + bytes(range(9)))
        self.learning = empty_trace()
        self.learning.update(pixel_bytes=36, fit_end=1)
        image = self.learning['images'][0]
        image.update(width=3, height=3)
        image['channels'][0].update(width=3, height=3, included=True, bytes=36)
        self.coding = {'schema': 1, 'event': 0, 'stream_id': 0, 'local_tree': True, 'channels': 1,
                       **{k: copy.deepcopy(image[k]) for k in ('width', 'height', 'metadata_channels',
                                                               'options', 'transforms')}}
        (self.root / 'streams/learning-0.i32le').write_bytes(struct.pack('<9i', *range(9)))

    def save(self):
        (self.root / 'streams/learning-0.json').write_text(json.dumps(self.learning))
        (self.root / 'streams/coding-0.json').write_text(json.dumps(self.coding))

    def run_replay(self, text):
        path = self.root / 'description.txt'
        path.write_text(text)
        return subprocess.run([str(BINARY), 'stream-encode-check', str(self.source), str(path),
                               str(self.root / 'encoded')], capture_output=True, text=True)

    def test_local_tree_scope(self):
        self.save()
        run = self.run_replay(project(self.root))
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        validate_result(self.root, result)
        self.assertEqual((result['local_trees'], result['global_trees'], result['canonical_slots']), (1, 0, 3))
        for name, value in [('tokens_exact', 1), ('trees_exact', False), ('decoded_exact', False),
                            ('timing_qualified', True), ('tokens', 10), ('local_trees', 0),
                            ('global_trees', 1), ('transforms', 1), ('coded_streams', 0)]:
            with self.subTest(name=name):
                bad = {**result, name: value}
                with self.assertRaises(ValueError): validate_result(self.root, bad)
        bad = copy.deepcopy(result)
        bad['encodes'][1]['storage_bound_bytes'] = 0
        with self.assertRaises(ValueError): validate_result(self.root, bad)

    def test_shared_tree_scope(self):
        self.learning['local_stream_id'] = -1
        self.coding['local_tree'] = False
        self.save()
        run = self.run_replay(project(self.root))
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        validate_result(self.root, result)
        self.assertEqual((result['local_trees'], result['global_trees']), (0, 1))

    def test_selection_cost_projection(self):
        from selection_cost_replay import validate_result as validate_cost
        self.save()
        path = self.root / 'cost.txt'; path.write_text(project(self.root))
        run = subprocess.run([str(BINARY), 'stream-cost-check', str(self.source), str(path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        source = {'width': 3, 'height': 3}
        self.assertEqual(validate_cost(self.root, source, result)['exact_estimates'], 3)
        for key, value in [('costs_exact', False), ('managed_allocations', 1), ('timing_qualified', True)]:
            with self.assertRaises(ValueError): validate_cost(self.root, source, {**result, key: value})
        for key, value in [('native_bits', 99), ('reference_bits', 99), ('pixels', 99)]:
            bad = copy.deepcopy(result); bad['streams'][0]['base'][key] = value
            with self.assertRaises(ValueError): validate_cost(self.root, source, bad)

    def test_selection_cost_rct_ties(self):
        from selection_cost_replay import validate_result as validate_cost
        self.source.write_bytes(b'P6\n3 3\n255\n' + bytes(27))
        image = self.learning['images'][0]
        image['channels'] = [{**image['channels'][0], 'component': c, 'offset': 36 * c}
                             for c in range(3)]
        self.learning['pixel_bytes'] = 108
        self.coding['channels'] = 3
        (self.root / 'streams/learning-0.i32le').write_bytes(bytes(108))
        self.save()
        path = self.root / 'cost.txt'; path.write_text(project(self.root))
        run = subprocess.run([str(BINARY), 'stream-cost-check', str(self.source), str(path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        source = {'width': 3, 'height': 3}
        self.assertEqual(validate_cost(self.root, source, result)['exact_rct_rankings'], 1)
        self.assertEqual(result['streams'][0]['selected_rct'], 0)
        for change in ('missing', 'reordered', 'winner', 'identity', 'scope'):
            bad = copy.deepcopy(result); row = bad['streams'][0]
            if change == 'missing': row['rct_candidates'].pop()
            if change == 'reordered': row['rct_candidates'].reverse()
            if change == 'winner': row['selected_rct'] = 6
            if change == 'identity': row['rct_candidates'][0]['cost']['extra_bits'] += 1
            if change == 'scope': bad['streams'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cost(self.root, source, bad)

    def test_projection_rejects_semantic_mismatch(self):
        good = copy.deepcopy(self.coding)
        for key, value in [('schema', 2), ('width', 4), ('stream_id', 1), ('local_tree', False),
                            ('channels', 2), ('metadata_channels', 1), ('options', {})]:
            with self.subTest(key=key):
                self.coding = {**good, key: value}
                self.save()
                with self.assertRaises(ValueError): project(self.root)
        self.coding = good
        for transform in [{'id': 2, 'begin_c': 0},
                          {'id': 1, 'begin_c': 0, 'num_c': 1, 'nb_colors': 1025,
                           'nb_deltas': 0, 'predictor': 0, 'lossy_palette': False},
                          {'id': 1, 'begin_c': 0, 'num_c': 1, 'nb_colors': 5,
                           'nb_deltas': 1, 'predictor': 0, 'lossy_palette': False}]:
            self.coding['transforms'] = [transform]
            self.learning['images'][0]['transforms'] = [transform]
            self.save()
            with self.assertRaises(ValueError): project(self.root)

    def test_reader_rejects_before_publication(self):
        self.save()
        good = project(self.root)
        bad = [good.replace('gjxl-coding-input-v1', 'v0'), good[:-5], good + 'trailing\n',
               good.replace('stream 0 3 3 8 0 1', 'stream 21 3 3 8 0 1'),
               good.replace('stream 0 3 3 8 0 1', 'stream 0 3 3 16 0 1'),
               good.replace('channel 3 3 0 0 0 36', 'channel 3 3 0 0 4 36'),
               good.replace('channel 3 3 0 0 0 36', 'channel 3 3 0 0 0 32'),
               good.replace('channel 3 3 0 0 0 36', 'channel 3 3 1 0 0 36'),
               good.replace('wp 16', 'wp 32'), good.replace('wp 16', 'wp 15'),
               good.replace('-1 0 0 0 6 0 1', '-1 0 0 0 6 0 2')]
        for text in bad:
            with self.subTest(text=text):
                run = self.run_replay(text)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertFalse((self.root / 'encoded').exists())
                self.assertFalse(run.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args, rest = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *rest])
