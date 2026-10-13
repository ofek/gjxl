#!/usr/bin/env python3
"""Project full captured stream scopes without rerunning learning/transforms."""
import json

from analyze_policy_study import numbered
from stream_replay import integer, project as project_learning


def project(directory):
    images, global_tree = {}, None
    for path in numbered(directory / 'streams', 'learning'):
        trace = json.loads(path.read_text())
        pixels = path.with_suffix('.i32le').resolve()
        # Reuse the already qualified validator for all pixel ranges, raw tree
        # fields, inclusion rules and learning-option bounds.
        project_learning(trace, pixels.stat().st_size)
        if trace['local_stream_id'] == -1:
            if global_tree is not None:
                raise ValueError('Multiple global learning scopes')
            global_tree = trace['tree']
        for im in trace['images']:
            if not im['channels']:
                continue  # Reserved ID hole.
            stream = trace['local_stream_id'] if trace['local_stream_id'] >= 0 else im['index']
            if stream in images:
                raise ValueError('Repeated captured stream')
            images[stream] = (im, trace['tree'] if trace['local_stream_id'] >= 0 else None,
                              pixels, trace['pixel_bytes'])
    rows = ['gjxl-coding-input-v1']

    def row(tag, values):
        rows.append(tag + ' ' + ' '.join(map(str, values)))

    def tree(value):
        row('tree', [len(value) if value else 0])
        rows.extend(' '.join(map(str, n)) for n in value or [])

    tree(global_tree)
    coding = [json.loads(p.read_text()) for p in numbered(directory / 'streams', 'coding')]
    if not coding or {c['stream_id'] for c in coding} != set(images) or len(coding) != len(images):
        raise ValueError('Learning and coding stream IDs differ')
    row('streams', [len(coding)])
    for c in sorted(coding, key=lambda x: x['stream_id']):
        im, local_tree, pixels, size = images[c['stream_id']]
        if (c['schema'] != 1 or c['local_tree'] is not (local_tree is not None) or
                c['channels'] != len(im['channels']) or
                any(c[k] != im[k] for k in ('width', 'height', 'metadata_channels', 'transforms', 'options'))):
            raise ValueError('Learning and coding stream semantics differ')
        channels = [ch for ch in im['channels'] if ch['included']]
        o = c['options']
        if o['max_properties'] != 0 or o['zero_tokens'] is not False or o['group_dim'] != 256:
            raise ValueError('Unsupported coding options')
        row('stream', [integer(c['stream_id'], 0, 65535), im['width'], im['height'], im['bitdepth'],
                       im['metadata_channels'], len(channels)])
        row('options', [o['max_chan_size'], o['group_dim'], o['predictor'], o['wp_mode'],
                        int(o['skip_encoder_fast_path'])])
        row('wp', [*o['wp_coefficients'], *o['wp_weights']])
        row('transforms', [integer(len(c['transforms']), 0, 8)])
        for t in c['transforms']:
            begin = integer(t['begin_c'], 0, 255)
            if t['id'] == 0:
                row('transform', [2, begin, 3, 1, 1, 1, integer(t['rct_type'], 0, 41)])
            elif t['id'] == 1:
                if t['nb_deltas'] != 0 or t['predictor'] != 0 or t['lossy_palette'] is not False:
                    raise ValueError('Unsupported captured palette')
                row('transform', [0, begin, integer(t['num_c'], 1, 4),
                                   integer(t['nb_colors'], 1, 1024), 1, 1, 0])
            else:
                raise ValueError('Captured transform outside RCT/palette replay cohort')
        tree(local_tree)
        # std::quoted reads escaped quotes/backslashes; JSON escapes other
        # characters differently, so reject those in this private path format.
        path = str(pixels)
        if any(ord(v) < 32 or ord(v) > 126 for v in path):
            raise ValueError('Non-ASCII/control character in replay pixel path')
        row('pixels', [json.dumps(path), size])
        for ch in channels:
            row('channel', [ch[k] for k in ('width', 'height', 'hshift', 'vshift', 'offset', 'bytes')])
    rows.append('end')
    result = '\n'.join(rows) + '\n'
    if len(result) > 16 << 20:
        raise ValueError('Coding description too large')
    return result


def validate_result(directory, result):
    coding = [json.loads(p.read_text()) for p in numbered(directory / 'streams', 'coding')]
    learning = [json.loads(p.read_text()) for p in numbered(directory / 'streams', 'learning')]
    if (result['schema'] != 1 or result['timing_qualified'] is not False or
            any(result[k] is not True for k in ('tokens_exact', 'trees_exact', 'decoded_exact'))):
        raise ValueError('Missing coding equality or invalid timing scope')
    expected = {'captured_streams': len(coding), 'tokens': sum(t['pixel_bytes'] // 4 for t in learning),
                'transforms': sum(len(c['transforms']) for c in coding),
                'global_trees': sum(t['local_stream_id'] == -1 for t in learning),
                'local_trees': sum(t['local_stream_id'] >= 0 and t['pixel_bytes'] > 0 for t in learning),
                'coded_streams': sum(any(ch['bytes'] for ch in im['channels'])
                                     for t in learning for im in t['images'])}
    if any(result[k] != v for k, v in expected.items()) or result['canonical_slots'] < len(coding):
        raise ValueError('Changed captured scope or population')
    if [e['mode'] for e in result['encodes']] != ['prefix', 'ans']:
        raise ValueError('Missing coding entropy mode')
    for e in result['encodes']:
        if not (0 < e['bytes'] <= e['output_bound_bytes'] and
                0 < e['storage_peak_bytes'] <= e['storage_bound_bytes']):
            raise ValueError('Coding resource bound violated')
