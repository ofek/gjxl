#!/usr/bin/env python3
"""Project a saved learning trace into the bounded C++ replay input format."""
import argparse
import json
from pathlib import Path

from analyze_policy_study import validate_options


def integer(value, low=0, high=(1 << 32) - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Invalid replay integer')
    return value


def project(trace, pixel_bytes):
    if trace['schema'] != 1 or trace['pixels_verified'] is not True:
        raise ValueError('Unverified or unknown learning capture')
    start = integer(trace['start'], 0, 65534)
    stop = integer(trace['stop'], start + 1, 65535)
    local = integer(trace['local_stream_id'], -1)
    fits = integer(trace['fit_end'] - trace['fit_begin'], 0, 1)
    if local >= 0 and (start, stop) != (0, 1):
        raise ValueError('Invalid local learning range')
    if integer(trace['pixel_bytes'], 0, 512 << 20) != pixel_bytes:
        raise ValueError('Pixel capture length mismatch')
    if [im['index'] for im in trace['images']] != list(range(start, stop)):
        raise ValueError('Missing image or group-ID hole')
    rows = ['gjxl-learning-input-v1', f'range {start} {stop} {local} {fits} {pixel_bytes}']

    def row(tag, values):
        rows.append(tag + ' ' + ' '.join(str(v) for v in values))

    common, offset, channels = None, 0, 0
    for im in trace['images']:
        w, h = (integer(im[k], 0, 1 << 20) for k in ('width', 'height'))
        count = integer(len(im['channels']), 0, 65535 - channels)
        channels += count
        meta = integer(im['metadata_channels'], 0, count)
        if not (w and h) and (count or w or h):
            raise ValueError('Invalid empty image')
        row('image', [im['index'], w, h, integer(im['bitdepth'], 1, 32), meta, count])
        o = im['options']
        validate_options(o)
        # WP headers vary per image. All other learning settings in this
        # private batch contract must be shared; reject instead of approximating.
        keys = ('max_chan_size', 'sampling_fraction_bits', 'split_threshold_bits',
                'fast_decode_bits', 'max_properties', 'max_property_values',
                'predictor', 'wp_tree_mode', 'tree_kind', 'properties')
        selected = {k: o[k] for k in keys}
        if common is None: common = selected
        if selected != common:
            raise ValueError('Per-image sampling/growth options are unsupported')
        props = o['properties']
        integer(len(props), 1, 16)
        if len(set(props)) != len(props) or o['max_properties'] != 0 or o['tree_kind'] != 1 or o['zero_tokens'] is not False:
            raise ValueError('Unsupported replay learning profile')
        if type(o['skip_encoder_fast_path']) is not bool:
            raise ValueError('Invalid fast-path flag')
        row('options', [integer(o['max_chan_size'], 1), integer(o['group_dim'], 1),
                        integer(o['sampling_fraction_bits']), integer(o['split_threshold_bits']),
                        integer(o['fast_decode_bits']), 0, integer(o['max_property_values'], 2, 256),
                        integer(o['predictor'], 0, 15), integer(o['wp_mode'], 0, 4),
                        integer(o['wp_tree_mode'], 0, 3), 1, int(o['skip_encoder_fast_path']),
                        0, len(props), *(integer(p, 0, 15) for p in props)])
        row('wp', [*(integer(x, 0, 31) for x in o['wp_coefficients']),
                   *(integer(x, 0, 15) for x in o['wp_weights'])])
        included = bool(w and h)
        for c, channel in enumerate(im['channels']):
            cw, ch = (integer(channel[k], 0, 1 << 20) for k in ('width', 'height'))
            if c >= meta and max(cw, ch) > o['max_chan_size']: included = False
            size = cw * ch * 4 if included else 0
            if (channel['included'] is not included or integer(channel['offset']) != offset or
                    integer(channel['bytes']) != size or offset + size > pixel_bytes):
                raise ValueError('Invalid captured channel slice')
            row('channel', [cw, ch, integer(channel['hshift'], -31, 31), integer(channel['vshift'], -31, 31),
                            integer(channel['component'], -1, 65535), int(included), offset, size])
            offset += size
    if offset != pixel_bytes:
        raise ValueError('Unclaimed capture pixels')
    row('multipliers', [integer(len(trace['multipliers']), 0, 65536)])
    for m in trace['multipliers']:
        if len(m['range']) != 2 or any(len(r) != 2 for r in m['range']):
            raise ValueError('Invalid multiplier range')
        rows.append(' '.join(map(str, [integer(m['multiplier'], 1, (1 << 31) - 1),
                                     *(integer(x, 0, 65535) for r in m['range'] for x in r)])))
    tree = trace['tree']
    row('tree', [integer(len(tree), 1, 16383)])
    for n in tree:
        if len(n) != 7: raise ValueError('Invalid tree row')
        bounds = [(-1, 15), (-(1 << 31), (1 << 31) - 1), (0, len(tree) - 1),
                  (0, len(tree) - 1), (0, 13), (-(1 << 31), (1 << 31) - 1), (1, (1 << 31) - 1)]
        rows.append(' '.join(str(integer(x, *b)) for x, b in zip(n, bounds)))
    rows.append('end')
    result = '\n'.join(rows) + '\n'
    if len(result) > 16 << 20: raise ValueError('Replay description too large')
    return result


def validate_result(trace, result, fit=None):
    if (result['schema'] != 1 or any(result[k] is not True for k in
            ('reference_tree_exact', 'samples_exact', 'tree_exact')) or result['timing_qualified'] is not False):
        raise ValueError('Missing replay equality or incorrect timing scope')
    if any(result[k] != trace[k] for k in ('start', 'stop', 'local_stream_id')):
        raise ValueError('Changed replay scope')
    fits = trace['fit_end'] - trace['fit_begin']
    if result['fits'] != fits or result['nodes'] != len(trace['tree']):
        raise ValueError('Changed replay tree or fit count')
    if result['scanned_pixels'] != trace['pixel_bytes'] // 4:
        raise ValueError('Replayed population differs from capture')
    samples, distinct = result['samples'], result['distinct']
    if not (0 <= distinct <= samples <= result['scanned_pixels'] < result['population']) or bool(samples) != bool(fits):
        raise ValueError('Invalid replay sample population')
    if fit is not None and any(result[k] != fit[k] for k in ('samples', 'distinct', 'nodes')):
        raise ValueError('Replayed samples differ from original fitted population')
    for stage in ('preparation', 'fitting'):
        if not 0 < result[f'{stage}_storage_peak_bytes'] <= result[f'{stage}_storage_bound_bytes']:
            raise ValueError('Replay exceeded its managed storage bound')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as out:
        out.write(project(json.loads(args.trace.read_text()), args.trace.with_suffix('.i32le').stat().st_size))


if __name__ == '__main__':
    main()
