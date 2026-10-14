#!/usr/bin/env python3
"""Exercise trace corruption and timing-scope rejection on a real captured encode."""
import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

from analyze_policy_study import measurements, validate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.case / 'stdout.json').read_text())
    good = validate_trace(args.case, result)
    rejected = 0
    with tempfile.TemporaryDirectory(prefix='gjxl-policy-trace-') as temporary:
        root = Path(temporary)
        shutil.copytree(args.case / 'streams', root / 'streams')
        meta = root / 'streams/learning-0.json'
        trace = json.loads(meta.read_text())
        populated = next(i for i, im in enumerate(trace['images']) if any(c['bytes'] for c in im['channels']))
        channel = next(i for i, c in enumerate(trace['images'][populated]['channels']) if c['bytes'])
        bad = []
        def altered(mutate):
            value = copy.deepcopy(trace); mutate(value); bad.append(value)
        altered(lambda t: t.update(pixels_verified=False))
        altered(lambda t: t.update(fit_begin=1))
        altered(lambda t: t.update(stop=t['stop'] + 1))
        altered(lambda t: t.update(pixel_bytes=t['pixel_bytes'] + 4))
        altered(lambda t: t['images'][populated]['channels'][channel].update(offset=1))
        altered(lambda t: t['images'][populated]['channels'][channel].update(included=False))
        altered(lambda t: t['images'][populated]['options'].update(sampling_fraction_bits=0))
        altered(lambda t: t['images'][populated]['options'].update(wp_coefficients=[32] * 7))
        altered(lambda t: t['tree'].append(t['tree'][-1]))
        for value in bad:
            meta.write_text(json.dumps(value))
            try: validate_trace(root, result)
            except ValueError: rejected += 1
            else: raise AssertionError('Corrupt trace accepted')
        meta.write_text(json.dumps(trace))
        pixels = meta.with_suffix('.i32le')
        with pixels.open('r+b') as stream: stream.truncate(pixels.stat().st_size - 1)
        try: validate_trace(root, result)
        except ValueError: rejected += 1
        else: raise AssertionError('Truncated pixel data accepted')
    try: measurements(result)
    except ValueError: rejected += 1
    else: raise AssertionError('Trace used as timing control')
    timed = copy.deepcopy(result)
    timed.update(traced_learning_calls=0, traced_coding_streams=0)
    timed['learning'].update(stream_trace=False, learning_includes_capture_overhead=False)
    measurements(timed)
    timed['reference_call_cpu'] = 0
    try: measurements(timed)
    except ValueError: rejected += 1
    else: raise AssertionError('Invalid nested timing accepted')
    print(json.dumps({'trace_validated': True, 'rejections': rejected,
                      'learning_calls': good['learning_calls'], 'coding_streams': good['coding_streams']}))


if __name__ == '__main__':
    main()
