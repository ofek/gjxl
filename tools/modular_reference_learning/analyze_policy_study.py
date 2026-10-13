#!/usr/bin/env python3
"""Validate reference policy traces and report a buffering workflow ablation."""
import argparse
from collections import Counter
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

from analyze import load_study, sha
from analyze_native_preparation import suite_passed


def numbered(root, prefix):
    paths = sorted(root.glob(prefix + '-*.json'), key=lambda p: int(p.stem.split('-')[-1]))
    if [p.stem for p in paths] != [f"{prefix}-{i}" for i in range(len(paths))]:
        raise ValueError("Missing or noncontiguous trace events")
    return paths


def validate_options(o):
    for value, bits in (("sampling_fraction", "sampling_fraction_bits"),
                        ("split_threshold", "split_threshold_bits"),
                        ("fast_decode_multiplier", "fast_decode_bits")):
        if not math.isfinite(o[value]) or struct.unpack('<I', struct.pack('<f', o[value]))[0] != o[bits]:
            raise ValueError("Invalid traced float value/bits")
    if not (0 <= o["predictor"] <= 15 and 0 <= o["wp_mode"] <= 4 and
            len(o["wp_coefficients"]) == 7 and len(o["wp_weights"]) == 4):
        raise ValueError("Invalid traced predictor parameters")
    if any(not 0 <= x <= 31 for x in o["wp_coefficients"]) or any(not 0 <= x <= 15 for x in o["wp_weights"]):
        raise ValueError("Invalid traced WP header")


def validate_trace(directory, result):
    root = directory / "streams"
    paths, coding = numbered(root, 'learning'), numbered(root, 'coding')
    learning = result["learning"]
    if (len(paths) != result["traced_learning_calls"] or len(paths) != len(learning["learning"]) or
            len(coding) != result["traced_coding_streams"] or not paths or not coding or
            learning["stream_trace"] is not True or learning["learning_includes_capture_overhead"] is not True):
        raise ValueError("Incomplete stream trace")
    cursor, raw_bytes, included_pixels, local = 0, 0, 0, 0
    signatures, stream_signatures, wp, rcts = Counter(), Counter(), Counter(), Counter()
    palette_count = 0
    for index, path in enumerate(paths):
        trace = json.loads(path.read_text())
        fits = learning["fits"]
        if (trace["schema"] != 1 or trace["event"] != index or trace["pixels_verified"] is not True or
                trace["fit_begin"] != cursor or trace["fit_end"] not in (cursor, cursor + 1) or
                trace["fit_end"] > len(fits)):
            raise ValueError("Trace does not partition learning fits")
        if trace["fit_end"] > cursor and fits[cursor]["nodes"] != len(trace["tree"]):
            raise ValueError("Traced tree differs from fit shape")
        cursor = trace["fit_end"]
        if trace["local_stream_id"] >= 0:
            local += 1
            if (trace["start"], trace["stop"]) != (0, 1):
                raise ValueError("Invalid local learning range")
        if [im["index"] for im in trace["images"]] != list(range(trace["start"], trace["stop"])):
            raise ValueError("Missing traced images or stream-ID holes")
        pixel_path = path.with_suffix('.i32le')
        if pixel_path.stat().st_size != trace["pixel_bytes"] or trace["pixel_bytes"] > 512 << 20:
            raise ValueError("Pixel capture size mismatch")
        raw = pixel_path.read_bytes()
        offset = 0
        for im in trace["images"]:
            o = im["options"]
            validate_options(o)
            if not 0 <= im["metadata_channels"] <= len(im["channels"]):
                raise ValueError("Invalid metadata channel count")
            included = im["width"] > 0 and im["height"] > 0
            begin = offset
            for c, channel in enumerate(im["channels"]):
                w, h = channel["width"], channel["height"]
                if w < 0 or h < 0:
                    raise ValueError("Invalid channel dimensions")
                if c >= im["metadata_channels"] and max(w, h) > o["max_chan_size"]:
                    included = False
                count = w * h * 4 if included else 0
                if (channel["included"] is not included or channel["offset"] != offset or
                        channel["bytes"] != count or offset + count > len(raw)):
                    raise ValueError("Invalid channel pixel slice")
                offset += count
            if offset != begin:
                # Exact pixel bytes, selected learning options, transforms and
                # shape as a multiset. This does not assert spatial ordering or
                # equality of global/local sampling problems.
                options = {k: v for k, v in o.items() if k != 'histogram_params'}
                shape = [{k: v for k, v in c.items() if k != 'offset'} for c in im["channels"]]
                semantics = {"width": im["width"], "height": im["height"], "bitdepth": im["bitdepth"],
                             "metadata_channels": im["metadata_channels"], "channels": shape,
                             "options": options, "transforms": im["transforms"],
                             "pixel_sha256": hashlib.sha256(memoryview(raw)[begin:offset]).hexdigest()}
                signature = hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()
                signatures[signature] += 1
                stream = trace['local_stream_id'] if trace['local_stream_id'] >= 0 else im['index']
                stream_signatures[f'{stream}:{signature}'] += 1
        if offset != len(raw):
            raise ValueError("Unclaimed pixel capture bytes")
        raw_bytes += len(raw)
        included_pixels += len(raw) // 4
    if cursor != len(learning["fits"]):
        raise ValueError("Untraced fits")
    coding_local = 0
    for index, path in enumerate(coding):
        trace = json.loads(path.read_text())
        if trace["schema"] != 1 or trace["event"] != index or type(trace["local_tree"]) is not bool:
            raise ValueError("Invalid coding trace")
        validate_options(trace["options"])
        coding_local += trace["local_tree"]
        wp[trace["options"]["wp_mode"]] += 1
        for transform in trace["transforms"]:
            if transform["id"] == 0:
                if not 0 <= transform["rct_type"] < 42:
                    raise ValueError("Invalid traced RCT")
                rcts[transform["rct_type"]] += 1
            elif transform["id"] == 1:
                palette_count += 1
            elif transform["id"] != 2:
                raise ValueError("Invalid traced transform ID")
    return {"learning_calls": len(paths), "local_learning_calls": local,
            "nonlocal_learning_calls": len(paths) - local, "fits": len(learning["fits"]),
            "fit_samples": sum(f["samples"] for f in learning["fits"]),
            "fit_nodes": sum(f["nodes"] for f in learning["fits"]),
            "coding_streams": len(coding), "local_coding_streams": coding_local,
            "coding_wp_modes": dict(sorted(wp.items())), "coding_rct_types": dict(sorted(rcts.items())),
            "coding_palette_transforms": palette_count, "captured_pixel_bytes": raw_bytes,
            "included_channel_pixels": included_pixels, "learning_image_signatures": dict(sorted(signatures.items())),
            "learning_stream_signatures": dict(sorted(stream_signatures.items()))}


def measurements(result):
    learning = result['learning']
    if (learning['capture'] or learning['stream_trace'] or learning['learning_includes_capture_overhead'] or
            result['traced_learning_calls'] or result['traced_coding_streams']):
        raise ValueError("Trace/capture overhead included in timing control")
    data = {}
    for metric in ('cpu', 'wall'):
        data['encode_' + metric] = result['reference_call_' + metric]
        data['learning_' + metric] = sum(x[metric] for x in learning['learning'])
        data['fit_' + metric] = sum(x[metric] for x in learning['fits'])
        data['non_fit_' + metric] = data['learning_' + metric] - data['fit_' + metric]
        if not data['encode_' + metric] + 2e-6 >= data['learning_' + metric] >= data['fit_' + metric]:
            raise ValueError("Invalid nested workflow timings")
    if not all(math.isfinite(x) and x >= 0 for x in data.values()) or data['encode_cpu'] == 0:
        raise ValueError("Nonfinite or nonpositive workflow timing")
    return data


def analyze(root):
    root = root.resolve()
    manifest_path, ledger = root / 'manifest.json', root / 'results.jsonl'
    manifest = json.loads(manifest_path.read_text())
    if manifest['variants'] != ['auto', 'buffered'] or not 3 <= manifest['repeats'] <= 20 or manifest['warmups'] != 1:
        raise ValueError("Invalid policy study configuration")
    for path, digest in manifest['artifacts'].items():
        if sha(path) != digest:
            raise ValueError(f"Changed policy study artifact: {path}")
    _, corpus, _ = load_study(Path(manifest['corpus']))
    stock = {r['input']: r for r in corpus if r['iteration'] == -1 and r['variant'] == 'reference'}
    inputs = {i['name']: i for i in manifest['inputs']}
    expected = {(name, variant, i) for name in inputs for variant in manifest['variants']
                for i in range(-1, manifest['repeats'] + 1)}
    rows = [json.loads(s) for s in ledger.read_text().splitlines()]
    order = []
    for index, name in enumerate(inputs):
        for iteration in range(-1, manifest['repeats'] + 1):
            variants = manifest['variants'] if (index + iteration) % 2 == 0 else list(reversed(manifest['variants']))
            order.extend((name, variant, iteration) for variant in variants)
    if [(r['input'], r['variant'], r['iteration']) for r in rows] != order:
        raise ValueError("Incomplete or noninterleaved policy study")
    by_case = {}
    previous_finish = 0
    for row in rows:
        key = row['input'], row['variant'], row['iteration']
        if key not in expected or key in by_case:
            raise ValueError("Duplicate or unexpected policy result")
        name, variant, iteration = key
        directory = root / f'{name}-{variant}-{iteration}'
        command = [str(root / 'reference_learning'), 'reference-policy', inputs[name]['path'], str(directory),
                   'trace' if iteration == -1 else 'timing', variant]
        if row['command'] != command or row['warmup'] is not (iteration == 0):
            raise ValueError("Incorrect policy command or warmup")
        if row['started_unix'] < previous_finish or row['finished_unix'] < row['started_unix']:
            raise ValueError("Overlapping policy processes or changed clock")
        previous_finish = row['finished_unix']
        for path, digest in row['artifacts'].items():
            if sha(path) != digest:
                raise ValueError("Changed policy result artifact")
        files = [p for p in directory.rglob('*') if p.is_file()]
        if set(map(str, files)) != row['artifacts'].keys():
            raise ValueError("Unbound or missing policy result file")
        result = row['result']
        if result != json.loads((directory / 'stdout.json').read_text()):
            raise ValueError("Policy ledger disagrees with stdout")
        encoded = directory / 'output.jxl'
        if (result['decoded_exact'] is not True or result['bytes'] != encoded.stat().st_size or
                row['output_sha256'] != sha(encoded) or result['buffering'] != (-1 if variant == 'auto' else 0)):
            raise ValueError("Invalid policy output")
        if variant == 'auto' and row['output_sha256'] != stock[name]['output_sha256']:
            raise ValueError("Automatic control differs from frozen stock e9")
        by_case[key] = row
    if by_case.keys() != expected:
        raise ValueError("Incomplete policy study")
    groups = []
    for name in inputs:
        variants = {}
        for variant in manifest['variants']:
            capture = by_case[name, variant, -1]
            summary = validate_trace(root / f'{name}-{variant}--1', capture['result'])
            fit_shape = lambda r: [{k: v for k, v in f.items() if k not in ('cpu', 'wall')} for f in r['result']['learning']['fits']]
            measures = []
            for i in range(manifest['repeats'] + 1):
                row = by_case[name, variant, i]
                if row['output_sha256'] != capture['output_sha256'] or fit_shape(row) != fit_shape(capture):
                    raise ValueError("Trace/timing output or fit shape changed")
                data = measurements(row['result'])
                if i: measures.append(data)
            variants[variant] = {'bytes': capture['result']['bytes'], 'trace': summary,
                                 'timing': {k: {'median': statistics.median(m[k] for m in measures),
                                               'runs': [m[k] for m in measures]} for k in measures[0]}}
        a, b = variants['auto'], variants['buffered']
        groups.append({'input': name, **variants, 'buffered_byte_delta': b['bytes'] - a['bytes'],
                       'buffered_byte_ratio': b['bytes'] / a['bytes'],
                       'buffered_learning_cpu_ratio': b['timing']['learning_cpu']['median'] / a['timing']['learning_cpu']['median'],
                       'buffered_encode_cpu_ratio': b['timing']['encode_cpu']['median'] / a['timing']['encode_cpu']['median'],
                       'learning_image_semantics_equal_as_multiset': a['trace']['learning_image_signatures'] == b['trace']['learning_image_signatures'],
                       'learning_image_semantics_equal_by_stream_id': a['trace']['learning_stream_signatures'] == b['trace']['learning_stream_signatures']})
    totals = {v: sum(g[v]['bytes'] for g in groups) for v in manifest['variants']}
    return {'schema': 1, 'inputs': len(inputs), 'encodes': len(rows), 'repeats': manifest['repeats'],
            'all_decodes_exact': True, 'all_automatic_bytes_match_stock': True,
            'all_trace_timing_outputs_identical': True, 'total_bytes': totals,
            'buffered_byte_ratio': totals['buffered'] / totals['auto'], 'results': groups,
            'timing_scope': manifest['timing_scope'],
            'limits': ['Workflow ablation: buffering changes tree, transform and entropy scope; not a pure tree-scope attribution.',
                       'All learning/fit times are summed per encode before taking medians; trace and warmup runs excluded.',
                       'Learning-input equality is a multiset of pixel/shape/option/transform signatures, not spatial or sample-set equality.',
                       'Captured pixels are little-endian signed int32 in tight channel rows; excluded channels retain metadata but no pixels.',
                       'Learning traces include actual raw trees and every included pixel checked against disk; coding traces cover global/local stream headers.',
                       'No native search-policy/default change or held-out qualification is implied.'],
            'artifacts': {str(p): sha(p) for p in (manifest_path, ledger)}, 'analyzer_sha256': sha(Path(__file__))}


def validate_builds(root, study):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    for artifact, digest in manifest['artifacts'].items():
        if sha(artifact) != digest:
            raise ValueError('Changed policy validation artifact')
    if (sha(root / 'release/reference_learning') != sha(study / 'reference_learning') or
            manifest['timed_binary_sha256'] != sha(study / 'reference_learning')):
        raise ValueError('Validated binary differs from timed binary')
    if not suite_passed((root / 'asan-tests.log').read_text(), 7):
        raise ValueError('Missing sanitizer suite')
    log = (root / 'release/LastTest.log').read_text()
    if log.count('Test Passed.') != 7 or 'Test Failed.' in log:
        raise ValueError('Missing Release suite')
    config = json.loads((study / 'manifest.json').read_text())
    _, corpus, _ = load_study(Path(config['corpus']))
    stock = {r['input']: r['output_sha256'] for r in corpus if r['variant'] == 'reference' and r['iteration'] == -1}
    for name in ('photo-macan', 'chessboard', 'alpha16-ramp'):
        case = root / ('asan-' + name)
        result = json.loads((case / 'stdout.json').read_text())
        validate_trace(case, result)
        if result['decoded_exact'] is not True or sha(case / 'output.jxl') != stock[name]:
            raise ValueError('Sanitizer encode differs from stock control')
    rejected = json.loads((root / 'trace-rejections.json').read_text())
    if rejected != {'trace_validated': True, 'rejections': 12, 'learning_calls': 1, 'coding_streams': 5}:
        raise ValueError('Missing trace corruption/scope checks')
    if '-fsanitize=address,undefined -fno-sanitize=vptr' not in (root / 'asan/CMakeCache.txt').read_text():
        raise ValueError('Missing policy sanitizer configuration')
    return {'release_asan_suites_passed': True, 'tests_per_suite': 7,
            'corruption_and_timing_scope_rejections': 12,
            'sanitizer_stock_bytes_and_decodes_exact': ['photo-macan', 'chessboard', 'alpha16-ramp'],
            'timed_release_binary_identical': True, 'scope': manifest['scope'],
            'artifacts': {str(path): sha(path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    result = analyze(args.study)
    if args.validation:
        result['build_validation'] = validate_builds(args.validation.resolve(), args.study.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for g in result['results']:
        print(g['input'], g['auto']['bytes'], g['buffered']['bytes'],
              'learning CPU ratio', round(g['buffered_learning_cpu_ratio'], 3),
              'same stream multiset', g['learning_image_semantics_equal_as_multiset'])


if __name__ == '__main__':
    main()
