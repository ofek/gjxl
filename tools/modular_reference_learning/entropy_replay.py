#!/usr/bin/env python3
"""Evidence contract for same-token downstream entropy ablations."""
import json

from analyze_policy_study import numbered

BASE_VARIANTS = ('reference', 'native_tree', 'native_data', 'native', 'no_lz77', 'rle',
                 'data_no_lz77', 'clusters32', 'clusters64', 'no_lz77_clusters32',
                 'hybrid420', 'ans_approx', 'nonstreaming')
PARTITION_VARIANTS = ('reference_map32', 'reference_map32_native_tree')
DIRECT_VARIANTS = ('native_direct_balanced', 'native_direct_high_density',
                   'native_direct_rate', 'native_direct_rate64')
VARIANTS = BASE_VARIANTS + PARTITION_VARIANTS + DIRECT_VARIANTS


def validate_result(directory, result):
    variant = result['variant']
    if (result['schema'] != 1 or variant not in VARIANTS or result['timing_qualified'] is not False or
            any(result[k] is not True for k in ('tokens_exact', 'trees_exact', 'decoded_exact')) or
            result['native_control_exact'] is not (True if variant == 'native' else None)):
        raise ValueError('Missing entropy equality or invalid evidence scope')
    captures = [json.loads(p.read_text()) for p in numbered(directory / 'streams', 'learning')]
    expected = {}
    for c in captures:
        scope = c['local_stream_id']
        if scope != -1 and not c['pixel_bytes']:
            continue  # Common assembly omits ignored empty local payloads.
        leaves = (len(c['tree']) + 1) // 2
        expected['tree', scope] = (5 * leaves + 2 * (len(c['tree']) - leaves), 6)
        expected['data', scope] = (c['pixel_bytes'] // 4, leaves)
    total = sum(c['pixel_bytes'] // 4 for c in captures)
    if result['original_tokens'] != total:
        raise ValueError('Changed entropy input population')
    fields = ('header_bits', 'toc_bits', 'stream_header_bits', 'section_padding_bits',
              'tree_bits', 'data_model_bits', 'data_token_bits')
    if (any(type(result[k]) is not int or result[k] < 0 for k in fields) or
            sum(result[k] for k in fields) != result['bytes'] * 8 or result['bytes'] <= 0):
        raise ValueError('Incomplete entropy bit accounting')
    seen, tree_bits, model_bits, token_bits = set(), 0, 0, 0
    for s in result['scopes']:
        key = s['kind'], s['id']
        if key in seen or key not in expected:
            raise ValueError('Repeated or unexpected entropy scope')
        seen.add(key)
        if (s['original_tokens'], s['contexts']) != expected[key]:
            raise ValueError('Changed entropy scope tree/population')
        tree = s['kind'] == 'tree'
        native = (variant == 'native' or variant in DIRECT_VARIANTS or
                  variant == ('native_tree' if tree else 'native_data') or
                  variant == 'reference_map32_native_tree' or (variant == 'reference_map32' and not tree))
        if (s['implementation'] != ('native' if native else 'reference') or
                type(s['prefix']) is not bool or type(s['lz77']) is not bool):
            raise ValueError('Incorrect entropy implementation label')
        if not (s['model_bits'] > 0 and s['token_bits'] >= 0 and s['coded_tokens'] >= 0 and
                1 <= s['clusters'] == len(s['uint_configs']) <= 256):
            raise ValueError('Invalid entropy model result')
        # Earlier frozen entropy controls predate the partition transplant and
        # did not record maps. New transplants must carry the complete map.
        if 'context_map' in s:
            if (len(s['context_map']) != s['contexts'] + int(s['lz77']) or
                    any(type(c) is not int or not 0 <= c < s['clusters'] for c in s['context_map'])):
                raise ValueError('Invalid entropy context map')
        if native or variant in ('no_lz77', 'no_lz77_clusters32') or (variant == 'data_no_lz77' and not tree):
            if s['lz77'] or s['coded_tokens'] != s['original_tokens']:
                raise ValueError('LZ77 occurred in a no-LZ77 control')
        if not s['lz77'] and s['lz77_lengths']:
            raise ValueError('LZ77 lengths without LZ77')
        if variant.startswith('reference_map32') and not tree:
            if (s.get('reference_partition_exact') is not True or s['clusters'] > 32 or
                    'context_map' not in s):
                raise ValueError('Native ANS changed the reference partition')
        if variant in ('clusters32', 'clusters64', 'no_lz77_clusters32') and not tree:
            if s['clusters'] > (64 if variant == 'clusters64' else 32):
                raise ValueError('Histogram-cap ablation did not enforce cap')
        if variant == 'hybrid420' and not tree and any(c != [4, 2, 0] for c in s['uint_configs']):
            raise ValueError('Fixed HybridUint ablation changed configuration')
        if variant in DIRECT_VARIANTS and not tree:
            cap = 64 if variant == 'native_direct_rate64' else 32
            if s['prefix'] or s['clusters'] > cap or 'context_map' not in s:
                raise ValueError('Direct-ANS control changed coder or exceeded cap')
            if variant == 'native_direct_balanced' and any(c != [4, 2, 0] for c in s['uint_configs']):
                raise ValueError('Balanced direct ANS changed its fixed configuration')
        if tree: tree_bits += s['model_bits'] + s['token_bits']
        else:
            model_bits += s['model_bits']; token_bits += s['token_bits']
    if seen != set(expected) or (tree_bits, model_bits, token_bits) != (
            result['tree_bits'], result['data_model_bits'], result['data_token_bits']):
        raise ValueError('Missing entropy scope or bit attribution')


def validate_controls(results):
    """Compare every transplanted map with the actual reference encoder map.

    A map emitted by the standalone clusterer is not sufficient evidence: the
    full no-LZ77 encoder must independently produce the same partition.
    """
    by_variant = {r['variant']: r for r in results}
    direct = [r for r in results if r['variant'] in DIRECT_VARIANTS]
    if direct:
        if 'native' not in by_variant:
            raise ValueError('Direct ANS requires the native coding control')
        reference_trees = [s for s in by_variant['native']['scopes'] if s['kind'] == 'tree']
        for row in direct:
            if [s for s in row['scopes'] if s['kind'] == 'tree'] != reference_trees:
                raise ValueError('Direct ANS changed tree coding')
    transplants = [r for r in results if r['variant'].startswith('reference_map32')]
    if not transplants:
        return 0
    if 'no_lz77_clusters32' not in by_variant:
        raise ValueError('Partition transplant requires the full reference control')
    reference = {(s['kind'], s['id']): s for s in by_variant['no_lz77_clusters32']['scopes']}
    verified = 0
    for row in transplants:
        for s in row['scopes']:
            control = reference[s['kind'], s['id']]
            if s['kind'] == 'data':
                if (s['clusters'] != control['clusters'] or
                        s.get('context_map') != control.get('context_map') or
                        'context_map' not in control):
                    raise ValueError('Transplant differs from the full reference partition')
                verified += 1
            elif row['variant'] == 'reference_map32' and s != control:
                raise ValueError('Partition transplant changed reference tree coding')
    if len(transplants) == 2:
        data = [[s for s in r['scopes'] if s['kind'] == 'data'] for r in transplants]
        if data[0] != data[1]:
            raise ValueError('Tree-coder swap changed transplanted data coding')
    return verified
