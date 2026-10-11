#!/usr/bin/env python3
"""Exercise pipeline artifact auditing without executing any encoder."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

from qualify_entropy_pipeline import audit, sha


def clone(source, destination):
    for path in source.rglob('*'):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.symlink_to(path)


def replace(path, data):
    # Never write through a symlink into an immutable original capture.
    if path.is_symlink():
        path.unlink()
    path.write_text(data)


def alter_result(root, manifest, variant, change):
    row = next(r for r in manifest['commands'] if r['variant'] == variant)
    path = root / (row['tag'] + '.stdout')
    result = json.loads(path.read_text())
    change(result)
    replace(path, json.dumps(result) + '\n')
    row['result'] = {k: v for k, v in result.items() if k != 'candidate_ledger'}
    row['files'][path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))


def check_recheck(source, recheck):
    from recheck_entropy_pipeline import audit_recheck
    _, controls, _ = audit(source)
    recheck = recheck.resolve()
    audit_recheck(recheck, controls)
    original = json.loads((recheck / 'manifest.json').read_text())
    cases = {
        'missing-mode': lambda root, m: m['commands'].pop(),
        'wrong-power': lambda root, m: m['commands'][0]['power_before'].update(stdout="Now drawing from 'Battery Power'"),
        'wrong-ledger': lambda root, m: alter_result(root, m, 'emit',
            lambda r: r['candidate_ledger'][0].update(improved=False)),
        'missing-inventory': lambda root, m: m['commands'][0]['files'].pop(m['commands'][0]['tag'] + '.jxl'),
    }
    with tempfile.TemporaryDirectory(prefix='entropy-recheck-audit-', dir=source.parent) as temporary:
        parent = Path(temporary)
        positive = parent / 'positive'; positive.mkdir(); clone(recheck, positive)
        audit_recheck(positive, controls)
        for name, mutation in cases.items():
            root = parent / name; root.mkdir(); clone(recheck, root)
            manifest = copy.deepcopy(original)
            mutation(root, manifest)
            replace(root / 'manifest.json', json.dumps(manifest) + '\n')
            try:
                audit_recheck(root, controls)
            except RuntimeError as error:
                print('recheck', name, 'rejected:', error, flush=True)
            else:
                raise RuntimeError(f'Broken recheck accepted: {name}')
    audit_recheck(recheck, controls)
    print('Original and relocated AC rechecks pass; four damaged rechecks rejected')


def main(source, recheck=None):
    source = source.resolve()
    audit(source)
    original = json.loads((source / 'manifest.json').read_text())
    cases = {
        'incomplete': lambda root, m: m.update(complete=False),
        'missing-command': lambda root, m: m['commands'].pop(),
        'missing-artifact-inventory': lambda root, m: m['commands'][0]['files'].pop(m['commands'][0]['tag'] + '.jxl'),
        'changed-arguments': lambda root, m: m['commands'][0]['argv'].__setitem__(4, '128'),
        'changed-dimensions': lambda root, m: alter_result(root, m, 'emit', lambda r: r.update(width=r['width'] + 1)),
        'wrong-emission-work': lambda root, m: alter_result(root, m, 'measure', lambda r: r.update(candidate_emissions=2)),
        'wrong-population-route': lambda root, m: alter_result(root, m, 'population',
            lambda r: r['adaptive_gpu'].update(population_gpu_calls=0)),
        'wrong-training-work': lambda root, m: alter_result(root, m, 'measure',
            lambda r: r['training'].update(samples=r['training']['samples'] + 1)),
        'wrong-candidate-cost': lambda root, m: alter_result(root, m, 'measure',
            lambda r: r['candidate_ledger'][-1].update(payload_bits=r['candidate_ledger'][-1]['payload_bits'] + 8,
                                                     bytes=r['candidate_ledger'][-1]['bytes'] + 1)),
        'wrong-improvement': lambda root, m: alter_result(root, m, 'emit',
            lambda r: r['candidate_ledger'][0].update(improved=False)),
    }
    with tempfile.TemporaryDirectory(prefix='entropy-audit-', dir=source.parent) as temporary:
        parent = Path(temporary)
        positive = parent / 'positive'; positive.mkdir()
        clone(source, positive)
        audit(positive)
        for name, mutation in cases.items():
            root = parent / name; root.mkdir(); clone(source, root)
            manifest = copy.deepcopy(original)
            mutation(root, manifest)
            replace(root / 'manifest.json', json.dumps(manifest) + '\n')
            try:
                audit(root)
            except RuntimeError as error:
                print(name, 'rejected:', error, flush=True)
            else:
                raise RuntimeError(f'Broken capture accepted: {name}')
    audit(source)
    print('Original and relocated captures pass; 10 damaged captures rejected; original reverified')
    if recheck:
        check_recheck(source, recheck)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--recheck', type=Path)
    args = parser.parse_args()
    main(args.capture, args.recheck)
