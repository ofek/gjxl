#!/usr/bin/env python3
"""Exercise positive and corrupt-capture paths without altering frozen artifacts."""
import json
from pathlib import Path
import sys
import tempfile

from entropy_efficiency import analyze, sha, require


def write(path, value):
    if path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(value, indent=2) + '\n')


def inventory(root):
    write(root / 'sha256-manifest.json', {str(p.relative_to(root)):
        dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name != 'sha256-manifest.json'})


def run(source):
    analyze(source)
    for kind in ('incomplete', 'changed_bytes', 'missing_inventory', 'matrix', 'exact_cost', 'gpu_route', 'capacity'):
        with tempfile.TemporaryDirectory(prefix='gjxl-entropy-audit-') as directory:
            root = Path(directory)
            # Real directories and file symlinks: writes always replace the
            # symlink first, so no negative test can modify the original file.
            for p in source.rglob('*'):
                target = root / p.relative_to(source)
                if p.is_dir(): target.mkdir(exist_ok=True)
                else: target.symlink_to(p)
            m = json.loads((root / 'manifest.json').read_text())
            for record in m['commands']:
                record['argv'][2] = str(root / (record['tag'] + '.jxl'))
            write(root / 'manifest.json', m); inventory(root)
            # The relocated clone is a valid positive control before damage.
            analyze(root)
            if kind == 'incomplete':
                del m['finished_utc']; write(root / 'manifest.json', m)
            elif kind == 'changed_bytes':
                path = root / (m['commands'][0]['tag'] + '.jxl')
                path.unlink(); path.write_bytes(b'corrupt')
            elif kind == 'missing_inventory':
                index = json.loads((root / 'sha256-manifest.json').read_text())
                del index['manifest.json']; write(root / 'sha256-manifest.json', index)
            elif kind == 'matrix':
                m['commands'].pop(); write(root / 'manifest.json', m); inventory(root)
            else:
                row = next(r for r in m['commands'] if r['route'] == 'estimated')
                if kind == 'exact_cost': row['result']['entropy_work']['ans_token_cost_ns'] = 1
                elif kind == 'gpu_route': row['result']['gpu_submissions'] += 1
                else: row['result']['planned_capacity'] = row['result']['peak_backing'] - 1
                write(root / (row['tag'] + '.stdout'), row['result'])
                write(root / 'manifest.json', m); inventory(root)
            failed = False
            try: analyze(root)
            except (AssertionError, RuntimeError, ValueError) as error:
                failed = True; print(kind, 'rejected:', str(error))
            require(failed, f'Auditor accepted {kind}')
    analyze(source)
    print('Original and relocated captures pass; seven damaged captures rejected; original reverified')


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve())
