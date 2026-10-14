#!/usr/bin/env python3
"""Small reproducible 14-model decode/immutable-archive qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from study import CELLS, sha


def run(argv, ok=True):
    p = subprocess.run([str(x) for x in argv], capture_output=True, text=True)
    assert (p.returncode == 0) == ok, (argv, p.returncode, p.stdout, p.stderr)
    return p


def test(binary):
    binary = binary.resolve()
    with tempfile.TemporaryDirectory(prefix='gjxl-entropy-replay-') as temporary:
        root = Path(temporary)
        image = root / 'tiny.pgm'
        image.write_bytes(b'P5\n3 2\n255\n' + bytes([0, 255, 0, 1, 17, 17]))
        for split in (False, True):
            directory = root / ('split' if split else 'single')
            directory.mkdir()
            policy = directory / 'policy.txt'
            policy.write_text(f'POLICY1 0 0 {3 if split else 1}\n16 10 7 7 7 0 0\n13 12 12 12\n' +
                ('0 0 1 2 0 0 1\n-1 0 0 0 0 0 1\n-1 0 0 0 5 0 1\n' if split else '-1 0 0 0 5 0 1\n'))
            archive = directory / 'tokens.bin'
            run([binary, 'freeze', image, policy, archive])
            frozen = sha(archive)
            result = run([binary, 'replay', image, policy, archive, directory / 'outputs'])
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            assert [r['cell'] for r in rows] == CELLS and all(r['decoded_exact'] for r in rows)
            assert sha(archive) == frozen
            original = archive.read_bytes()
            for name, data in [('truncated', original[:-1]), ('extra', original + b'x'),
                               ('bad-magic', b'BADTOK00' + original[8:]),
                               ('bad-count', original[:16] + bytes([255]) * 8 + original[24:])]:
                bad = directory / (name + '.bin'); bad.write_bytes(data)
                run([binary, 'replay', image, policy, bad, directory / (name + '-outputs')], ok=False)
            run([binary, 'freeze', image, policy, archive], ok=False)
            assert sha(archive) == frozen
            print('split' if split else 'single', '14 model decodes; four malformed archives and overwrite rejected')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('binary', type=Path)
    test(p.parse_args().binary)
