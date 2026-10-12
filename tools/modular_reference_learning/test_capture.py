#!/usr/bin/env python3
"""Exercise persistence, exact replay, and rejection using a captured fixture."""
import argparse
import struct
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    fixture = args.fixture.resolve()
    subprocess.run([binary, "self-test"], check=True)
    original = fixture.read_bytes()
    with tempfile.TemporaryDirectory(prefix="gjxl-learning-replay-") as temporary:
        root = Path(temporary)
        rewritten = root / "roundtrip.bin"
        subprocess.run([binary, "rewrite", str(fixture), str(rewritten)], check=True)
        if rewritten.read_bytes() != original:
            raise AssertionError("Fixture roundtrip lost fields")
        subprocess.run([binary, "replay", str(rewritten), "1"], check=True)
        failures = {"empty": b"", "truncated": original[:-1], "trailing": original + b"x"}

        def altered(offset, data):
            result = bytearray(original)
            result[offset:offset + len(data)] = data
            return result

        failures["magic"] = altered(0, b"BAD!")
        failures["version"] = altered(8, struct.pack("<I", 2))
        failures["nan-threshold"] = altered(12, struct.pack("<I", 0x7FC00000))
        failures["invalid-preference"] = altered(16, struct.pack("<f", 0))
        failures["empty-static-range"] = altered(24, original[20:24])
        failures["oversized-multiplier-count"] = altered(36, struct.pack("<Q", 1 << 30))
        # The last uint32 is a leaf multiplier in every valid captured tree.
        failures["zero-leaf-multiplier"] = altered(len(original) - 4, b"\0" * 4)
        for name, data in failures.items():
            malformed = root / (name + ".bin")
            malformed.write_bytes(data)
            result = subprocess.run([binary, "rewrite", str(malformed), str(root / (name + "-out.bin"))],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 1:
                raise AssertionError(f"{name} did not fail cleanly: {result.returncode}, {result.stderr}")
        # Change a valid expected predictor: parsing succeeds, but differential
        # replay must detect the false oracle rather than accept a valid shape.
        old_predictor = struct.unpack_from("<I", original, len(original) - 16)[0]
        wrong_tree = root / "wrong-tree.bin"
        wrong_tree.write_bytes(altered(len(original) - 16, struct.pack("<I", (old_predictor + 1) % 14)))
        result = subprocess.run([binary, "replay", str(wrong_tree), "1"], capture_output=True, text=True, timeout=60)
        if result.returncode != 1 or "tree mismatch" not in result.stderr:
            raise AssertionError(f"False expected tree was not detected: {result.stderr}")
    print(f"Fixture persistence/replay passed; {len(failures)} malformed fixtures and one false tree rejected.")


if __name__ == "__main__":
    main()
