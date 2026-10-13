#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Exercise the research replay on six formats and global/local tree scopes."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from compare_reference_workflow import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=args.out))
    cases = [("gray8", 17, 13, 1, 8, "auto", "best"),
             ("palette8", 64, 32, 3, 8, "auto", "best"),
             ("rgb16", 29, 17, 3, 16, "auto", "all"),
             ("rgba8", 257, 9, 4, 8, "auto", "best"),
             ("gray16-local", 2049, 1, 1, 16, "auto", "best"),
             ("gray16-shared", 2049, 1, 1, 16, "buffered", "best"),
             ("rgba16", 33, 19, 4, 16, "auto", "best")]
    results = []
    for name, width, height, channels, bits, buffering, predictors in cases:
        header = (f"P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH {channels}\nMAXVAL {(1 << bits) - 1}\n"
                  f"TUPLTYPE { {1: 'GRAYSCALE', 3: 'RGB', 4: 'RGB_ALPHA'}[channels]}\nENDHDR\n")
        pixels = bytearray()
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    value = (x * 4571 + y * 1301 + c * 17383 + x * y * 13) % (1 << bits)
                    if name == "palette8":
                        value = ((x // 8 + y // 8) % 5) * (19 + c * 7)
                    if c == 3 and x % 3 == 0:
                        value = 0  # Invisible RGB must also survive.
                    pixels.extend(value.to_bytes(bits // 8, "big"))
        source = root / (name + ".pam")
        source.write_bytes(header.encode() + pixels)
        output = root / name
        command = [str(args.binary), str(source), str(output), buffering, predictors]
        subprocess.run(command, check=True)
        result = json.loads((output / "result.json").read_text())
        require(result["exact_decode"] and result["selected_replay_exact"], "Replay failed")
        require(result["local_trees"] == (name == "gray16-local"), "Unexpected tree scope")
        require(result["buffering"] == buffering and result["predictors"] == predictors, "Options changed")
        require([c["mode"] for c in result["candidates"]] == list(range(-1, 5)), "Incomplete WP domain")
        require(digest(output / "baseline.jxl") == digest(output / "selected-replay.jxl"), "Replay bytes changed")
        for row in result["candidates"]:
            require(row["exact_decode"], "Candidate decode failed")
            require((output / (row["name"] + ".jxl")).stat().st_size == row["bytes"], "Wrong byte count")
            expected = sum(s["eligible"] and s["selected_wp"] != row["mode"]
                           for s in result["streams"]) if row["mode"] >= 0 else 0
            require(row["changed_streams"] == expected, "Incorrect WP assignment")
            for key in ("learning_calls", "fits", "empty_learning_calls", "scanned_pixels", "selected_samples"):
                require(row["stats"][key] == result["baseline"]["stats"][key], "Learning scope changed")
        results.append({"name": name, "path": str(output), "sha256": digest(output / "result.json")})
    # Repeated invocations must preserve all seven complete files.
    source = root / "rgba16.pam"
    repeat = root / "rgba16-repeat"
    subprocess.run([str(args.binary), str(source), str(repeat), "auto", "best"], check=True)
    for path in (root / "rgba16").glob("*.jxl"):
        require(digest(path) == digest(repeat / path.name), "Nondeterministic output")
    refused = subprocess.run([str(args.binary), str(source), str(repeat), "auto", "best"],
                             capture_output=True, text=True)
    require(refused.returncode != 0 and "already exists" in refused.stderr, "Existing evidence overwritten")
    (root / "tests.json").write_text(json.dumps({"cases": results, "files": 56,
        "exact_decodes": 56, "deterministic_repeat": True, "overwrite_rejected": True}, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
