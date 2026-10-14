#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Corruption probes for the workflow verifier, using one real frozen capture."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from compare_reference_workflow import compare, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    compare(args.native, args.reference)
    for probe in ("pixels", "tree", "wp", "shape", "scope", "fits", "decode", "duplicate", "extra_learning"):
        with tempfile.TemporaryDirectory(prefix="gjxl-workflow-compare-") as temporary:
            native = Path(temporary) / "native"
            shutil.copytree(args.native, native)
            path = native / "workflow.json"
            data = json.loads(path.read_text())
            result_path = native / "result.json"
            result = json.loads(result_path.read_text())
            active = next(s for s in data["streams"] if any(c["bytes"] for c in s["channels"]))
            learned = next(s for s in data["streams"] if s["tree"])
            if probe == "pixels":
                with (native / "pixels.i32le").open("r+b") as file:
                    value = file.read(1)
                    file.seek(0)
                    file.write(bytes([value[0] ^ 1]))
            elif probe == "tree":
                learned["tree"][0][1] += 1
            elif probe == "wp":
                active["wp_coefficients"][0] ^= 1
            elif probe == "shape":
                active["channels"][0]["width"] += 1
            elif probe == "scope":
                learned["learning"]["start"] += 1
            elif probe == "fits":
                data["stats"]["fits"] += 1
                result["stats"]["fits"] += 1
            elif probe == "decode":
                result["exact_decode"] = False
            elif probe == "duplicate":
                data["streams"].append(data["streams"][0])
            else:
                empty = next(s for s in data["streams"] if not s["channels"])
                empty["learning"]["stop"] = 1
            path.write_text(json.dumps(data))
            result_path.write_text(json.dumps(result))
            try:
                compare(native, args.reference)
            except ValueError as error:
                print(json.dumps({"probe": probe, "rejected": True, "reason": str(error)}))
            else:
                require(False, "Workflow comparison accepted corruption: " + probe)


if __name__ == "__main__":
    main()
