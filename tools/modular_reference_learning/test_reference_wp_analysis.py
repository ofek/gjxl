#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Corrupt cloned study artifacts to exercise hash and semantic rejection."""
import argparse
import json
import subprocess
from pathlib import Path

from analyze_reference_wp_sweep import analyze
from compare_reference_workflow import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, root = args.study.resolve(), args.out.resolve()
    require(not root.exists(), "Probe output already exists")
    root.mkdir(parents=True)
    clone = root / "study"
    subprocess.run(["/bin/cp", "-cR", str(source), str(clone)], check=True)
    manifest_path = clone / "manifest.json"
    manifest = json.loads(manifest_path.read_text().replace(str(source), str(clone)))
    ledger_path = clone / "results.jsonl"
    ledger_path.write_text(ledger_path.read_text().replace(str(source), str(clone)))
    # The cloned commands now point at cloned files. All remaining artifacts
    # retain their content; bind the changed ledger before semantic probes.
    manifest["artifacts"] = {path: digest(Path(path)) for path in manifest["artifacts"]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    analyze(clone)
    pristine_manifest = manifest_path.read_bytes()
    name = manifest["inputs"][0]["name"]
    result_path = clone / name / "result.json"
    output_path = clone / name / "wp-0.jxl"
    pristine = {path: path.read_bytes() for path in (result_path, ledger_path, output_path)}
    rejected = []

    def restore():
        for path, content in pristine.items():
            path.write_bytes(content)
        manifest_path.write_bytes(pristine_manifest)

    def rebind():
        current = json.loads(manifest_path.read_text())
        for path in pristine:
            current["artifacts"][str(path)] = digest(path)
        manifest_path.write_text(json.dumps(current, indent=2) + "\n")

    def result_change(change):
        result = json.loads(result_path.read_text())
        change(result)
        result_path.write_text(json.dumps(result) + "\n")
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        rows[0]["result"] = result
        ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        rebind()

    def reject(label, expected):
        try:
            analyze(clone)
        except ValueError as error:
            require(expected in str(error), "Wrong rejection: " + str(error))
            rejected.append({"probe": label, "reason": str(error)})
        else:
            raise ValueError("Accepted corruption: " + label)
        restore()

    output_path.write_bytes(pristine[output_path][:-1])
    reject("unbound output mutation", "Changed artifact")
    output_path.write_bytes(pristine[output_path][:-1])
    rebind()
    reject("rebound incorrect byte count", "Wrong candidate bytes")
    result_change(lambda r: r.__setitem__("exact_decode", False))
    reject("unverified decode", "Invalid replay")
    result_change(lambda r: r["candidates"][1].__setitem__("mode", 1))
    reject("duplicate/missing WP candidate", "Missing/duplicate candidates")
    result_change(lambda r: r["candidates"][1].__setitem__("changed_streams", 9999))
    reject("incorrect WP assignment", "Incorrect WP override")
    result_change(lambda r: r["candidates"][1]["stats"].__setitem__("fits", 9999))
    reject("changed learning scope", "Learning coverage changed")
    result_change(lambda r: r.__setitem__("local_trees", not r["local_trees"]))
    reject("wrong global/local policy", "Changed learning scope")
    incomplete = json.loads(manifest_path.read_text())
    del incomplete["artifacts"][str(output_path)]
    manifest_path.write_text(json.dumps(incomplete) + "\n")
    reject("unbound artifact", "Artifact inventory differs")
    analyze(clone)
    report = {"source_manifest": str(source / "manifest.json"),
              "source_manifest_sha256": digest(source / "manifest.json"),
              "all_rejected": True, "probes": rejected,
              "restored_clone_verified": True}
    (root / "probes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
