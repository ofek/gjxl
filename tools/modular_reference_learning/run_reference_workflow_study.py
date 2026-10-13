#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Original-input automatic-policy rate control, with Best/all predictors.

Timing fields are diagnostic only. A separate interleaved, otherwise idle run
is required for performance claims. Existing stock policy captures are reused
only after checking their input and output hashes against the corpus manifest.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from compare_reference_workflow import compare, digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--reference-binary", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--policy-captures", required=True, type=Path)
    parser.add_argument("--reuse-run", type=Path,
                        help="Reuse completed captures from the same frozen executables")
    parser.add_argument("--auxiliary-frame-input", action="append", default=[],
                        help="Explicitly diagnosed reference auxiliary-frame input; retain rate, exclude tree parity")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    corpus_path = args.corpus.resolve() / "manifest.json"
    corpus = json.loads(corpus_path.read_text())
    corpus_results_path = args.corpus.resolve() / "results.jsonl"
    reference_rows = [json.loads(line) for line in corpus_results_path.read_text().splitlines()]
    reference_rows = [r for r in reference_rows if r["variant"] == "reference" and r["iteration"] == -1]
    require(len({r["input"] for r in reference_rows}) == len(reference_rows), "Duplicate reference controls")
    reference_controls = {r["input"]: {"encoded_bytes": r["result"]["bytes"],
                            "output": str(args.corpus.resolve() / (r["input"] + "-reference--1/output.jxl")),
                            "output_sha256": r["output_sha256"]} for r in reference_rows}
    require(set(reference_controls) == {im["name"] for im in corpus["inputs"]}, "Incomplete reference controls")
    require(set(args.auxiliary_frame_input) <= set(reference_controls), "Unknown auxiliary-frame exclusion")
    policy_path = args.policy_captures.resolve() / "manifest.json"
    policy = json.loads(policy_path.read_text())
    stock_inputs = {x["name"]: x for x in policy["inputs"]}
    binaries = {}
    for label, source in (("native", args.binary), ("reference", args.reference_binary)):
        target = root / (label + "_workflow")
        shutil.copy2(source, target)
        require(digest(source) == digest(target), "Binary copy differs")
        binaries[label] = {"path": str(target), "sha256": digest(target), "source": str(source.resolve())}
    if args.reuse_run:
        for label in binaries:
            require(digest(args.reuse_run / (label + "_workflow")) == binaries[label]["sha256"],
                    "Reuse executable differs")
    def reuse(directory):
        previous = args.reuse_run / directory.name if args.reuse_run else None
        if previous and (previous / "output.jxl").is_file():
            # The study runs on APFS; cloning avoids duplicating large captures.
            subprocess.run(["/bin/cp", "-cR", str(previous), str(directory)], check=True)
            return True
        return False
    artifacts = {str(corpus_path): digest(corpus_path), str(policy_path): digest(policy_path),
                 str(corpus_results_path): digest(corpus_results_path)}
    rows = []
    for image in corpus["inputs"]:
        name = image["name"]
        input_path = Path(image["path"])
        require(digest(input_path) == image["sha256"], "Changed original input: " + name)
        artifacts[str(input_path)] = image["sha256"]
        control = reference_controls[name]
        if name in corpus["reference_controls"]:
            require(control["output_sha256"] == corpus["reference_controls"][name]["output_sha256"],
                    "Older stock control differs")
        control_path = Path(control["output"])
        require(digest(control_path) == control["output_sha256"], "Changed stock control: " + name)
        artifacts[str(control_path)] = control["output_sha256"]
        old = corpus["native_controls"][name]
        require(digest(Path(old["path"])) == old["sha256"], "Changed legacy native control: " + name)
        artifacts[old["path"]] = old["sha256"]
        if name in stock_inputs:
            require(stock_inputs[name]["sha256"] == image["sha256"], "Mismatched existing capture input")
            reference = args.policy_captures.resolve() / (name + "-auto--1")
        else:
            reference = root / (name + "-reference")
            if not reuse(reference):
                with (root / (name + "-reference.log")).open("x") as log:
                    subprocess.run([binaries["reference"]["path"], "reference-policy", str(input_path),
                                    str(reference), "trace", "auto"], stdout=log, stderr=subprocess.STDOUT, check=True)
        require(digest(reference / "output.jxl") == control["output_sha256"],
                "Traced reference changed stock bytes: " + name)
        for predictors in ("best", "all"):
            native = root / (name + "-" + predictors)
            if not reuse(native):
                with (root / (native.name + ".log")).open("x") as log:
                    subprocess.run([binaries["native"]["path"], str(input_path), str(native), "auto",
                                    predictors, "trace"], stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((native / "result.json").read_text())
            require(result["exact_decode"], "Native decode failed")
            require(result['buffering'] == 'auto' and result['predictors'] == predictors,
                    "Reused native options differ")
            comparison = None
            if predictors == "best":
                if name in args.auxiliary_frame_input:
                    coding = [json.loads(p.read_text()) for p in (reference / "streams").glob("coding-*.json")]
                    globals_ = sorted((s for s in coding if s["stream_id"] == 0), key=lambda s: s["event"])
                    require(len(globals_) == 2 and
                            (globals_[0]["width"], globals_[0]["height"]) != (image["width"], image["height"]),
                            "Declared auxiliary-frame capture does not have the diagnosed structure")
                    comparison = {"decisions_and_trees_exact": False,
                                  "parity_exclusion": "Reference auxiliary patch frame and patch-subtracted main pixels",
                                  "auxiliary_extent": [globals_[0]["width"], globals_[0]["height"]]}
                else:
                    comparison = compare(native, reference)
            row = {"input": name, "predictors": predictors, "native": str(native),
                   "reference": str(reference), "result": result,
                   "stock_bytes": control["encoded_bytes"], "legacy_native_bytes": old["bytes"],
                   "comparison": comparison}
            with (root / "results.jsonl").open("a") as ledger:
                ledger.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
            print(f"{name} {predictors}: {result['bytes']} / stock {control['encoded_bytes']} bytes; exact decode",
                  flush=True)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            artifacts[str(path)] = digest(path)
    manifest = {"schema": 1, "kind": "native-reference-workflow-corpus", "binaries": binaries,
                "corpus": str(corpus_path), "policy_captures": str(policy_path),
                "inputs": corpus["inputs"], "variants": ["best", "all"], "buffering": "auto",
                "timing_qualified": False, "reuse_run": str(args.reuse_run.resolve()) if args.reuse_run else None,
                "auxiliary_frame_inputs": args.auxiliary_frame_input, "artifacts": artifacts}
    with (root / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
