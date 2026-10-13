#!/usr/bin/env python3
"""Compare stock e9 automatic buffering with forced buffering; capture actual streams."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from analyze import load_study, sha
from analyze_policy_study import validate_trace
from study_provenance import freeze_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 20:
        parser.error("repeats must be in [3,20]")
    args.binary, args.corpus, args.output = (p.resolve() for p in (args.binary, args.corpus, args.output))
    corpus, rows, artifacts = load_study(args.corpus)
    requested = set(args.inputs)
    inputs = [i for i in corpus["inputs"] if i["name"] in requested]
    if len(inputs) != len(requested):
        parser.error("Unknown input")
    controls = {r["input"]: r for r in rows if r["variant"] == "reference" and r["iteration"] == -1}
    args.output.mkdir(parents=True, exist_ok=False)
    binary, build = freeze_build(args.binary, args.output)
    build["artifacts"].update(artifacts)
    for i in inputs:
        if sha(i["path"]) != i["sha256"]:
            raise ValueError("Changed input")
        build["artifacts"][i["path"]] = i["sha256"]
    manifest = {"schema": 1, **build, "inputs": inputs, "variants": ["auto", "buffered"],
                "repeats": args.repeats, "warmups": 1, "corpus": str(args.corpus),
                "automatic_controls": {i["name"]: controls[i["name"]]["output_sha256"] for i in inputs},
                "timing_scope": "Serial fresh-process e9 encodes, one discarded warmup per variant/input, then interleaved variants. Encoder API CPU/wall includes all preparation, learning and entropy; input unpack, decode and file IO outside. Per-encode learning CPU/wall sums include all LearnTree events. Trace runs include pixel/metadata IO and are excluded from all timing conclusions. No thread pool; uncontrolled desktop."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    trace_sha = {}
    with (args.output / "results.jsonl").open("x", buffering=1) as ledger:
        for index, item in enumerate(inputs):
            for iteration in range(-1, args.repeats + 1):
                variants = manifest["variants"]
                if (index + iteration) % 2:
                    variants = list(reversed(variants))
                for variant in variants:
                    directory = args.output / f"{item['name']}-{variant}-{iteration}"
                    directory.mkdir()
                    command = [str(binary), "reference-policy", item["path"], str(directory),
                               "trace" if iteration == -1 else "timing", variant]
                    start = time.time()
                    with (directory / "stdout.json").open("x") as out, (directory / "stderr.log").open("x") as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True)
                    result = json.loads((directory / "stdout.json").read_text())
                    encoded = directory / "output.jxl"
                    digest = sha(encoded)
                    if result["decoded_exact"] is not True or encoded.stat().st_size != result["bytes"]:
                        raise ValueError("Incorrect reference output")
                    if variant == "auto" and digest != manifest["automatic_controls"][item["name"]]:
                        raise ValueError("Stock e9 bytes changed with hooks")
                    if iteration == -1:
                        validate_trace(directory, result)
                        trace_sha[item["name"], variant] = digest
                    elif digest != trace_sha[item["name"], variant]:
                        raise ValueError("Trace/timing output changed")
                    row = {"input": item["name"], "variant": variant, "iteration": iteration,
                           "warmup": iteration == 0, "command": command, "output_sha256": digest,
                           "started_unix": start, "finished_unix": time.time(), "result": result,
                           "artifacts": {str(p): sha(p) for p in directory.rglob("*") if p.is_file()}}
                    ledger.write(json.dumps(row) + "\n")
                    print(item["name"], variant, iteration, result["bytes"],
                          "fits", len(result["learning"]["fits"]), flush=True)


if __name__ == "__main__":
    main()
