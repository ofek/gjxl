#!/usr/bin/env python3
"""Bind the validation-scan attribution to paired timing and full-tree replay."""
import argparse
import json
import statistics
from pathlib import Path
from analyze import sha
from analyze_native_replay import analyze
from analyze_native_preparation import suite_passed


def manifest(root):
    data = json.loads((root / "manifest.json").read_text())
    for path, digest in data["artifacts"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed artifact: {path}")
    return data


def exact_replays(root, baseline):
    data = manifest(root)
    if data["mode"] != "verify" or data["repeats"] != 1:
        raise ValueError("Expected one correctness replay per fixture")
    fixtures = {f["path"]: f for f in data["fixtures"]}
    if len(fixtures) != len(data["fixtures"]):
        raise ValueError("Duplicate declared correctness fixture")
    rows = [json.loads(s) for s in (root / "results.jsonl").read_text().splitlines()]
    seen = set()
    for row in rows:
        key = row["input"], row["variant"], row["fit"]
        fixture = fixtures[row["path"]]
        if key in seen or any(row[k] != v for k, v in fixture.items()) or sha(row["path"]) != row["sha256"]:
            raise ValueError("Changed or duplicate correctness fixture")
        seen.add(key)
        for path, digest in row["artifacts"].items():
            if sha(path) != digest:
                raise ValueError("Changed correctness output")
        outputs = [p for p in row["artifacts"] if p.endswith(".jsonl")]
        if len(outputs) != 1 or [json.loads(s) for s in Path(outputs[0]).read_text().splitlines()] != row["runs"]:
            raise ValueError("Correctness ledger disagrees with stdout")
        if row["command"] != [str(root / "reference_learning"), "native-verify", fixture["path"]]:
            raise ValueError("Wrong correctness command")
        if len(row["runs"]) != 1 or row["runs"][0]["tree_exact"] is not True:
            raise ValueError("Missing exact native replay")
        if row["runs"][0]["scratch_peak_bytes"] > row["runs"][0]["scratch_bound_bytes"]:
            raise ValueError("Replay exceeded its storage bound")
    expected = {(c["input"], c["variant"], c["fit"]) for c in baseline["cases"]}
    if seen != expected or len(rows) != len(fixtures):
        raise ValueError("Incomplete full-corpus replay")
    return {"fits": len(rows), "all_trees_exact": True, "all_storage_within_bound": True,
            "artifacts": {str(root / name): sha(root / name) for name in ("manifest.json", "results.jsonl")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("paired", "verify", "scan", "validation", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        setattr(args, name, getattr(args, name).resolve())
    baseline = json.loads(args.baseline.read_text())
    paired = analyze(args.paired)
    full = exact_replays(args.verify, baseline)
    manifest(args.scan)
    records = json.loads((args.scan / "repeated.json").read_text())
    if {(r["iteration"], r["implementation"]) for r in records} != {(i, n) for i in range(5) for n in ("before", "after")} or len(records) != 10:
        raise ValueError("Incomplete validation attribution")
    stages = {}
    for name in ("before", "after"):
        runs = [r["result"] for r in records if r["implementation"] == name]
        if not all(r["validated"] is True and r["repeats"] == 1000 and r["cpu"] > 0 for r in runs):
            raise ValueError("Missing validation check")
        fits = [json.loads(s) for s in (args.scan / f"{name}-paired.jsonl").read_text().splitlines()]
        if len(fits) != 9 or not all(r["tree_exact"] is True for r in fits):
            raise ValueError("Missing paired attribution checks")
        stages[name] = {"validation_cpu_seconds_per_call": statistics.median(r["cpu"] / r["repeats"] for r in runs),
                        "native_fit_cpu_seconds": statistics.median(r["cpu"] for r in fits),
                        "reference_fit_cpu_seconds": statistics.median(r["reference_cpu"] for r in fits)}
    manifest(args.validation)
    for name in ("release", "scalar", "asan", "root"):
        if not suite_passed((args.validation / f"{name}-tests.log").read_text(), 2 if name == "root" else 6):
            raise ValueError("Missing validation suite")
    result = {"schema": 1, "paired": paired, "full_replay": full, "isolated_stages": stages,
              "scope": "Validation-only execution change. Repeated timings cover 64 stock-reference spline fits, including the previous 19 outliers. Full 233-fit replay supplies correctness and storage evidence; no new repeated full-corpus timing or complete-learning speedup claim.",
              "artifacts": {str(p): sha(p) for p in (args.scan / "manifest.json", args.validation / "manifest.json", args.baseline)},
              "analyzer_sha256": sha(Path(__file__))}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"paired_fits": paired["fits"], "maximum_cpu_ratio": paired["cpu_ratio_max"],
                      "outliers": len(paired["cpu_gate_1_5_exceeded"]), "exact_replays": full["fits"],
                      "isolated_stages": stages}, indent=2))


if __name__ == "__main__":
    main()
