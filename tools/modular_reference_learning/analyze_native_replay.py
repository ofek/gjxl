#!/usr/bin/env python3
"""Validate a complete paired native/reference replay ledger and report all fits."""
import argparse
import json
import statistics
from pathlib import Path

from analyze import sha


def analyze(root):
    root = root.resolve()
    manifest_file, ledger = root / "manifest.json", root / "results.jsonl"
    manifest = json.loads(manifest_file.read_text())
    if manifest["mode"] != "paired" or manifest["repeats"] < 3:
        raise ValueError("Qualification requires at least three paired repeats")
    for path, digest in manifest["artifacts"].items():
        if sha(path) != digest:
            raise ValueError(f"Manifest artifact changed: {path}")
    fixtures = {f["path"]: f for f in manifest["fixtures"]}
    rows = [json.loads(s) for s in ledger.read_text().splitlines()]
    if len(fixtures) != len(manifest["fixtures"]) or len(rows) != len(fixtures):
        raise ValueError("Incomplete or duplicate paired ledger")
    cases, seen = [], set()
    for row in rows:
        path = row["path"]
        if path in seen or path not in fixtures:
            raise ValueError("Duplicate or unknown paired fixture")
        seen.add(path)
        fixture = fixtures[path]
        if any(row[k] != v for k, v in fixture.items()) or sha(path) != fixture["sha256"]:
            raise ValueError("Changed paired input")
        for path, digest in row["artifacts"].items():
            if sha(path) != digest:
                raise ValueError(f"Replay result artifact changed: {path}")
        runs = row["runs"]
        outputs = [Path(p) for p in row["artifacts"] if p.endswith(".jsonl")]
        if len(outputs) != 1 or [json.loads(s) for s in outputs[0].read_text().splitlines()] != runs:
            raise ValueError("Replay ledger differs from raw stdout")
        expected_command = [str(root / "reference_learning"), "paired-replay",
                            fixture["path"], str(manifest["repeats"])]
        if row["command"] != expected_command:
            raise ValueError("Unexpected replay command")
        if len(runs) != manifest["repeats"] or not all(r["tree_exact"] for r in runs):
            raise ValueError("Incomplete or inexact paired replay")
        fixed = ("samples", "distinct", "nodes", "properties", "predictors",
                 "sample_backing_bytes", "scratch_bound_bytes", "scratch_peak_bytes")
        for i, run in enumerate(runs):
            if run["iteration"] != i or run["reference_first"] != (i % 2 == 0):
                raise ValueError("Replay order or repetition mismatch")
            if any(run[k] != runs[0][k] for k in fixed):
                raise ValueError("Replay shape or storage is not deterministic")
            if run["scratch_peak_bytes"] > run["scratch_bound_bytes"]:
                raise ValueError("Native fit exceeded its storage admission")
        case = {k: fixture[k] for k in ("input", "variant", "fit")}
        case.update({k: runs[0][k] for k in fixed})
        for metric in ("cpu", "wall", "reference_cpu", "reference_wall"):
            values = [r[metric] for r in runs]
            if not all(v > 0 for v in values):
                raise ValueError("Nonpositive timing")
            case[metric] = {"median": statistics.median(values), "runs": values}
        for metric in ("cpu", "wall"):
            case[metric + "_ratio"] = case[metric]["median"] / case["reference_" + metric]["median"]
        cases.append(case)
    groups = []
    for key in dict.fromkeys((c["input"], c["variant"]) for c in cases):
        fits = [c for c in cases if (c["input"], c["variant"]) == key]
        group = {"input": key[0], "variant": key[1], "fits": len(fits)}
        for metric in ("cpu", "wall"):
            group[metric] = sum(c[metric]["median"] for c in fits)
            group["reference_" + metric] = sum(c["reference_" + metric]["median"] for c in fits)
            group[metric + "_ratio"] = group[metric] / group["reference_" + metric]
        groups.append(group)
    ratios = [c["cpu_ratio"] for c in cases]
    return {
        "schema": 1, "timing_scope": manifest["timing_scope"],
        "fits": len(cases), "groups": len(groups), "repeats": manifest["repeats"],
        "all_trees_exact": True, "all_storage_within_bound": True,
        "cpu_ratio_of_summed_case_medians": sum(c["cpu"]["median"] for c in cases) /
                                            sum(c["reference_cpu"]["median"] for c in cases),
        "cpu_ratio_min": min(ratios), "cpu_ratio_median": statistics.median(ratios),
        "cpu_ratio_max": max(ratios),
        "cpu_gate_1_5_exceeded": [c for c in cases if c["cpu_ratio"] > 1.5],
        "image_variants": groups, "cases": cases,
        "ledgers": {str(p): sha(p) for p in (manifest_file, ledger)},
        "analyzer_sha256": sha(Path(__file__).resolve()),
        "limits": ["Pure fitting only; no native sampling or automatic selection promotion.",
                   "Quantized sample copies/conversion are outside both fit timers.",
                   "Native validation and fitting allocations are inside its timer.",
                   "Managed peak is measured in a separate native warmup; excludes caller sample columns, old output and allocator overhead.",
                   "Warmups and measured fits use the same pinned expected tree on every call.",
                   "Scalar portability checks are separate; paired timings use AArch64 NEON on an uncontrolled desktop."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path,
                        help="Optional final-build sanitizer/scalar/contract ledger")
    args = parser.parse_args()
    result = analyze(args.study)
    if args.validation:
        root = args.validation.resolve()
        manifest = root / "manifest.json"
        validation = json.loads(manifest.read_text())
        for path, digest in validation["artifacts"].items():
            if sha(path) != digest:
                raise ValueError(f"Validation artifact changed: {path}")
        if (not validation["same_release_native_object"] or
                sha(root / "paired-native-fitter.o") != sha(root / "final-native-fitter.o")):
            raise ValueError("Final native fitter differs from timed machine code")
        for name in ("root", "release", "scalar", "asan"):
            summaries = {"100% tests passed out of 3",
                         "100% tests passed, 0 tests failed out of 3"}
            if not summaries.intersection((root / (name + "-tests.log")).read_text().splitlines()):
                raise ValueError("Missing final validation suite completion")
        for case in validation["cases"]:
            if sha(case["command"][-1]) != case["fixture_sha256"] or not case["result"]["tree_exact"]:
                raise ValueError("Changed or inexact validation replay")
        result["validation"] = {"manifest": str(manifest), "sha256": sha(manifest),
                                "same_release_native_object": True,
                                "representative_replays": len(validation["cases"]),
                                "root_release_scalar_asan_suites_passed": True}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("fits", "groups", "all_trees_exact", "cpu_ratio_of_summed_case_medians", "cpu_ratio_min", "cpu_ratio_max")}, indent=2))


if __name__ == "__main__":
    main()
