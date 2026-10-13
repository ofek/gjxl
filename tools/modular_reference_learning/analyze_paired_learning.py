#!/usr/bin/env python3
"""Validate matched preparation-plus-fitting evidence without inferring encode speed."""
import argparse
import json
import math
import statistics
from pathlib import Path

from analyze import load_study, sha
from analyze_native_preparation import suite_passed


SHAPE = ("samples", "distinct", "nodes", "predictors", "properties")
STORAGE = ("preparation_storage_bound_bytes", "fitting_storage_bound_bytes")
STAGES = ("native_total", "native_preparation", "native_fit", "reference_total",
          "reference_fit", "reference_non_fit")


def summarize_case(case, row, repeats):
    if (case["input"], case["variant"]) != (row["input"], row["variant"]):
        raise ValueError("Case identity mismatch")
    runs = row["runs"]
    if len(runs) != repeats or not 3 <= repeats <= 20:
        raise ValueError("Expected three to twenty paired repeats")
    for i, run in enumerate(runs):
        if (run["iteration"] != i or run["reference_first"] is not (i % 2 == 0) or
                run["samples_exact_in_warmup"] is not True or run["trees_exact"] is not True):
            raise ValueError("Missing exact checks or alternating order")
        if any(run[k] != case["expected_shape"][k] for k in SHAPE):
            raise ValueError("Shape differs from captured bridge")
        if not (0 < run["distinct"] <= run["samples"] and run["nodes"] > 0 and run["nodes"] % 2):
            raise ValueError("Invalid sample/tree population")
        for key in STORAGE:
            if type(run[key]) is not int or run[key] <= 0 or run[key] != runs[0][key]:
                raise ValueError("Invalid or changing storage bound")
        for stage in STAGES:
            for metric in ("cpu", "wall"):
                value = run[stage + "_" + metric]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Nonpositive or nonfinite timing")
        for metric in ("cpu", "wall"):
            # std::clock has microsecond resolution on the qualified platform.
            if run["native_total_" + metric] + 2e-6 < sum(
                    run[stage + "_" + metric] for stage in ("native_preparation", "native_fit")):
                raise ValueError("Native nested timings exceed total")
            difference = run["reference_total_" + metric] - run["reference_fit_" + metric]
            if difference < 0 or not math.isclose(
                    difference, run["reference_non_fit_" + metric], rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("Incorrect reference non-fit difference")
    result = {"input": case["input"], "variant": case["variant"],
              **{k: runs[0][k] for k in (*SHAPE, *STORAGE)}}
    for stage in (*STAGES, "native_non_fit"):
        for metric in ("cpu", "wall"):
            values = [r["native_total_" + metric] - r["native_fit_" + metric] for r in runs] if stage == "native_non_fit" else [r[stage + "_" + metric] for r in runs]
            result[stage + "_" + metric] = {"median": statistics.median(values), "runs": values}
    for stage in ("total", "fit", "non_fit"):
        for metric in ("cpu", "wall"):
            result[stage + "_" + metric + "_ratio"] = (
                result["native_" + stage + "_" + metric]["median"] /
                result["reference_" + stage + "_" + metric]["median"])
    return result


def analyze(root, corpus_root):
    root, corpus_root = root.resolve(), corpus_root.resolve()
    manifest_path, ledger = root / "manifest.json", root / "results.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["mode"] != "paired-learning" or not 3 <= manifest["repeats"] <= 20 or
            manifest["warmups_per_implementation"] != 1):
        raise ValueError("Paired learning needs one warmup and three to twenty repeats")
    for path, digest in manifest["artifacts"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed paired-learning artifact: {path}")
    corpus_manifest, corpus, corpus_artifacts = load_study(corpus_root)
    inputs = {i["name"]: i for i in corpus_manifest["inputs"]}
    reference = {(r["input"], r["variant"]): r for r in corpus if r["iteration"] == -1}
    expected = {}
    for case in manifest["cases"]:
        key = case["input"], case["variant"]
        if key in expected or key not in reference or key[1] not in ("best", "all"):
            raise ValueError("Duplicate or unknown learning case")
        original = reference[key]
        if (case["path"] != inputs[key[0]]["path"] or
                manifest["artifacts"].get(case["path"]) != inputs[key[0]]["sha256"]):
            raise ValueError("Input differs from frozen corpus")
        for field in ("fixture", "policy", "expected_output"):
            if (case[field] not in original["artifacts"] or
                    original["artifacts"][case[field]] != manifest["artifacts"].get(case[field])):
                raise ValueError("Learning control differs from frozen bridge")
        fits = original["result"]["learning"]["fits"]
        if len(fits) != 1 or case["expected_shape"] != {k: fits[0][k] for k in SHAPE}:
            raise ValueError("Incorrect expected bridge shape")
        expected[key] = case
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if not expected or len(rows) != len(expected):
        raise ValueError("Incomplete paired-learning ledger")
    seen, cases = set(), []
    for row in rows:
        key = row["input"], row["variant"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected learning result")
        seen.add(key)
        case = expected[key]
        command = [str(root / "reference_learning"), "paired-learning", case["path"],
                   case["policy"], case["fixture"], case["variant"], str(manifest["repeats"])]
        if row["command"] != command or str(root / "reference_learning") not in manifest["artifacts"]:
            raise ValueError("Incorrect or unbound learning command")
        for path, digest in row["artifacts"].items():
            if sha(path) != digest:
                raise ValueError("Changed paired-learning result")
        outputs = [p for p in row["artifacts"] if Path(p).name == "stdout.jsonl"]
        if len(outputs) != 1 or [json.loads(s) for s in Path(outputs[0]).read_text().splitlines()] != row["runs"]:
            raise ValueError("Learning ledger disagrees with raw stdout")
        cases.append(summarize_case(case, row, manifest["repeats"]))
    ratios = {}
    for stage in ("total", "fit", "non_fit"):
        for metric in ("cpu", "wall"):
            ratios[stage + "_" + metric] = (
                sum(c["native_" + stage + "_" + metric]["median"] for c in cases) /
                sum(c["reference_" + stage + "_" + metric]["median"] for c in cases))
    return {"schema": 1, "cases": len(cases), "inputs": len({c["input"] for c in cases}),
            "repeats": manifest["repeats"], "all_prepared_samples_exact_in_warmup": True,
            "all_measured_trees_exact": True, "timing_scope": manifest["timing_scope"],
            "ratio_of_summed_case_medians": ratios,
            "fit_cpu_gate_1_5_exceeded": [c for c in cases if c["fit_cpu_ratio"] > 1.5],
            "results": cases,
            "limits": ["Matched global learning on frozen native transforms and default WP; not stock e9 search or whole-encoder speed.",
                       "No transforms, encoding or decoding performed inside either learning timer.",
                       "Non-fit is computed within each run before taking medians; includes sample release and call overhead.",
                       "Reference fit hooks add small timing/event-recording overhead to its total.",
                       "Storage values are admission bounds, not new measured peaks; no accounting scope inside timings.",
                       "Warmups compare every native prepared field against the captured fixture; every measured call checks its tree.",
                       "Release AArch64 NEON, serial same-process pairs on an uncontrolled desktop."],
            "artifacts": {str(manifest_path): sha(manifest_path), str(ledger): sha(ledger), **corpus_artifacts},
            "analyzer_sha256": sha(Path(__file__))}


def validate_builds(root, timed_binary):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, digest in manifest["artifacts"].items():
        if sha(path) != digest:
            raise ValueError("Changed paired-learning validation artifact")
    if (sha(root / "release/reference_learning") != sha(timed_binary) or
            manifest["timed_release_sha256"] != sha(timed_binary)):
        raise ValueError("Validated Release binary differs from timed binary")
    if not suite_passed((root / "asan-tests.log").read_text(), 7):
        raise ValueError("Incomplete sanitizer suite")
    release_log = (root / "release/LastTest.log").read_text()
    if release_log.count("Test Passed.") != 7 or "Test Failed." in release_log:
        raise ValueError("Incomplete Release suite")
    for name in ("photo-macan", "alpha16-ramp"):
        runs = [json.loads(s) for s in (root / f"asan-{name}.jsonl").read_text().splitlines()]
        if (len(runs) != 1 or runs[0]["samples_exact_in_warmup"] is not True or
                runs[0]["trees_exact"] is not True):
            raise ValueError("Missing sanitizer paired sample/tree verification")
    if "-fsanitize=address,undefined -fno-sanitize=vptr" not in (root / "asan/CMakeCache.txt").read_text():
        raise ValueError("Missing sanitizer configuration")
    return {"release_asan_suites_passed": True, "tests_per_suite": 7,
            "sanitizer_paired_controls_exact": ["photo-macan/best", "alpha16-ramp/all"],
            "release_binary_identical_to_timed": True, "scope": manifest["scope"],
            "artifacts": {str(manifest_path): sha(manifest_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    result = analyze(args.study, args.corpus)
    if args.validation:
        result["build_validation"] = validate_builds(
            args.validation.resolve(), args.study.resolve() / "reference_learning")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("results", "artifacts")}, indent=2))


if __name__ == "__main__":
    main()
