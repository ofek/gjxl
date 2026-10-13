#!/usr/bin/env python3
"""Verify native preparation evidence; no timing or default-promotion inference."""
import argparse
import json
import re
from pathlib import Path

from analyze import load_study, sha


def validate_case(case, row):
    if (case["input"], case["variant"]) != (row["input"], row["variant"]):
        raise ValueError("Case identity mismatch")
    result = row["result"]
    if not all(result[k] is True for k in ("samples_exact", "tree_exact", "decoded_exact")):
        raise ValueError("Missing exact sample/tree/decode verification")
    if result["bytes"] != case["expected_bytes"] or row["output_sha256"] != case["expected_sha256"]:
        raise ValueError("Native preparation changed bytes")
    if not 0 < result["distinct"] <= result["samples"] <= result["scanned_pixels"] < result["population"]:
        raise ValueError("Invalid preparation populations")
    if not 0 < result["storage_peak_bytes"] <= result["storage_bound_bytes"]:
        raise ValueError("Unqualified preparation storage bound")


def analyze(root, corpus_root):
    manifest_path, ledger = root / "manifest.json", root / "results.jsonl"
    manifest = json.loads(manifest_path.read_text())
    _, corpus, corpus_artifacts = load_study(corpus_root)
    for path, digest in manifest["artifacts"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed preparation artifact: {path}")
    reference = {(r["input"], r["variant"]): r for r in corpus if r["iteration"] == -1}
    expected = {}
    for case in manifest["cases"]:
        key = case["input"], case["variant"]
        if key in expected or key not in reference or key[1] not in ("best", "all"):
            raise ValueError("Duplicate or unknown preparation case")
        original = reference[key]
        if case["expected_sha256"] != original["output_sha256"] or case["expected_bytes"] != original["result"]["bytes"]:
            raise ValueError("Preparation expectation differs from frozen bridge")
        for field in ("fixture", "policy", "expected_output"):
            if original["artifacts"].get(case[field]) != manifest["artifacts"].get(case[field]):
                raise ValueError("Preparation source not bound to bridge artifact")
        expected[key] = case
    if not expected:
        raise ValueError("Empty preparation study")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    seen = set()
    for row in rows:
        key = row["input"], row["variant"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected preparation result")
        seen.add(key)
        for path, digest in row["artifacts"].items():
            if sha(path) != digest:
                raise ValueError("Changed preparation result artifact")
        outputs = [p for p in row["artifacts"] if Path(p).name == "output.jxl"]
        stdouts = [p for p in row["artifacts"] if Path(p).name == "stdout.json"]
        if len(outputs) != 1 or sha(outputs[0]) != row["output_sha256"] or Path(outputs[0]).stat().st_size != row["result"]["bytes"]:
            raise ValueError("Missing encoded-output evidence")
        if len(stdouts) != 1 or json.loads(Path(stdouts[0]).read_text()) != row["result"]:
            raise ValueError("Result disagrees with runner stdout")
        validate_case(expected[key], row)
    if seen != expected.keys():
        raise ValueError("Incomplete preparation ledger")
    return {"schema": 1, "cases": len(rows), "inputs": len({r["input"] for r in rows}),
            "all_sample_fields_exact": True, "all_trees_exact": True,
            "all_bridge_bytes_exact": True, "all_independent_decodes_exact": True,
            "all_preparation_storage_bounds_passed": True,
            "maximum_preparation_peak_bytes": max(r["result"]["storage_peak_bytes"] for r in rows),
            "total_bytes": {v: sum(r["result"]["bytes"] for r in rows if r["variant"] == v) for v in ("best", "all")},
            "scope": "Native quantization, gathering and fitting on frozen winning native transforms, reference samples and estimated-ANS output. Correctness/storage qualification, not a speed claim, search-policy comparison or default promotion.",
            "results": [{"input": r["input"], "variant": r["variant"], **r["result"]} for r in rows],
            "artifacts": {str(manifest_path): sha(manifest_path), str(ledger): sha(ledger), **corpus_artifacts}}


def suite_passed(log, count):
    return re.search(rf"^100% tests passed(?:, 0 tests failed)? out of {count}$", log, re.MULTILINE) is not None


def validate_builds(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, digest in manifest["artifacts"].items():
        if sha(path) != digest:
            raise ValueError("Changed preparation validation artifact")
    for before, after in manifest["native_source_pairs"]:
        if sha(before) != sha(after):
            raise ValueError("Final native preparation source differs from qualified source")
    for name in ("final-release-tests.log", "final-scalar-tests.log", "final-asan-tests.log"):
        if not suite_passed((root / name).read_text(), 6):
            raise ValueError("Missing complete preparation validation suite")
    if not suite_passed((root / "root-tests.log").read_text(), 4):
        raise ValueError("Missing root Modular validation")
    for name in ("asan-photo-macan.json", "asan-alpha16.json", "scalar-photo-macan.json", "scalar-alpha16.json"):
        data = json.loads((root / name).read_text())
        if data["samples_exact"] is not True or data["tree_exact"] is not True:
            raise ValueError("Missing scalar/sanitizer preparation parity")
    for name in ("release-preparation-tests.json", "asan-preparation-tests.json", "scalar-preparation-tests.json"):
        data = json.loads((root / name).read_text())
        if data != {"preparation_differential_cases": 101, "samples_exact": True, "tree_exact": True,
                    "reference_weight_overflow_rejected": True}:
            raise ValueError("Missing preparation differential or overflow regression checks")
    if "-fsanitize=address,undefined -fno-sanitize=vptr" not in (root / "asan/CMakeCache.txt").read_text():
        raise ValueError("Missing sanitizer configuration")
    return {"differential_cases": 101, "reference_weight_overflow_rejected": True,
            "scope": "Native preparation/fitting, harness and generated reference units instrumented; linked archives uninstrumented; vptr excluded for pinned no-RTTI interoperability. UBSAN_OPTIONS=halt_on_error=1. Scalar and sanitizer samples/trees verified on photographic and alpha16 inputs.",
            "artifacts": {str(manifest_path): sha(manifest_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    result = analyze(args.study.resolve(), args.corpus.resolve())
    if args.validation:
        result["build_validation"] = validate_builds(args.validation.resolve())
    result["analyzer_sha256"] = sha(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("results", "artifacts")}, indent=2))


if __name__ == "__main__":
    main()
