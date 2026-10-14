#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Recheck the frozen native selection/learning/encoding workflow qualification."""
import argparse
import json
import re
from pathlib import Path

from analyze_reference_palette import PIN, TESTS
from compare_reference_workflow import compare, digest, require


def qualify(root, baseline):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prior_path = baseline / "manifest.json"
    prior = json.loads(prior_path.read_text())
    require(manifest["kind"] == "native-reference-workflow" and manifest["reference_revision"] == PIN and
            manifest["source_snapshot_is_authoritative"] and prior["kind"] == "native-reference-selection",
            "Wrong workflow qualification protocol")
    for path, expected in manifest["artifacts"].items():
        require(digest(Path(path)) == expected, "Changed frozen artifact: " + path)

    def bound(path):
        require(str(path) in manifest["artifacts"], "Unbound workflow artifact: " + str(path))
        return path

    def compared(native, reference):
        result = compare(native, reference)
        for path, expected in result["artifacts"].items():
            require(manifest["artifacts"].get(path) == expected, "Unbound comparison evidence: " + path)
        return result

    tests = TESTS | {"modular_reference_channel_palette", "modular_reference_channel_palette_oracle",
                     "modular_reference_selection", "modular_reference_selection_oracle",
                     "modular_reference_workflow", "modular_histogram_learning"}
    generated = {"selection": 216, "channel-palette": 64, "palette": 172, "generated": 648,
                 "transforms": 906, "ordered": 350, "workflow": 98}
    for mode in ("release", "asan"):
        build = root / mode
        log = bound(build / "LastTest.log").read_text()
        summary = bound(build / "ctest.log").read_text()
        require(set(re.findall(r"^\d+/\d+ Testing: (\S+)$", log, re.M)) == tests and
                log.count("Test Passed.") == len(tests) and
                re.search(r"^100% tests passed(?:, 0 tests failed)? out of 19$", summary, re.M),
                "Incomplete workflow CTest controls")
        require('{"files":98,"allocation_failures":5138,"exact_decode":true}' in log,
                "Missing workflow decode/fault sweep")
        commands = json.loads(bound(build / "compile_commands.json").read_text())
        for source in ("/src/codestream/modular/reference_workflow.cpp", "/tests/modular_reference_workflow_test.cpp",
                       "/tools/modular_reference_learning/reference_workflow_check.cpp"):
            selected = [c["command"] for c in commands if c["file"].endswith(source)]
            require(selected and all(("-fsanitize=address,undefined" in c) == (mode == "asan") for c in selected),
                    "Workflow instrumentation differs")
        for folder, count in generated.items():
            files = sorted((build / folder).glob("*.jxl"))
            require(len(files) == count, "Missing generated workflow/control files")
            for file in files:
                bound(file)
                require(digest(file) == digest(bound(root / "release" / folder / file.name)),
                        "Release and sanitizer codestreams differ")
                if folder != "workflow":
                    previous = baseline / mode / folder / file.name
                    require(digest(previous) == prior["artifacts"].get(str(previous)) and
                            digest(file) == digest(previous), "Prior complete codestream changed")
    corpus = root / "corpus"
    study = json.loads(bound(corpus / "manifest.json").read_text())
    names = [im["name"] for im in study["inputs"]]
    require(len(names) == len(set(names)) == 16 and study["auxiliary_frame_inputs"] == ["gray-patches"],
            "Unexpected corpus or parity exclusion")
    rows = [json.loads(line) for line in bound(corpus / "results.jsonl").read_text().splitlines()]
    require({(r["input"], r["predictors"]) for r in rows} ==
            {(name, variant) for name in names for variant in ("best", "all")} and len(rows) == 32,
            "Incomplete Best/all controls")
    parity = []
    for row in rows:
        name, variant = row["input"], row["predictors"]
        native = corpus / (name + "-" + variant)
        result = json.loads(bound(native / "result.json").read_text())
        require(result == row["result"] and result["exact_decode"] and result["predictors"] == variant and
                result["buffering"] == "auto" and bound(native / "output.jxl").stat().st_size == result["bytes"],
                "Native corpus output differs from its successful encode record")
        control = root / "controls" / name
        require(bound(control / "stock-e9.jxl").stat().st_size == row["stock_bytes"] and
                bound(control / "legacy-native.jxl").stat().st_size == row["legacy_native_bytes"],
                "Rate control bytes differ")
        input_info = next(im for im in study["inputs"] if im["name"] == name)
        require(digest(bound(control / "input.pam")) == input_info["sha256"], "Corpus input differs")
        if variant == "best" and name != "gray-patches":
            reference = Path(manifest["reference_captures"][name + "-auto"])
            require(digest(bound(reference / "output.jxl")) == digest(control / "stock-e9.jxl"),
                    "Policy tracing changed reference output")
            comparison = compared(native, reference)
            for key in ("learning_calls", "fits", "tree_nodes", "compared_samples", "coding_streams"):
                require(comparison[key] == row["comparison"][key], "Recomputed parity counters differ")
            parity.append({k: v for k, v in comparison.items() if k != "artifacts"})
    for name in manifest["buffered_inputs"]:
        native = root / "original" / (name + "-buffered")
        reference = Path(manifest["reference_captures"][name + "-buffered"])
        comparison = compared(native, reference)
        parity.append({k: v for k, v in comparison.items() if k != "artifacts"})
    for key, reference_path in manifest["sanitizer_cases"].items():
        native = root / "asan-original" / key
        reference = Path(reference_path)
        compared(native, reference)
        name, buffering = key.rsplit("-", 1)
        release = corpus / (name + "-best") if buffering == "auto" else root / "original" / key
        for filename in ("output.jxl", "workflow.json", "pixels.i32le"):
            require(digest(bound(native / filename)) == digest(bound(release / filename)),
                    "Sanitizer original-input decisions or bytes differ")
    for key, captured in manifest["captured_decision_controls"].items():
        require(digest(bound(root / "original" / key / "output.jxl")) == digest(bound(Path(captured))),
                "Integrated file differs from the prior captured-decision encode")
    rates = {}
    for variant in ("best", "all"):
        selected = [r for r in rows if r["predictors"] == variant]
        native = sum(r["result"]["bytes"] for r in selected)
        stock = sum(r["stock_bytes"] for r in selected)
        old = sum(r["legacy_native_bytes"] for r in selected)
        rates[variant] = {"native_bytes": native, "stock_e9_bytes": stock, "legacy_native_bytes": old,
                          "gap_vs_e9_pct": 100 * (native / stock - 1),
                          "gap_vs_legacy_native_pct": 100 * (native / old - 1),
                          "maximum_managed_backing_bytes": max(r["result"]["peak_backing_bytes"] for r in selected),
                          "maximum_managed_committed_bytes": max(r["result"]["peak_committed_bytes"] for r in selected),
                          "images": [{"name": r["input"], "native_bytes": r["result"]["bytes"],
                            "stock_bytes": r["stock_bytes"], "legacy_native_bytes": r["legacy_native_bytes"],
                            "gap_vs_e9_pct": 100 * (r["result"]["bytes"] / r["stock_bytes"] - 1),
                            "gap_vs_legacy_native_pct": 100 * (r["result"]["bytes"] / r["legacy_native_bytes"] - 1),
                            "stats": r["result"]["stats"]} for r in selected]}
    return {"schema": 1, "reference_revision": PIN,
            "scope": "Private original-input selection, one-fit-per-scope learning, and direct estimated ANS64 encoding",
            "qualification": {"ctest_entries_per_build": 19, "generated_workflow_files_per_build": 98,
                "injected_allocation_failures_per_build": 5138, "prior_complete_files_unchanged": 2356,
                "release_sanitizer_identical_generated_files": 2454,
                "original_input_rate_encodes": 37, "sanitized_original_input_cases": 3,
                "parity_cases": len(parity), "parity_inputs_auto": 15, "parity_inputs_buffered": 5,
                "parity_totals": {k: sum(r[k] for r in parity) for k in
                    ("learning_calls", "fits", "tree_nodes", "compared_samples", "compared_channels", "coding_streams")},
                "parity_exclusions": {"gray-patches": "Stock e9 auxiliary patch frame and patch-subtracted main pixels"},
                "timing_qualified": False, "held_out_qualified": False, "exhaustive_new_learner_search_qualified": False,
                "default_promotion_qualified": False, "sanitizer_limits": manifest["sanitizer_limits"]},
            "rates": rates, "parity": parity, "validation_manifest": str(manifest_path),
            "validation_manifest_sha256": digest(manifest_path), "bound_validation_files": len(manifest["artifacts"]),
            "baseline_validation_manifest": str(prior_path), "baseline_validation_manifest_sha256": digest(prior_path),
            "analyzer_sha256": digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation", "baseline-validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.validation.resolve(), args.baseline_validation.resolve())
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
