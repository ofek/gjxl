#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Verify frozen complete-file WP controls; report lower bounds on search loss."""
import argparse
import json
from pathlib import Path

from compare_reference_workflow import digest, require


def verify_artifacts(root, manifest):
    inventory = {str(path) for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    require(inventory == set(manifest["artifacts"]), "Artifact inventory differs")
    for path, expected in manifest["artifacts"].items():
        require(digest(Path(path)) == expected, "Changed artifact: " + path)


def validation(root, study):
    root = root.resolve()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    require(manifest["schema"] == 1 and manifest["kind"] == "uniform-wp-validation", "Wrong validation")
    require(manifest["study_manifest_sha256"] == study["manifest_sha256"], "Validation covers another study")
    verify_artifacts(root, manifest)
    require(digest(Path(manifest["release_binary"])) ==
            digest(Path(study["manifest"]).parent / "reference_wp_sweep"), "Final study executable differs")
    require(len(manifest["file_pairs"]) == 56, "Incomplete generated-file matrix")
    require(len({p["release"] for p in manifest["file_pairs"]}) == 56 and
            len({p["sanitizer"] for p in manifest["file_pairs"]}) == 56, "Duplicate file pair")
    for pair in manifest["file_pairs"]:
        require(digest(Path(pair["release"])) == digest(Path(pair["sanitizer"])) == pair["sha256"],
                "Release/sanitizer bytes differ")
    for label in ("release", "sanitizer"):
        require("modular_reference_wp_sweep" in Path(manifest["test_logs"][label]).read_text() and
                "Test Passed." in Path(manifest["test_logs"][label]).read_text(), "Missing CTest pass")
    probes = json.loads(Path(manifest["probes"]).read_text())
    require(probes["all_rejected"] and probes["restored_clone_verified"] and
            probes["source_manifest_sha256"] == study["manifest_sha256"] and len(probes["probes"]) == 8 and
            len({p["probe"] for p in probes["probes"]}) == 8, "Incomplete analyzer rejection probes")
    require(digest(Path(__file__)) == digest(root / "source/analyze_reference_wp_sweep.py"),
            "Validation analyzer differs from frozen source")
    return {"manifest": str(path), "manifest_sha256": digest(path),
            "verified_artifacts": len(manifest["artifacts"]), "cross_build_exact_files": 56,
            "final_release_binary_identical_to_study": True,
            "analyzer_rejected_corruptions": 8, "sanitizer_limits": manifest["sanitizer_limits"]}


def analyze(root):
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == 1 and manifest["kind"] == "uniform-wp-ablation", "Wrong study")
    require(manifest["buffering"] == "auto" and manifest["predictors"] == "best" and
            manifest["modes"] == list(range(-1, 5)), "Wrong candidate domain")
    require(not manifest["timing_qualified"] and not manifest["exhaustive_search_qualified"], "Overstated scope")
    verify_artifacts(root, manifest)
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    names = [im["name"] for im in manifest["inputs"]]
    require([row["input"] for row in rows] == names and len(set(names)) == len(names), "Incomplete input set")
    cases = []
    for image, row in zip(manifest["inputs"], rows):
        name = image["name"]
        directory, controls = root / name, root / "controls" / name
        require(digest(controls / "input.pam") == image["sha256"], "Changed input")
        require(row["command"] == [manifest["binary"], str(controls / "input.pam"), str(directory), "auto", "best"],
                "Unexpected command")
        result = json.loads((directory / "result.json").read_text())
        require(row["result"] == result and result["schema"] == 1 and result["kind"] == manifest["kind"],
                "Result/ledger mismatch")
        require(result["buffering"] == "auto" and result["predictors"] == "best" and
                result["exact_decode"] and result["selected_replay_exact"], "Invalid replay")
        streams = result["streams"]
        require(len({s["id"] for s in streams}) == len(streams), "Duplicate stream IDs")
        for stream in streams:
            require(0 <= stream["selected_wp"] < 5, "Invalid WP preset")
            require(len(stream["wp_costs"]) == (5 if stream["eligible"] else 0), "Wrong estimator domain")
            if stream["eligible"]:
                require(stream["selected_wp"] == min(range(5), key=lambda i: stream["wp_costs"][i]),
                        "Reference estimator selection changed")
        groups = ((image["width"] + 255) // 256) * ((image["height"] + 255) // 256)
        require(result["local_trees"] == (groups > 8), "Changed learning scope")
        baseline = result["baseline"]["bytes"]
        require((directory / "baseline.jxl").stat().st_size == baseline and
                digest(directory / "baseline.jxl") == digest(controls / "qualified-best.jxl"), "Wrong baseline")
        require(digest(directory / "selected-replay.jxl") == digest(directory / "baseline.jxl"), "Replay changed bytes")
        candidates = result["candidates"]
        require([c["mode"] for c in candidates] == manifest["modes"], "Missing/duplicate candidates")
        require(candidates[0]["stats"] == result["baseline"]["stats"], "Replay changed work")
        for candidate in candidates:
            mode = candidate["mode"]
            require(candidate["name"] == ("selected-replay" if mode == -1 else f"wp-{mode}"), "Wrong candidate label")
            require(candidate["exact_decode"], "Unverified candidate")
            require((directory / (candidate["name"] + ".jxl")).stat().st_size == candidate["bytes"], "Wrong candidate bytes")
            changed = sum(s["eligible"] and s["selected_wp"] != mode for s in streams) if mode >= 0 else 0
            require(candidate["changed_streams"] == changed, "Incorrect WP override")
            for key in ("learning_calls", "fits", "empty_learning_calls", "scanned_pixels", "selected_samples"):
                require(candidate["stats"][key] == result["baseline"]["stats"][key], "Learning coverage changed")
            if not changed:
                require(digest(directory / (candidate["name"] + ".jxl")) == digest(directory / "baseline.jxl"),
                        "Unchanged assignment changed bytes")
        winner = min(candidates, key=lambda c: c["bytes"])
        cases.append({"input": name, "baseline_bytes": baseline, "minimum_tested_bytes": winner["bytes"],
            "winner": winner["name"], "delta_bytes": baseline - winner["bytes"],
            "loss_vs_minimum_percent": 100 * (baseline / winner["bytes"] - 1),
            "per_image_search_gate_already_fails": 100 * (baseline / winner["bytes"] - 1) > 1,
            "stock_e9_bytes": (controls / "stock-e9.jxl").stat().st_size,
            "legacy_native_bytes": (controls / "legacy-native.jxl").stat().st_size,
            "local_trees": result["local_trees"], "eligible_streams": sum(s["eligible"] for s in streams),
            "baseline_stats": result["baseline"]["stats"], "candidates": candidates,
            "peak_backing_bytes": result["peak_backing_bytes"], "peak_committed_bytes": result["peak_committed_bytes"]})
    baseline = sum(row["baseline_bytes"] for row in cases)
    minimum = sum(row["minimum_tested_bytes"] for row in cases)
    stock = sum(row["stock_e9_bytes"] for row in cases)
    loss = 100 * (baseline / minimum - 1)
    return {"schema": 1, "kind": "uniform-wp-ablation-analysis", "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path), "verified_artifacts": len(manifest["artifacts"]),
        "source_revision": manifest["source_revision"], "scope": manifest["scope"],
        "timing_qualified": False, "exhaustive_search_qualified": False,
        "interpretation": "The minimum tested complete file is an upper bound on the exhaustive optimum; "
                          "the reported selected-policy rate loss is therefore a lower bound. Exceeding a "
                          "search-loss gate falsifies it for any exhaustive domain containing these assignments. "
                          "Passing this subset cannot qualify the exhaustive-search gate.",
        "aggregate": {"inputs": len(cases), "exact_files": 7 * len(cases), "baseline_bytes": baseline,
            "minimum_tested_bytes": minimum, "delta_bytes": baseline - minimum,
            "loss_vs_minimum_percent": loss, "aggregate_search_gate_already_fails": loss > 0.2,
            "per_image_search_gate_failures": [row["input"] for row in cases if row["per_image_search_gate_already_fails"]],
            "stock_e9_bytes": stock, "minimum_vs_e9_percent": 100 * (minimum / stock - 1),
            "baseline_fits": sum(row["baseline_stats"]["fits"] for row in cases),
            "candidate_fits": sum(c["stats"]["fits"] for row in cases for c in row["candidates"])},
        "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.study)
    if args.validation:
        result["validation"] = validation(args.validation, result)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
