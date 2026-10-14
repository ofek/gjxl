#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Validate and summarize the counterbalanced integrated workflow timing."""
import argparse
import json
import math
import statistics
from pathlib import Path

from compare_reference_workflow import digest, require


def distribution(values):
    require(len(values) == 3 and all(math.isfinite(x) and x >= 0 for x in values), "Invalid timing distribution")
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "runs": values}


def analyze(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["kind"] == "native-reference-workflow-timing" and manifest["repeats"] == 3 and
            manifest["warmups"] == 1 and manifest["variants"] == ["reference", "best", "all"], "Unexpected timing protocol")
    require(digest(Path(manifest["validation_manifest"])) == manifest["validation_manifest_sha256"],
            "Changed integration qualification")
    for path, expected in manifest["artifacts"].items():
        require(digest(Path(path)) == expected, "Changed timing artifact: " + path)
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    names = [image["name"] for image in manifest["inputs"]]
    expected = []
    for index, name in enumerate(names):
        for iteration in range(4):
            shift = (index + iteration) % 3
            variants = manifest["variants"][shift:] + manifest["variants"][:shift]
            expected += [(name, v, iteration) for v in variants]
    require([(r["input"], r["variant"], r["iteration"]) for r in rows] == expected and len(rows) == 192,
            "Incomplete or non-counterbalanced timing ledger")
    require(all(rows[i]["finished_unix"] <= rows[i + 1]["started_unix"] for i in range(len(rows) - 1)),
            "Timing encodes overlapped")
    validation = Path(manifest["validation_manifest"]).parent
    qualified = json.loads(Path(manifest["validation_manifest"]).read_text())
    lookup = {(r["input"], r["variant"], r["iteration"]): r for r in rows}
    for row in rows:
        name, variant, iteration = row["input"], row["variant"], row["iteration"]
        require(row["warmup"] == (iteration == 0), "Warmup classification differs")
        directory = root / f"{name}-{variant}-{iteration}"
        control = validation / (f"controls/{name}/stock-e9.jxl" if variant == "reference" else
                                f"corpus/{name}-{variant}/output.jxl")
        require(digest(control) == qualified["artifacts"][str(control)] == row["output_sha256"] ==
                digest(directory / "output.jxl"), "Timed bytes differ from the qualified output")
        result = row["result"]
        if variant == "reference":
            require(result["decoded_exact"] and not result["learning"]["capture"] and
                    not result["learning"]["stream_trace"] and
                    not result["learning"]["learning_includes_capture_overhead"], "Reference trace polluted timing")
        else:
            require(result["schema"] == 2 and result["exact_decode"] and result["total_includes_prepared_teardown"] and
                    not (directory / "workflow.json").exists(), "Invalid native timing boundary")
            require(result == json.loads((directory / "result.json").read_text()), "Native timing result changed")
            source = validation / f"corpus/{name}-{variant}/result.json"
            require(result["stats"] == json.loads(source.read_text())["stats"], "Timed native work changed")
            if name != "gray-patches" and variant == "best":
                reference = lookup[name, "reference", iteration]["result"]["learning"]
                fits, calls = reference["fits"], reference["learning"]
                stats = result["stats"]
                require((stats["learning_calls"], stats["fits"], stats["selected_samples"],
                         stats["distinct_samples"], stats["tree_nodes"]) ==
                        (len(calls), len(fits), sum(f["samples"] for f in fits),
                         sum(f["distinct"] for f in fits), sum(f["nodes"] for f in fits) + len(calls) - len(fits)),
                        "Matched Best/reference learning work differs")
            for kind in ("cpu", "wall"):
                values = result["profile_" + kind]
                require(len(values) == 7 and min(values) >= -1e-6 and
                        sum(values) <= result["total"][kind] + 1e-4, "Stage accounting exceeds total")
    images = []
    for name in names:
        image = {"name": name, "best_reference_learning_problem_matched": name != "gray-patches"}
        for variant in manifest["variants"]:
            results = [lookup[name, variant, i]["result"] for i in (1, 2, 3)]
            item = {"bytes": results[0]["bytes"]}
            for kind in ("cpu", "wall"):
                if variant == "reference":
                    item["total_" + kind] = distribution([r["reference_call_" + kind] for r in results])
                    item["learning_" + kind] = distribution([sum(t[kind] for t in r["learning"]["learning"]) for r in results])
                    item["fit_" + kind] = distribution([sum(t[kind] for t in r["learning"]["fits"]) for r in results])
                else:
                    item["total_" + kind] = distribution([r["total"][kind] for r in results])
                    for index, stage in enumerate(manifest["stages"]):
                        item[("learning" if stage == "training" else stage) + "_" + kind] = distribution(
                            [max(0, r["profile_" + kind][index]) for r in results])
                    item["preparation_" + kind] = distribution([r["prepare"][kind] for r in results])
            image[variant] = item
        for variant in ("best", "all"):
            image[variant]["total_cpu_ratio_vs_reference"] = image[variant]["total_cpu"]["median"] / image["reference"]["total_cpu"]["median"]
            image[variant]["learning_cpu_ratio_vs_reference"] = image[variant]["learning_cpu"]["median"] / image["reference"]["learning_cpu"]["median"]
        images.append(image)
    groups = {}
    for group in ("all_16", "matched_15"):
        subset = [im for im in images if group == "all_16" or im["best_reference_learning_problem_matched"]]
        sums = {}
        for variant in manifest["variants"]:
            fields = [key for key, value in subset[0][variant].items() if isinstance(value, dict)]
            sums[variant] = {key: sum(im[variant][key]["median"] for im in subset) for key in fields}
            sums[variant]["learning_share_of_total_cpu_pct"] = 100 * sums[variant]["learning_cpu"] / sums[variant]["total_cpu"]
        ratios = {variant: {key + "_vs_reference": sums[variant][key] / sums["reference"][key]
                             for key in ("total_cpu", "total_wall", "learning_cpu", "learning_wall")}
                  for variant in ("best", "all")}
        ratios["all_vs_best"] = {key: sums["all"][key] / sums["best"][key]
                                 for key in ("total_cpu", "total_wall", "learning_cpu", "learning_wall")}
        groups[group] = {"inputs": len(subset), "sum_of_case_medians": sums, "ratios": ratios}
    return {"schema": 1, "scope": manifest["scope"], "source_revision": manifest["source_revision"],
            "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
            "analyzer_sha256": digest(Path(__file__)), "timed_encodes": len(rows), "warmups": 48,
            "measured_encodes": 144, "all_timed_files_match_qualified_bytes": True,
            "matched_work_counter_checks": 60, "groups": groups, "images": images,
            "limits": ["Three repetitions on one uncontrolled desktop; no statistical confidence interval",
                       "Native whole interval covers prepare, coding and teardown; stock uses the encoder API wrapper",
                       "Reference patch input has an additional frame and different learning problem",
                       "All-predictor native work is a rate/cost control, not identical reference fitting",
                       "No new held-out or exhaustive-search qualification"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.study.resolve())
    with args.out.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
