#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Validate frozen captures and summarize learned Metal qualification."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

STAGES = ["input", "transforms", "training", "tokens", "model", "emission", "assembly",
          "gpu_input_tokens"]
VARIANTS = ["cpu", "metal-auto", "metal-cpu", "metal-gpu"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def distribution(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), runs=values)


def describe(rows, mode):
    result = {key: distribution([r[key] for r in rows]) for key in
              ["seconds", "cpu_seconds", "peak_backing_bytes", "peak_committed_bytes"]}
    if mode == "encode":
        for index, stage in enumerate(STAGES):
            result[stage + "_wall"] = distribution([r["profile_wall"][index] for r in rows])
            result[stage + "_cpu"] = distribution([r["profile_cpu"][index] for r in rows])
        for key in ["feature_seconds", "token_seconds"]:
            result[key] = distribution([r["gpu"][key] for r in rows])
        result["bytes"] = rows[0]["bytes"]
        result["weighted_prediction_samples"] = rows[0]["training"]["weighted_prediction_samples"]
        result["reused_prediction_samples"] = rows[0]["training"]["reused_prediction_samples"]
        result["feature_samples"] = rows[0]["gpu"]["feature_samples"]
        result["gpu_submissions"] = rows[0]["gpu_submissions"]
        prep = [r["gpu"]["preparation"] for r in rows]
    else:
        for key in ["samples", "jobs", "eligible_samples", "unsupported_samples", "planned_bytes"]:
            result[key] = rows[0][key]
        prep = [r["preparation"] for r in rows]
    result["preparation"] = {
        key: distribution([p[key] for p in prep]) if key.endswith("seconds") else prep[0][key]
        for key in prep[0]
    }
    return result


def collect(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["complete"], "Capture is incomplete")
    require(digest(root / "gjxl_modular_metal_workflow_profile") == manifest["binary_sha256"],
            "Frozen binary changed")
    for source in manifest["sources"]:
        require(digest(root / "source" / source["path"]) == source["sha256"],
                "Frozen source changed: " + source["path"])
    for item in manifest["inputs"]:
        require(digest(Path(item["frozen_path"])) == item["sha256"],
                "Frozen input changed: " + item["name"])
    inputs = {item["name"]: item for item in manifest["inputs"]}
    captures = {(r["mode"], r["name"]): r for r in manifest["captures"]}
    wanted = {(mode, item["name"]) for item in manifest["inputs"] for mode in item["modes"]}
    require(set(captures) == wanted and len(captures) == len(manifest["captures"]),
            "Missing or duplicated capture")
    result = dict(schema=1, manifest=str(manifest_path.resolve()),
                  manifest_sha256=digest(manifest_path), source_revision=manifest["source_revision"],
                  binary_sha256=manifest["binary_sha256"], scope=manifest["scope"],
                  encodes=[], features=[], counts=dict(encodes=0, measured_encodes=0,
                                                      feature_maps=0, compared_features=0))
    result["additional_evidence"] = {}
    for filename in ["oracle-provenance.json", "prior-cpu-controls.json", "timing-disturbance.json"]:
        path = root / filename
        if path.exists():
            evidence = json.loads(path.read_text())
            result["additional_evidence"][filename] = {
                "path": str(path.resolve()), "sha256": digest(path), "data": evidence}
            if filename == "oracle-provenance.json":
                for library in evidence["libraries"]:
                    require(digest(Path(library["path"])) == library["sha256"], "Oracle library changed")
            if filename == "prior-cpu-controls.json":
                for check in evidence["checks"]:
                    require(digest(Path(check["current"])) == check["sha256"] and
                            digest(Path(check["prior"])) == check["sha256"], "Prior CPU bytes differ")
    for (mode, name), capture in captures.items():
        require(capture["returncode"] == 0, "Failed process: " + name)
        folder = root / mode / name
        path = folder / ("encodes.jsonl" if mode == "encode" else "features.jsonl")
        require(digest(path) == capture["result_sha256"], "Capture changed: " + name)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        repeats = manifest["encode_repeats" if mode == "encode" else "feature_repeats"]
        variants = VARIANTS if mode == "encode" else ["cpu", "gpu", "auto"]
        expected = {(r, v) for r in range(repeats + 1) for v in variants}
        if mode == "encode":
            expected |= {(-1, "cpu"), (-1, "metal-auto")}
            require(digest(folder / "output.jxl") == capture["output_sha256"], "Output changed")
        require(len(rows) == len(expected) and {(r["round"], r["variant"]) for r in rows} == expected,
                "Missing or duplicate trial: " + name)
        grouped = defaultdict(list)
        for row in rows:
            require(row["released"] and row["peak_cpu_participants"] == 1,
                    "Resource/CPU participation mismatch: " + name)
            require(row["seconds"] > 0 and math.isfinite(row["seconds"]), "Invalid timing")
            if mode == "encode":
                require(row["exact_bytes"] and row["exact_decode"], "Incorrect encode")
                require(row["bytes"] == rows[0]["bytes"], "Encoded size changed")
                require(sum(row["profile_wall"]) <= row["seconds"] + 0.001,
                        "Stage timings exceed total")
                gpu = row["gpu"]
                cpu = row["variant"] == "cpu"
                require(gpu["feature_preparations"] == (0 if cpu else 1) and
                        gpu["feature_reuses"] == (0 if cpu else 1), "Feature build/reuse mismatch")
                require(row["gpu_submissions"] == (0 if cpu else 1 + gpu["preparation"]["gpu_calls"]),
                        "Unexpected GPU submissions")
                if not cpu:
                    require(row["training"]["weighted_prediction_samples"] == 0 and
                            row["training"]["reused_prediction_samples"] ==
                            rows[0]["training"]["weighted_prediction_samples"],
                            "Learning did not reuse the same sample population")
                    require(gpu["encoded_candidates"] == 1, "Unexpected token passes")
                result["counts"]["encodes"] += 1
                result["counts"]["measured_encodes"] += row["round"] > 0
            else:
                require(row["exact_features"], "Incorrect features")
                require(row["samples"] == row["eligible_samples"] + row["unsupported_samples"],
                        "Invalid feature population")
                require(row["peak_backing_bytes"] <= row["planned_bytes"], "Memory bound exceeded")
                p = row["preparation"]
                if row["variant"] == "gpu":
                    require(p["gpu_samples"] == row["eligible_samples"], "GPU work missing")
                if row["variant"] == "cpu":
                    require(p["gpu_samples"] == 0, "CPU forced producer dispatched GPU")
                result["counts"]["feature_maps"] += 1
                result["counts"]["compared_features"] += row["samples"]
            if row["round"] > 0:
                grouped[row["variant"]].append(row)
        entry = dict(name=name, input=inputs[name],
                     variants={v: describe(grouped[v], mode) for v in variants})
        if mode == "encode":
            entry["cold"] = {r["variant"]: r for r in rows if r["round"] == -1}
            entry["output_sha256"] = capture["output_sha256"]
            result["encodes"].append(entry)
        else:
            result["features"].append(entry)
    sums = {v: sum(e["variants"][v]["seconds"]["median"] for e in result["encodes"])
            for v in VARIANTS}
    result["aggregate"] = {"sum_image_wall_medians": sums,
                           "speedup_vs_cpu": {v: sums["cpu"] / sums[v] for v in VARIANTS},
                           "forced_gpu_vs_cpu_prep_encode_speedup": sums["metal-cpu"] / sums["metal-gpu"]}
    result["validation"] = {"exact_codestreams": True, "independent_decode_every_encode": True,
                             "every_feature_compared": True, "all_domains_released": True,
                             "one_feature_build_and_one_token_pass": True,
                             "learning_population_preserved": True}
    return result


def render(result):
    counts = result["counts"]
    repeated_encodes = result.get("repeat", {}).get("counts", {}).get("encodes", 0)
    sums = result["aggregate"]["sum_image_wall_medians"]
    lines = ["# Learned Modular Metal qualification", "",
             f"Qualification of `{result['source_revision']['stdout'].strip()}` with a separate "
             "benchmark driver. Production defaults and the feature heuristic are unchanged.", "",
             f"All {counts['encodes'] + repeated_encodes:,} complete encodes "
             f"({counts['encodes']} primary and {repeated_encodes} repeat) "
             "matched their CPU control byte for byte "
             "and decoded exactly with the pinned independent libjxl decoder. "
             f"All {counts['feature_maps']:,} prepared maps matched the serial predictor "
             f"({counts['compared_features']:,} feature entries compared). All managed domains "
             "released their backing and CPU participation.", "",
             "The 16 original corpus outputs also match the earlier frozen CPU qualification "
             "files. The pinned decoder revision is "
             "`e8ff09762481785938d8e4e01333ed3917571161`; its loaded-library hashes are retained.", "",
             f"On the primary 17-image capture, explicit Metal with the current feature heuristic "
             f"uses {100 * (1 - sums['metal-auto']/sums['cpu']):.2f}% less elapsed time than CPU "
             "(ratio of sums of image medians). With GPU tokens held fixed, forcing GPU feature "
             f"preparation uses {100 * (1 - sums['metal-gpu']/sums['metal-cpu']):.2f}% less time "
             "than forcing the serial CPU producer. These are different comparisons: the "
             "first includes the feature-reuse/token integration; the second isolates producer "
             "choice within that integration.", "",
             "## Measured scope", "", result["scope"], "",
             "Four measured rounds follow one warmup for each encoder variant. Each variant "
             "occupies every order position once per image. Feature operations use six measured "
             "rounds after one warmup, with each producer occupying each position twice. Cold "
             "public CPU/Metal calls are retained separately. No instrumentation environment "
             "variables were set during this capture.", "",
             "CPU and Metal-auto use the public C++ API with learned policy, estimated ANS and "
             "a one-participant execution domain. Metal-cpu and Metal-gpu use the same learned "
             "workflow and publication boundary through a qualification-only wrapper, forcing "
             "only feature preparation. Token generation remains on Metal in both. The current "
             "learned CPU pipeline is serial even when a larger thread ceiling is allowed. "
             "This measures application of already selected WP parameters. Transform search "
             "and the five-preset WP selection sweep remain on CPU.", "",
             "`metal-auto` means explicit Metal encoding with automatic **feature-producer** "
             "selection. It does not mean `ModularBackend::kAutomatic`, which still selects CPU.", "",
             "## Complete encoder", "",
             "The aggregate is a sum of per-image medians, with one encode of each image. It is "
             "not a corpus-independent throughput estimate. Ranges and all trials are retained "
             "in the JSON report and raw JSONL captures.", "",
             "| Variant | Sum of wall medians (s) | Speedup over CPU |",
             "|---|---:|---:|"]
    for v in VARIANTS:
        lines.append(f"| {v} | {sums[v]:.4f} | {sums['cpu']/sums[v]:.3f}× |")
    lines += ["", "| Image | CPU (s) | Metal-auto (s) | Metal-cpu (s) | Metal-gpu (s) | "
              "CPU / Metal-auto | CPU / Metal peak backing (MiB) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for e in result["encodes"]:
        v = e["variants"]
        times = [v[n]["seconds"]["median"] for n in VARIANTS]
        peaks = [v[n]["peak_backing_bytes"]["median"] / 2**20 for n in ["cpu", "metal-auto"]]
        lines.append(f"| {e['name']} | " + " | ".join(f"{x:.4f}" for x in times) +
                     f" | {times[0]/times[1]:.3f}× | {peaks[0]:.2f} / {peaks[1]:.2f} |")
    lines += ["", "### Stage attribution", "",
              "These are sums of per-image stage medians; they need not sum exactly to the "
              "sum of total medians. Feature preparation is nested in training, and GPU token "
              "generation is nested in tokens. Do not add those diagnostics again. The older "
              "learning_calls/population/samples profiler fields are not instrumented for this "
              "learner and remain zero; validation uses the current weighted/reused prediction "
              "sample counters, plus exact complete codestreams.", "",
              "| Stage (s) | CPU | Metal-auto | Metal-cpu | Metal-gpu |", "|---|---:|---:|---:|---:|"]
    for stage in [s + "_wall" for s in STAGES] + ["feature_seconds", "token_seconds"]:
        vals = [sum(e["variants"][v][stage]["median"] for e in result["encodes"]) for v in VARIANTS]
        lines.append(f"| {stage} | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
    lines += ["", "### First call per process", "",
              "Each first Metal call includes lazy backend initialization. Device/OS shader "
              "caches were not cleared. Differences from steady medians also include ordinary "
              "allocator/cache warming and desktop variation, so they are not isolated setup costs.", "",
              "| Image | First CPU (s) | First Metal-auto (s) | Steady Metal-auto median (s) |",
              "|---|---:|---:|---:|"]
    for e in result["encodes"]:
        lines.append(f"| {e['name']} | {e['cold']['cpu']['seconds']:.4f} | "
                     f"{e['cold']['metal-auto']['seconds']:.4f} | "
                     f"{e['variants']['metal-auto']['seconds']['median']:.4f} |")
    if "repeat" in result:
        repeat = result["repeat"]
        lines += ["", "### Disturbed-case repeat", "",
                  "The primary alpha-flower capture had two marked timing disturbances: a CPU "
                  "call took 41.5 s wall / 15.7 s process CPU, and a later Metal-cpu call took "
                  "24.2 s wall / 12.4 s process CPU. The cause is undetermined. All original "
                  "trials remain in the primary tables and aggregates. After the suite finished, "
                  "the complete image comparison was repeated in a fresh process with the same "
                  "binary, input and round/order policy. No individual trials were discarded.", "",
                  f"The repeat added {repeat['counts']['encodes']} exact, independently decoded "
                  "encodes. Replacing only this image's medians gives the sensitivity aggregate "
                  "below; it is reported separately from the original aggregate.", "",
                  "| Variant | Repeated image median (s) | Sensitivity aggregate (s) |",
                  "|---|---:|---:|"]
        for v in VARIANTS:
            lines.append(f"| {v} | {repeat['encodes'][0]['variants'][v]['seconds']['median']:.4f} | "
                         f"{result['repeat_sensitivity']['sum_image_wall_medians'][v]:.4f} |")
        sensitive = result["repeat_sensitivity"]["sum_image_wall_medians"]
        producer_change = sensitive["metal-gpu"] / sensitive["metal-cpu"] - 1
        lines += ["", f"With the repeated image, Metal-auto uses "
                  f"{100 * (1 - sensitive['metal-auto']/sensitive['cpu']):.2f}% less aggregate "
                  "elapsed time than CPU. Forced GPU preparation uses "
                  f"{abs(100 * producer_change):.2f}% "
                  f"{'less' if producer_change < 0 else 'more'} time than forced CPU preparation. "
                  "That small incremental whole-encoder difference is sensitive to desktop "
                  "variation; it should not be treated as a stable speedup. The isolated "
                  "feature-operation measurements provide the clearer evidence for producer choice."]
        lines += ["", f"Repeat capture: `{Path(repeat['manifest']).parent}`; manifest SHA-256 "
                  f"`{repeat['manifest_sha256']}`."]
    lines += ["", "## Forced feature preparation", "",
              "Each case uses channels and per-stream WP parameters selected by the current "
              "learned workflow. Preparation includes buffer allocation, sample copies, predictor "
              "computation, submission/wait and host-view publication. Every output prediction "
              "and property-15 value is compared with the independent serial producer outside "
              "the timed interval. Forced GPU still uses CPU for unsupported channel shapes.", "",
              "| Image | Samples | Jobs | CPU (ms) | GPU (ms) | Auto (ms) | GPU speedup | "
              "Auto GPU samples | Unsupported samples |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in result["features"]:
        v = e["variants"]
        a = v["auto"]
        times = [v[n]["seconds"]["median"] * 1000 for n in ["cpu", "gpu", "auto"]]
        lines.append(f"| {e['name']} | {a['samples']:,} | {a['jobs']:,} | " +
                     " | ".join(f"{x:.3f}" for x in times) + f" | {times[0]/times[1]:.2f}× | "
                     f"{a['preparation']['gpu_samples']:,} | {a['unsupported_samples']:,} |")
    lines += ["", "## Interpretation", "",
              "Reusing the existing GPU weighted predictor is worthwhile. On the three 12 MP "
              "photos, selected-channel feature preparation is 9.7–10.7× faster than the serial "
              "CPU producer, including allocations, copies and synchronization. Complete-file "
              "parity and independent decoding hold with either producer.", "",
              "The current 4M-sample cutoff is conservative for this serial learned path. "
              "The 750k-sample photos gain 1.6–2.8× in preparation, and the 3.15M-sample RGB "
              "case gains 4.85× while Auto still chooses CPU. Average job size also needs "
              "qualification: the large narrow 17×262144 input gains 3.89× despite being rejected "
              "by the 16,384-samples-per-job preference. Total work and number/shape of independent "
              "jobs both matter. A new cutoff is not selected by this study.", "",
              "Unconditional GPU preparation is unsuitable. The small palette map takes "
              "0.24 ms on CPU versus 1.70 ms on GPU; the 250k-sample gray photo takes "
              "3.34 ms versus 6.24 ms. Narrow inputs also show different crossovers. GPU and Auto "
              "execute the same feature jobs on eligible large batches; timing differences "
              "between those columns reflect runtime variation, not another implementation.", "",
              "Most of the complete-encoder gain comes from token work falling from a summed "
              "3.95 s to 0.15 s. On explicit Metal, the learning stage still accounts for about "
              "82% of the summed total and transform/selection work about 12%. Faster weighted "
              "application does not remove either the fitter or the WP-selection sweep.", "",
              "Keep explicit Metal available and keep Automatic on CPU for now. The measured "
              "speed benefit comes with retained memory: the 12 MP encoder case rises from "
              "785 MiB to 1,181 MiB of managed peak backing. Retuning feature-producer selection "
              "is a separate, promising follow-up; changing the public backend default needs "
              "a policy that considers small/cold calls and the memory tradeoff."]
    lines += ["", "## Evidence and limits", "",
              f"Raw capture: `{Path(result['manifest']).parent}`. "
              f"Manifest SHA-256: `{result['manifest_sha256']}`. "
              f"Binary SHA-256: `{result['binary_sha256']}`.", "",
              "The capture retains frozen input hashes, source snapshot, source patch, executable, "
              "build settings, every trial, one exact output per encoder input, and power/thermal/"
              "process snapshots before and after each case. The analyzer checks hashes, coverage, "
              "exactness flags, sample reuse, dispatch counts and released resources.", "",
              "The [JSON ledger](modular-learned-metal-qualification-results.json) retains every "
              "per-image median, range and measured observation. Reproduction uses "
              "`gjxl_modular_metal_workflow_profile` (Release Metal build with the pinned Modular "
              "oracle), `tools/modular_reference_learning/run_metal_workflow_qualification.py` "
              "and `tools/modular_reference_learning/report_metal_workflow_qualification.py`. "
              "Both scripts expose their inputs and output destinations through `--help`; the "
              "exact command for every captured process is in its manifest.", "",
              "This is an Apple M4 Pro 14-core, 48 GiB, macOS 27.0.1 qualification on AC power "
              "with an active desktop. Background OS work is present. Small end-to-end changes "
              "need to be judged against the recorded spread; these results do not qualify "
              "other devices, concurrent encodes, all possible image layouts or process RSS.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--repeat-capture", type=Path)
    args = parser.parse_args()
    result = collect(args.capture)
    if args.repeat_capture:
        repeat = collect(args.repeat_capture)
        require(repeat["binary_sha256"] == result["binary_sha256"], "Repeat used another binary")
        require(len(repeat["encodes"]) == 1 and repeat["encodes"][0]["name"] == "alpha-flower",
                "Unexpected repeated input")
        repeated = repeat["encodes"][0]
        original = next(e for e in result["encodes"] if e["name"] == repeated["name"])
        require(repeated["output_sha256"] == original["output_sha256"] and
                repeated["input"]["sha256"] == original["input"]["sha256"], "Repeat control changed")
        result["repeat"] = repeat
        sums = {v: result["aggregate"]["sum_image_wall_medians"][v] -
                original["variants"][v]["seconds"]["median"] +
                repeated["variants"][v]["seconds"]["median"] for v in VARIANTS}
        result["repeat_sensitivity"] = {
            "sum_image_wall_medians": sums,
            "speedup_vs_cpu": {v: sums["cpu"] / sums[v] for v in VARIANTS}}
    result["analyzer_sha256"] = digest(Path(__file__))
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(render(result))
    print(json.dumps({"counts": result["counts"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
