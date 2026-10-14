#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize Modular entropy attribution and token-backing replays.

Substage medians describe instrumented calls only. The nested prefix aggregation
counter is subtracted from configuration search before calculating stage sums.
Paired deltas are medians of within-round differences, not differences of medians.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

STAGES = ("input", "transforms", "training", "tokens", "model", "emission", "assembly",
          "gpu_input_tokens")
TIMES = ("prefix_histogram_build", "prefix_histogram_cost", "prefix_clustering",
         "prefix_code_build", "prefix_value_collection", "prefix_config_search",
         "prefix_value_aggregation", "prefix_exact_measurement", "ans_prefix_validation",
         "ans_value_collection", "ans_value_aggregation", "ans_prepared_value_validation",
         "ans_uint_config", "ans_histogram_build", "ans_model_build", "ans_token_cost", "selection")
COUNTS = ("ans_uint_config_candidates", "ans_histogram_candidates", "ans_alphabet_width_candidates")
ROUTES = ("cpu", "gpu", "tokens")
BACKINGS = ("cpu_interleaved", "host_split", "metal_split")
MATCH = ("width", "height", "format", "rct", "entropy", "threads", "jobs", "stride",
         "byte_order", "bytes", "fingerprint", "domain_capacity", "split_tree")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def breakdown(row):
    values = {name: row[name + "_work_ns"] for name in TIMES}
    values["prefix_config_search"] -= values["prefix_value_aggregation"]
    require(values["prefix_config_search"] >= 0, "Nested aggregation exceeds its parent")
    values["unattributed_model"] = row["model_ns"] - sum(values.values())
    require(values["unattributed_model"] >= 0, "Entropy work exceeds serial model wall time")
    return values


def describe(rows, wall):
    parts = [breakdown(r) for r in rows]
    return dict(samples=len(rows), median_wall_ms=stats.median(r[wall] for r in rows) / 1e6,
                median_model_ms=stats.median(r["model_ns"] for r in rows) / 1e6,
                median_stage_ms={k: stats.median(p[k] for p in parts) / 1e6 for k in parts[0]},
                median_stage_fraction_of_model={k: stats.median(p[k] / r["model_ns"]
                    for p, r in zip(parts, rows)) for k in parts[0]},
                candidate_counts={k: sorted({r[k] for r in rows}) for k in COUNTS})


def paired_delta(groups, a, b):
    # b minus a; positive means b took longer.
    return dict(model_ms=stats.median(g[b]["model_ns"] - g[a]["model_ns"]
                                     for g in groups) / 1e6,
                stage_ms={k: stats.median(breakdown(g[b])[k] - breakdown(g[a])[k]
                                         for g in groups) / 1e6
                          for k in breakdown(groups[0][a])})


def analyze(root):
    provenance = json.loads((root / "entropy-run-provenance.json").read_text())
    require("finished_utc" in provenance, "Incomplete capture")
    require(sha(Path(provenance["binary"])) == provenance["binary_sha256"], "Binary changed")
    require(sha(root / "cases.tsv") == provenance["manifest_sha256"], "Case manifest changed")
    for path, digest in provenance["inputs_and_expected"].items():
        require(sha(Path(path)) == digest, f"Changed artifact: {path}")
    manifest = [line.split("\t") for line in (root / "cases.tsv").read_text().splitlines()
                if line and not line.startswith("#")]
    cases = {r[0]: r for r in manifest}
    require(len(cases) == len(manifest), "Duplicate cases")
    expected = {}
    for name in cases:
        data = (root / "expected" / (name + ".jxl")).read_bytes()
        expected[name] = (len(data), fingerprint(data))
    groups = defaultdict(dict)
    ordinary = defaultdict(list)
    raw_hashes = {}
    reference_rows = {}
    row_counts = Counter()
    for command in provenance["commands"]:
        require(command["returncode"] == 0, "Failed phase")
        mode, seed = command["mode"], command["seed"]
        path = root / f"{mode}-{seed}.csv"
        raw_hashes[path.name] = sha(path)
        phase_groups = defaultdict(dict)
        with path.open(newline="") as file:
            for raw in csv.DictReader(file):
                row = {k: v if k in ("case", "kind", "backend", "backing") else int(v)
                       for k, v in raw.items()}
                name, kind = row["case"], row["kind"]
                require(name in cases and kind in ("warmup", mode), "Unexpected case/kind")
                require(all(isinstance(v, str) or v >= 0 for v in row.values()), "Negative counter")
                labels = BACKINGS if mode == "replay" else ROUTES
                route = row["backing" if mode == "replay" else "backend"]
                key = (name, kind, row["round"])
                require(route in labels and route not in phase_groups[key], "Duplicate/invalid route")
                phase_groups[key][route] = row
                if mode == "measure":
                    require(all(row[s + "_ns"] == 0 for s in STAGES) and
                            all(row[s + "_work_ns"] == 0 for s in TIMES) and
                            all(row[k] == 0 for k in COUNTS), "Ordinary call was instrumented")
                else:
                    breakdown(row)
                    # Individual tiny scopes may round to zero at clock resolution.
                    require(sum(row[s + "_work_ns"] for s in TIMES if s.startswith("prefix_")) > 0,
                            "Missing prefix profile")
                    require((row["ans_histogram_candidates"] > 0) == (cases[name][3] == "ans"),
                            "ANS candidate counters disagree with policy")
                if mode != "replay":
                    require((row["bytes"], row["fingerprint"]) == expected[name], "Wrong expected bytes")
                    require(row["submissions"] == int(route != "cpu") and
                            row["allocations"] == 2 * ROUTES.index(route), "Wrong GPU path")
                    require(row["peak_backing"] <= row["domain_capacity"] and
                            0 < row["cpu_peak"] <= 4 and row["jobs"] == 1, "Bad resource envelope")
                    require(row["rct"] == int(cases[name][2]) and row["threads"] == int(cases[name][4])
                            and row["entropy"] == int(cases[name][3] == "ans") and
                            row["split_tree"] == int(cases[name][8] == "split7"), "Wrong case policy")
                    signature = tuple(row[k] for k in MATCH)
                    require(reference_rows.setdefault(name, signature) == signature, "Unmatched full calls")
                    if mode == "profile":
                        require(sum(row[s + "_ns"] for s in STAGES) <= row["ns"], "Overlapping wall stages")
                        require((row["gpu_input_tokens_ns"] > 0) == (route == "tokens"), "Bad composed profile")
                        if route == "tokens":
                            require(all(row[s + "_ns"] == 0 for s in STAGES[:4]), "Scalar token work on GPU route")
                row_counts[mode + ":" + kind] += 1
        rounds, warmups = provenance["rounds"], provenance["warmups"]
        wanted = {(name, kind, round) for name in cases for kind, count in
                  ((mode, rounds), ("warmup", warmups)) for round in range(count)}
        require(set(phase_groups) == wanted, "Incomplete case/round matrix")
        for name in cases:
            orders = Counter()
            for kind, count in ((mode, rounds), ("warmup", warmups)):
                for round in range(count):
                    triplet = phase_groups[(name, kind, round)]
                    labels = BACKINGS if mode == "replay" else ROUTES
                    require(set(triplet) == set(labels), "Incomplete triplet")
                    require(len({r["route_order"] for r in triplet.values()}) == 1, "Order mismatch")
                    require(all(0 <= r["route_order"] < 6 for r in triplet.values()), "Bad route order")
                    if mode != "measure":
                        require(len({tuple(r[k] for k in COUNTS) for r in triplet.values()}) == 1,
                                "Candidate counts changed across routes")
                    if mode == "replay":
                        require(len({(r["token_count"], r["context_count"]) for r in triplet.values()}) == 1,
                                "Replay token metadata differs")
                    if kind == mode:
                        orders[next(iter(triplet.values()))["route_order"]] += 1
                        if mode == "measure":
                            for route, row in triplet.items():
                                ordinary[(name, route)].append(row)
                        else:
                            groups[(mode, name)][(seed, round)] = triplet
            require(orders == Counter({i: rounds // 6 for i in range(6)}), "Unbalanced route orders")
    results = {}
    for name in cases:
        full = list(groups[("profile", name)].values())
        replay = list(groups[("replay", name)].values())
        require(full and replay and all(ordinary[(name, r)] for r in ROUTES), "Missing diagnostic phase")
        results[name] = dict(
            ordinary_ms={r: stats.median(x["ns"] for x in ordinary[(name, r)]) / 1e6 for r in ROUTES},
            profile={r: describe([g[r] for g in full], "ns") for r in ROUTES},
            profile_cohorts={str(seed): {r: describe([g[r] for (s, _), g in
                groups[("profile", name)].items() if s == seed], "ns") for r in ROUTES}
                for seed in sorted({s for s, _ in groups[("profile", name)]})},
            tokens_minus_cpu=paired_delta(full, "cpu", "tokens"),
            replay={r: describe([g[r] for g in replay], "model_ns") for r in BACKINGS},
            replay_cohorts={str(seed): {r: describe([g[r] for (s, _), g in
                groups[("replay", name)].items() if s == seed], "model_ns") for r in BACKINGS}
                for seed in sorted({s for s, _ in groups[("replay", name)]})},
            host_split_minus_interleaved=paired_delta(replay, "cpu_interleaved", "host_split"),
            metal_split_minus_host_split=paired_delta(replay, "host_split", "metal_split"))
    return dict(schema_version=1, binary_sha256=provenance["binary_sha256"],
                raw_csv_sha256=raw_hashes, validated_rows=dict(row_counts), cases=results,
                boundaries={"prefix_config_search": "Exclusive of nested raw-value aggregation in this summary",
                            "replay": "Warmed isolated model calls; preparation/copy/equality/destruction excluded",
                            "scope": "Selected cases on one active desktop; not dispatch qualification"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / "entropy-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["validated_rows"], indent=2))
    for name, case in result["cases"].items():
        p = case["profile"]["tokens"]
        largest = sorted(p["median_stage_ms"].items(), key=lambda item: item[1], reverse=True)[:3]
        print(name, f'model={p["median_model_ms"]:.3f} ms', largest)


if __name__ == "__main__":
    main()
