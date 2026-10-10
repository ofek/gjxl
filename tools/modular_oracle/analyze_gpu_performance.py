#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze complete-call paired CPU/Metal measurements (requires numpy).

Bootstrap the median per-pair CPU/GPU ratio within each process cohort; equal
cohort sizes are required. Intervals describe this run's paired sampling noise,
not cross-machine/session uncertainty. Profiles are diagnostic and not pooled.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

STAGES = ["input_ns", "transforms_ns", "training_ns", "tokens_ns", "model_ns", "emission_ns", "assembly_ns"]


def load(paths, kind):
    cases = {}
    for path in paths:
        for row in csv.DictReader(path.open()):
            if row["kind"] != kind:
                continue
            backend, name = row.pop("backend"), row.pop("case")
            row.pop("kind")
            numeric = {k: int(v) for k, v in row.items()}
            key = (path.stem, numeric["round"])
            pair = cases.setdefault(name, {}).setdefault(key, {})
            if backend in pair:
                raise ValueError(f"Duplicate backend: {name} {key}")
            pair[backend] = numeric
    for name, pairs in cases.items():
        for key, pair in pairs.items():
            if set(pair) != {"cpu", "gpu"}:
                raise ValueError(f"Unpaired sample: {name} {key}")
            for field in ("width", "height", "format", "rct", "entropy", "threads", "jobs", "stride",
                          "byte_order", "bytes", "fingerprint", "cpu_first", "domain_capacity"):
                if pair["cpu"][field] != pair["gpu"][field]:
                    raise ValueError(f"Pair mismatch: {name} {key} {field}")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    ordinary = load(sorted(root.glob("measure-*.csv")), "measure")
    profiles = load(sorted(root.glob("profile-*.csv")), "profile")
    if not ordinary:
        raise ValueError("No ordinary measurements")
    provenance = json.loads((root / "run-provenance.json").read_text())
    if "finished_utc" not in provenance or any(c.get("returncode") != 0 for c in provenance["commands"]):
        raise ValueError("Incomplete/failed qualification run")
    expected_cases = {line.split("\t")[0]: int(line.split("\t")[5])
                      for line in (root / "cases.tsv").read_text().splitlines()
                      if line and not line.startswith("#")}
    if set(ordinary) != set(expected_cases) or set(profiles) != {
            name for name, jobs in expected_cases.items() if jobs == 1}:
        raise ValueError("Measured/profiled case set differs from the manifest")
    expected_pairs = {f"{c['argv'][1]}-{c['argv'][6]}": int(c["argv"][4])
                      for c in provenance["commands"]}
    for mode, data in (("measure", ordinary), ("profile", profiles)):
        expected = {(cohort, r) for cohort, count in expected_pairs.items()
                    if cohort.startswith(mode + "-") for r in range(count)}
        if any(set(pairs) != expected for pairs in data.values()):
            raise ValueError(f"Missing/unexpected pairs in {mode}")
    results = {}
    rng = np.random.default_rng(20261010)
    for name, pairs in ordinary.items():
        groups = {}
        orders = {0: [], 1: []}
        for (cohort, _), p in pairs.items():
            ratio = p["cpu"]["ns"] / p["gpu"]["ns"]
            groups.setdefault(cohort, []).append(ratio)
            orders[p["cpu"]["cpu_first"]].append(ratio)
            if p["cpu"]["submissions"] or p["gpu"]["submissions"] != p["gpu"]["jobs"]:
                raise ValueError("Unexpected submission count")
            if p["cpu"]["allocations"] or p["gpu"]["allocations"] != 2 * p["gpu"]["jobs"]:
                raise ValueError("Unexpected allocation count")
        if len({len(v) for v in groups.values()}) != 1 or len(orders[0]) != len(orders[1]):
            raise ValueError("Unbalanced cohort/order counts")
        ratios = [v for group in groups.values() for v in group]
        draws = np.concatenate([rng.choice(v, size=(10000, len(v)), replace=True)
                                for v in groups.values()], axis=1)
        ci = np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()
        p0 = next(iter(pairs.values()))["cpu"]
        r = {k: p0[k] for k in ("width", "height", "format", "rct", "entropy", "threads", "jobs", "stride", "byte_order", "bytes", "domain_capacity")}
        r.update(pairs=len(pairs), speedup=statistics.median(ratios), speedup_ci95=ci,
                 cohort_speedups={k: statistics.median(v) for k, v in groups.items()},
                 order_speedups={str(k): statistics.median(v) for k, v in orders.items()},
                 verdict="gpu faster" if ci[0] > 1 else "cpu faster" if ci[1] < 1 else "inconclusive")
        for backend in ("cpu", "gpu"):
            rows = [p[backend] for p in pairs.values()]
            ns = statistics.median(row["ns"] for row in rows)
            r[backend] = dict(median_ms=ns / 1e6,
                p10_ms=float(np.quantile([row["ns"] for row in rows], 0.1)) / 1e6,
                p90_ms=float(np.quantile([row["ns"] for row in rows], 0.9)) / 1e6,
                mpixels_per_second=p0["width"] * p0["height"] * p0["jobs"] * 1000 / ns,
                max_managed_peak=max(row["peak_backing"] for row in rows),
                max_cpu_participants=max(row["cpu_peak"] for row in rows))
            if name in profiles:
                prof = [p[backend] for p in profiles[name].values()]
                stages = {s.replace("_ns", "_ms"): statistics.median(row[s] for row in prof) / 1e6
                          for s in STAGES}
                stages["preparation_ms"] = statistics.median(row["input_ns"] + row["transforms_ns"] for row in prof) / 1e6
                stages["unattributed_ms"] = statistics.median(row["ns"] - sum(row[s] for s in STAGES) for row in prof) / 1e6
                stages["full_ms"] = statistics.median(row["ns"] for row in prof) / 1e6
                r[backend]["profile_ms"] = stages
        results[name] = r
    report = dict(bootstrap_seed=20261010, bootstrap_replicates=10000,
                  interval="95% stratified within-cohort bootstrap of median paired CPU/GPU ratio",
                  cases=results)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Case | CPU ms | GPU ms | CPU/GPU paired ratio [95% CI] | Result |",
             "| --- | ---: | ---: | ---: | --- |"]
    for name, r in results.items():
        lo, hi = r["speedup_ci95"]
        lines.append(f"| {name} | {r['cpu']['median_ms']:.3f} | {r['gpu']['median_ms']:.3f} | "
                     f"{r['speedup']:.3f} [{lo:.3f}, {hi:.3f}] | {r['verdict']} |")
    (root / "results-table.md").write_text("\n".join(lines) + "\n")
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*")) if p.is_file() and p.name != "artifacts-sha256.json"}
    (root / "artifacts-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
