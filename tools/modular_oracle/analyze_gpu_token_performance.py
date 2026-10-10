#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify and analyze balanced CPU/preparation/token complete-call triplets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROUTES = ("cpu", "gpu", "tokens")
STAGES = ("input", "transforms", "training", "tokens", "model", "emission", "assembly",
          "gpu_input_tokens")
MATCH = ("width", "height", "format", "rct", "entropy", "threads", "jobs", "stride",
         "byte_order", "bytes", "fingerprint", "cpu_first", "domain_capacity", "split_tree",
         "route_order")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(paths, kind):
    cases = {}
    for path in paths:
        with path.open() as file:
            for row in csv.DictReader(file):
                if row["kind"] != kind:
                    continue
                backend, name = row.pop("backend"), row.pop("case")
                row.pop("kind")
                values = {k: int(v) for k, v in row.items()}
                key = (path.stem, values["round"])
                triplet = cases.setdefault(name, {}).setdefault(key, {})
                require(backend not in triplet, f"Duplicate route: {name} {key}")
                triplet[backend] = values
    for name, rounds in cases.items():
        for key, triplet in rounds.items():
            require(set(triplet) == set(ROUTES), f"Incomplete triplet: {name} {key}")
            for route, row in triplet.items():
                require(all(row[f] == triplet["cpu"][f] for f in MATCH),
                        f"Triplet mismatch: {name} {key} {route}")
                require(row["ns"] > 0 and row["peak_backing"] <= row["domain_capacity"] and
                        row["cpu_peak"] <= 4, f"Invalid time/resource count: {name} {key}")
                require(row["submissions"] == (0 if route == "cpu" else row["jobs"]) and
                        row["allocations"] == ROUTES.index(route) * 2 * row["jobs"],
                        f"Unexpected device route: {name} {key} {route}")
                require(0 <= row["route_order"] < 6 and row["cpu_first"] ==
                        int(row["route_order"] < 3), f"Invalid route order: {name} {key}")
                if kind == "profile":
                    require(sum(row[s + "_ns"] for s in STAGES) <= row["ns"],
                            f"Overlapping profile scopes: {name} {key} {route}")
                    require((route == "tokens") == (row["gpu_input_tokens_ns"] > 0),
                            f"Incorrect composed profile: {name} {key} {route}")
                    if route == "tokens":
                        require(all(row[s + "_ns"] == 0 for s in
                                    ("input", "transforms", "training", "tokens")),
                                f"Unexpected scalar preparation: {name} {key}")
                elif kind == "measure":
                    require(all(row[s + "_ns"] == 0 for s in STAGES),
                            f"Ordinary run was profiled: {name} {key} {route}")
    return cases


def comparison(rounds, control, rng):
    cohorts, orders = {}, {}
    for (cohort, _), triplet in rounds.items():
        ratio = triplet[control]["ns"] / triplet["tokens"]["ns"]
        cohorts.setdefault(cohort, []).append(ratio)
        orders.setdefault(str(triplet[control]["route_order"]), []).append(ratio)
    require(len({len(v) for v in cohorts.values()}) == 1 and set(orders) == set(map(str, range(6)))
            and len({len(v) for v in orders.values()}) == 1, "Unbalanced cohorts/permutations")
    draws = np.concatenate([rng.choice(v, size=(10000, len(v)), replace=True)
                            for v in cohorts.values()], axis=1)
    return dict(median=statistics.median(v for values in cohorts.values() for v in values),
                ci95=np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist(),
                cohorts={k: statistics.median(v) for k, v in cohorts.items()},
                orders={k: statistics.median(v) for k, v in sorted(orders.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    root = parser.parse_args().run
    ordinary = load(sorted(root.glob("measure-*.csv")), "measure")
    profiles = load(sorted(root.glob("profile-*.csv")), "profile")
    qualified = load(sorted(root.glob("qualify-*.csv")), "qualify")
    manifest = [line.split("\t") for line in (root / "cases.tsv").read_text().splitlines()
                if line and not line.startswith("#")]
    expected_cases = {row[0]: int(row[5]) for row in manifest}
    require(set(ordinary) == set(qualified) == set(expected_cases), "Incomplete ordinary/qualified matrix")
    require(set(profiles) == {n for n, jobs in expected_cases.items() if jobs == 1},
            "Incomplete profile matrix")
    provenance = json.loads((root / "run-provenance.json").read_text())
    require("finished_utc" in provenance and all(c.get("returncode") == 0 for c in provenance["commands"]),
            "Failed/incomplete timing run")
    require(provenance["measurement_environment"] ==
            {"MTL_DEBUG_LAYER": "0", "MTL_SHADER_VALIDATION": "0"},
            "Unexpected measurement instrumentation")
    require(sha256(Path(provenance["binary"])) == provenance["binary_sha256"],
            "Measured binary changed")
    inputs = json.loads((root / "inputs.json").read_text())["inputs"]
    require({row[1] for row in manifest} == {i["path"] for i in inputs.values()},
            "Input provenance does not match manifest")
    for item in inputs.values():
        require(sha256(Path(item["path"])) == item["sha256"], "Input samples changed")
    acceptance = json.loads((root / "acceptance.json").read_text())
    require(hashlib.sha256((root / "cases.tsv").read_bytes()).hexdigest() == acceptance["manifest_sha256"],
            "Acceptance/input matrix changed")
    require(acceptance["recorded_unix"] < min(c["started"] for c in provenance["commands"]),
            "Acceptance was not fixed before measurement")
    csvs = sorted(root.glob("measure-*.csv")) + sorted(root.glob("profile-*.csv")) + sorted(root.glob("qualify-*.csv"))
    for path in csvs:
        setup = path.with_name(path.stem + "-setup.txt")
        require("reference=1" in setup.read_text() and "token_routes=1" in setup.read_text(),
                "Missing independent oracle/three-route setup")
    for mode, data in (("measure", ordinary), ("profile", profiles)):
        expected = {(f"{mode}-{c['argv'][6]}", r) for c in provenance["commands"]
                    if c["argv"][1] == mode for r in range(int(c["argv"][4]))}
        require(expected and all(set(rounds) == expected for rounds in data.values()),
                f"Missing/unexpected {mode} rounds")
        warmups = load(sorted(root.glob(f"{mode}-*.csv")), "warmup")
        expected_warmups = {(f"{mode}-{c['argv'][6]}", r) for c in provenance["commands"]
                            if c["argv"][1] == mode for r in range(int(c["argv"][5]))}
        require(set(warmups) == set(data) and
                all(set(rounds) == expected_warmups for rounds in warmups.values()),
                f"Missing/unexpected {mode} warmups")
        for name, rounds in warmups.items():
            baseline = next(iter(qualified[name].values()))["cpu"]
            require(all(t["cpu"][f] == baseline[f] for t in rounds.values() for f in MATCH
                        if f not in ("cpu_first", "route_order")), "Warmup differs from qualification")
    results = {}
    rng = np.random.default_rng(20261010)
    for name, rounds in sorted(ordinary.items()):
        baseline = next(iter(qualified[name].values()))["cpu"]
        for data in (rounds, profiles.get(name, {})):
            for triplet in data.values():
                require(all(triplet["cpu"][f] == baseline[f] for f in MATCH
                            if f not in ("cpu_first", "route_order")),
                        f"Timed case changed qualified samples/policy/cap: {name}")
        r = {k: baseline[k] for k in MATCH if k not in ("fingerprint", "cpu_first", "route_order")}
        r["rounds"] = len(rounds)
        r["cpu_over_tokens"] = comparison(rounds, "cpu", rng)
        r["preparation_over_tokens"] = comparison(rounds, "gpu", rng)
        for route in ROUTES:
            rows = [t[route] for t in rounds.values()]
            ns = statistics.median(row["ns"] for row in rows)
            r[route] = dict(median_ms=ns / 1e6,
                           mpixels_per_second=r["width"] * r["height"] * r["jobs"] * 1000 / ns,
                           max_managed_peak=max(row["peak_backing"] for row in rows),
                           max_cpu_participants=max(row["cpu_peak"] for row in rows))
            if name in profiles:
                prof = [t[route] for t in profiles[name].values()]
                stages = {s + "_ms": statistics.median(row[s + "_ns"] for row in prof) / 1e6
                          for s in STAGES}
                stages["unattributed_ms"] = statistics.median(
                    row["ns"] - sum(row[s + "_ns"] for s in STAGES) for row in prof) / 1e6
                stages["full_ms"] = statistics.median(row["ns"] for row in prof) / 1e6
                r[route]["profile"] = stages
        results[name] = r
    primary, concurrency = {}, {}
    for name in acceptance["primary_cases"]:
        require(name in results and results[name]["threads"] == 4 and results[name]["jobs"] == 1,
                "Primary case absent or not four-participant single-call")
        primary[name] = all(results[name][c]["median"] >= acceptance["minimum_primary_speedup"] and
                            results[name][c]["ci95"][0] > 1.0
                            for c in ("cpu_over_tokens", "preparation_over_tokens"))
    for name in acceptance["concurrency_cases"]:
        require(name in results and results[name]["jobs"] == 4, "Missing four-call acceptance case")
        concurrency[name] = all(results[name][c]["median"] >= acceptance["minimum_concurrency_ratio"] and
                                results[name][c]["ci95"][0] >= acceptance["minimum_concurrency_ratio"]
                                for c in ("cpu_over_tokens", "preparation_over_tokens"))
    require(primary and concurrency, "Empty acceptance groups")
    report = dict(analysis_provenance=dict(
                      script_sha256=sha256(Path(__file__)),
                      run_provenance_sha256=sha256(root / "run-provenance.json"),
                      acceptance_sha256=sha256(root / "acceptance.json"),
                      input_manifest_sha256=sha256(root / "inputs.json")),
                  bootstrap_seed=20261010, bootstrap_replicates=10000,
                  interval="95% within-process stratified bootstrap of median paired control/token ratio",
                  primary_cases=primary, concurrency_cases=concurrency,
                  pilot_gate_passed=all(primary.values()) and all(concurrency.values()),
                  cases=results)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Case | CPU ms | Prep ms | Tokens ms | CPU/tokens [95% CI] | Prep/tokens [95% CI] |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        ratios = []
        for key in ("cpu_over_tokens", "preparation_over_tokens"):
            q = r[key]
            ratios.append(f"{q['median']:.3f} [{q['ci95'][0]:.3f}, {q['ci95'][1]:.3f}]")
        lines.append(f"| {name} | {r['cpu']['median_ms']:.3f} | {r['gpu']['median_ms']:.3f} | "
                     f"{r['tokens']['median_ms']:.3f} | {' | '.join(ratios)} |")
    (root / "results-table.md").write_text("\n".join(lines) + "\n")
    hashes = {str(p.relative_to(root)): sha256(p)
              for p in sorted(root.rglob("*")) if p.is_file() and p.name != "artifacts-sha256.json"}
    (root / "artifacts-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Verified {len(results)} cases; primary {sum(primary.values())}/{len(primary)}, "
          f"concurrency {sum(concurrency.values())}/{len(concurrency)}; "
          f"pilot gate passed: {report['pilot_gate_passed']}")


if __name__ == "__main__":
    main()
