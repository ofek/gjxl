#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Counterbalanced, exact-byte CPU/hybrid qualification of one histogram policy."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def environment():
    result = {"platform": platform.platform(), "load_average": os.getloadavg()}
    if platform.system() == "Darwin":
        result["power"] = command(["pmset", "-g", "batt"])
        result["cpu"] = command(["sysctl", "-n", "machdep.cpu.brand_string"])
        result["physical_memory_bytes"] = int(command(["sysctl", "-n", "hw.memsize"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", choices=["cpu", "shared-cpu", "hybrid", "auto"],
                        default=["cpu", "shared-cpu", "hybrid", "auto"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    require(args.repeats > 0 and args.warmups >= 0, "Invalid repeat/warmup count")
    require("cpu" in args.routes and len(set(args.routes)) == len(args.routes), "Routes need one CPU control")
    repo = Path(__file__).resolve().parents[2]
    status = command(["git", "-C", str(repo), "status", "--porcelain=v1"])
    require(not status, "Commit the implementation before freezing a qualification run")
    binary, out = args.binary.resolve(), args.out.resolve()
    inputs = [p.resolve() for p in args.input]
    require(len({p.stem for p in inputs}) == len(inputs), "Input names must be unique")
    require(binary.is_file() and all(p.is_file() for p in inputs), "Missing binary or input")
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "hybrid_profile"
    if platform.system() == "Darwin":
        subprocess.run(["cp", "-c", str(binary), str(frozen)], check=True)
    else:
        shutil.copy2(binary, frozen)
    manifest = {
        "schema": 1, "status": "running", "started_unix": time.time(),
        "source_revision": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "source_status": status, "binary_source": str(binary), "binary_sha256": digest(frozen),
        "environment_start": environment(), "routes": args.routes,
        "repeats": args.repeats, "warmups": args.warmups,
        "inputs": [{"name": p.stem, "path": str(p), "sha256": digest(p)} for p in inputs],
        "stages": ["input", "transforms", "training", "tokens", "model", "emission", "assembly", "gpu_input_tokens"],
        "scope": "Fresh serial processes in rotating route order. Same automatic/local-tree Best histogram "
                 "policy and estimated direct ANS (64 clusters). Complete encoder API interval includes "
                 "selection, feature/sample preparation, fitting, tokens, entropy, admission, output copy "
                 "and prepared-frame teardown. Excludes input IO, backend initialization, independent "
                 "source unpack/decode and output IO. CPU ceiling four for every route; ordinary CPU "
                 "histogram workflow remains serial. Managed memory excludes packed input, reference "
                 "decode buffers, stack, allocator bookkeeping and backend initialization. Feature/sample/fit "
                 "detail is nested within training; GPU-token detail is nested within tokens.",
    }
    if platform.system() == "Darwin":
        manifest["linked_libraries"] = command(["otool", "-L", str(frozen)])
        manifest["dependency_sha256"] = {}
        for line in manifest["linked_libraries"].splitlines()[1:]:
            path = Path(line.strip().split(" (", 1)[0])
            if path.is_absolute() and path.is_file() and not str(path).startswith(("/usr/lib/", "/System/")):
                manifest["dependency_sha256"][str(path)] = digest(path)
    for name in ["CMakeCache.txt", "compile_commands.json"]:
        source = binary.parent / name
        if source.is_file():
            shutil.copy2(source, out / name)
    manifest_path = out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    rows, controls = [], {}
    try:
        with (out / "results.jsonl").open("x", buffering=1) as ledger:
            for image_index, source in enumerate(inputs):
                for iteration in range(args.warmups + args.repeats):
                    offset = (image_index + iteration) % len(args.routes)
                    routes = args.routes[offset:] + args.routes[:offset]
                    for route in routes:
                        name = f"{source.stem}-{route}-{iteration}"
                        directory = out / name
                        argv = [str(frozen), str(source), str(directory), route, "auto", "best"]
                        started = time.time()
                        with (out / (name + ".stdout")).open("x") as stdout, (out / (name + ".stderr")).open("x") as stderr:
                            subprocess.run(argv, stdout=stdout, stderr=stderr, check=True)
                        result = json.loads((directory / "result.json").read_text())
                        checksum = digest(directory / "output.jxl")
                        require(result["schema"] == 1 and result["exact_decode"] and result["route"] == route,
                                "Invalid qualification result")
                        require(result["buffering"] == "auto" and result["predictors"] == "best", "Policy changed")
                        control = (checksum, result["work"], result["bytes"])
                        require(controls.setdefault(source.stem, control) == control,
                                "Route/repetition changed encoded bytes or learning work")
                        for kind in ["wall", "cpu"]:
                            values = result["profile_" + kind]
                            require(len(values) == len(manifest["stages"]) and min(values) >= -1e-6 and
                                    sum(values) <= result["total"][kind] + 1e-4, "Invalid stage attribution")
                        if result["work"]["local_trees"]:
                            require(result["memory"]["peak_feature_samples"] <= 8 * 1024 * 1024,
                                    "Local workflow exceeded its live feature batch bound")
                        row = {"input": source.stem, "route": route, "iteration": iteration,
                               "warmup": iteration < args.warmups, "started_unix": started,
                               "finished_unix": time.time(), "command": argv,
                               "output_sha256": checksum, "result": result}
                        rows.append(row)
                        ledger.write(json.dumps(row, sort_keys=True) + "\n")
                        print(name, f"{result['total']['wall']:.6f} s", result["bytes"],
                              f"GPU feature batches={result['routing']['gpu_feature_batches']}", flush=True)
        summary = []
        for source in inputs:
            for route in args.routes:
                results = [r["result"] for r in rows if r["input"] == source.stem and r["route"] == route and not r["warmup"]]
                summary.append({"input": source.stem, "route": route, "repeats": len(results),
                                "bytes": results[0]["bytes"],
                                "median_wall": statistics.median(r["total"]["wall"] for r in results),
                                "median_cpu": statistics.median(r["total"]["cpu"] for r in results),
                                "median_stage_wall": [statistics.median(r["profile_wall"][i] for r in results)
                                                      for i in range(len(manifest["stages"]))],
                                "median_detail_wall": {k: statistics.median(r["detail_wall"][k] for r in results)
                                                       for k in results[0]["detail_wall"]},
                                "routing": results[0]["routing"],
                                "maximum_peak_backing_bytes": max(r["memory"]["peak_backing_bytes"] for r in results),
                                "maximum_live_feature_samples": max(r["memory"]["peak_feature_samples"] for r in results)})
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for image in manifest["inputs"]:
            require(digest(Path(image["path"])) == image["sha256"], "Input changed during qualification")
        for path, checksum in manifest.get("dependency_sha256", {}).items():
            require(digest(Path(path)) == checksum, "Linked library changed during qualification")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_unix"] = time.time()
        manifest["environment_finish"] = environment()
        manifest["completed_calls"] = len(rows)
        manifest["artifacts"] = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p != manifest_path}
        save()


if __name__ == "__main__":
    main()
