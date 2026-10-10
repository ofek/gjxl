#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run fresh-process paired cohorts, retaining host and binary provenance.

Run qualify separately with Metal validation before using this driver. This
driver disables validation/profiling instrumentation for ordinary timings and
logs, but does not terminate, other processes. Inspect host activity before
accepting performance claims. Three even-sized cohorts balance CPU/GPU order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return dict(command=list(args), returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 43, 91])
    parser.add_argument("--tokens", action="store_true", help="Measure all three routes in balanced triplets")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        parser.error("Use an even number of pairs (at least two)")
    if args.tokens and args.rounds % 6:
        parser.error("Token comparison requires rounds divisible by six to balance all route orders")
    run, binary = args.run.resolve(), args.binary.resolve()
    env = os.environ.copy()
    removed = {}
    for key in list(env):
        if key.startswith(("MTL_", "METAL_")) or key in (
                "ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "DYLD_INSERT_LIBRARIES"):
            removed[key] = env.pop(key)
    # Explicitly disable layers, including a possible inherited Xcode default.
    env.update(MTL_DEBUG_LAYER="0", MTL_SHADER_VALIDATION="0")
    provenance = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      removed_environment=removed,
                      measurement_environment={k: v for k, v in env.items() if k.startswith("MTL_")},
                      commands=[], system=[command("sw_vers"),
                      command("sysctl", "hw.memsize", "hw.ncpu", "machdep.cpu.brand_string"),
                      command("pmset", "-g", "batt"), command("pmset", "-g", "custom"),
                      command("pmset", "-g", "therm"), command("otool", "-L", str(binary))])
    record = run / "run-provenance.json"
    record.write_text(json.dumps(provenance, indent=2) + "\n")
    phases = [("measure", seed, args.rounds) for seed in args.seeds] + [("profile", 137, 6 if args.tokens else 4)]
    with (run / "host-activity.jsonl").open("w") as activity:
        for mode, seed, rounds in phases:
            cmd = [str(binary), mode, str(run / "cases.tsv"), str(run), str(rounds), "2", str(seed)] + (["tokens"] if args.tokens else [])
            if (run / f"{mode}-{seed}.csv").exists():
                raise FileExistsError(f"Refusing to overwrite cohort {mode}-{seed}")
            entry = dict(argv=cmd, started=time.time())
            provenance["commands"].append(entry)
            with (run / f"{mode}-{seed}.log").open("w") as log:
                with subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env) as child:
                    while True:
                        activity.write(json.dumps(dict(time=time.time(), mode=mode, seed=seed,
                            pid=child.pid, processes=command("ps", "-axo", "pid,ppid,pcpu,comm"),
                            thermal=command("pmset", "-g", "therm"))) + "\n")
                        activity.flush()
                        try:
                            code = child.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                entry.update(finished=time.time(), returncode=code)
                record.write_text(json.dumps(provenance, indent=2) + "\n")
                if code:
                    raise RuntimeError(f"Cohort failed: {mode}-{seed}, see log")
            print(f"Finished {mode}-{seed}", flush=True)
    provenance["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance["system"].extend([command("pmset", "-g", "batt"), command("pmset", "-g", "therm")])
    record.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
