#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture ordinary calls, entropy profiles and warmed token-backing replays.

Use a frozen Release benchmark and a run directory containing cases.tsv and
previously qualified expected/*.jxl. Does not stop other applications. Each
phase gets a fresh process; this is diagnostic attribution, not dispatch
qualification. Existing cohorts are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_gpu_performance import command


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--measure-seeds", type=int, nargs="*", default=[17])
    parser.add_argument("--profile-seeds", type=int, nargs="*", default=[137, 239])
    parser.add_argument("--replay-seeds", type=int, nargs="*", default=[181, 283])
    args = parser.parse_args()
    if not 0 < args.rounds <= 96 or args.rounds % 6:
        parser.error("Rounds must balance all six orders (6..96)")
    run, binary = args.run.resolve(), args.binary.resolve()
    phases = [(mode, seed) for mode in ("measure", "profile", "replay")
              for seed in getattr(args, mode + "_seeds")]
    if not phases or len(set(phases)) != len(phases):
        parser.error("Need distinct phases")
    record = run / "entropy-run-provenance.json"
    if record.exists() or any((run / f"{mode}-{seed}.csv").exists() for mode, seed in phases):
        raise FileExistsError("Refusing to overwrite an existing capture")
    env = os.environ.copy()
    removed = {}
    for key in list(env):
        if key.startswith(("MTL_", "METAL_")) or key in (
                "ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "DYLD_INSERT_LIBRARIES"):
            removed[key] = env.pop(key)
    env.update(MTL_DEBUG_LAYER="0", MTL_SHADER_VALIDATION="0")
    cases = [line.split("\t") for line in (run / "cases.tsv").read_text().splitlines()
             if line and not line.startswith("#")]
    if any(int(row[5]) != 1 for row in cases):
        raise ValueError("Entropy diagnostics require single-call cases")
    files = {str(Path(row[1]).resolve()): sha(Path(row[1])) for row in cases}
    files.update({str(run / "expected" / (row[0] + ".jxl")):
                  sha(run / "expected" / (row[0] + ".jxl")) for row in cases})
    provenance = dict(binary=str(binary), binary_sha256=sha(binary),
                      manifest_sha256=sha(run / "cases.tsv"), inputs_and_expected=files,
                      rounds=args.rounds, warmups=2, commands=[],
                      started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      removed_environment=removed, measurement_environment={k: v for k, v in env.items()
                                                                          if k.startswith("MTL_")},
                      system=[command("sw_vers"), command("sysctl", "hw.memsize", "hw.ncpu",
                                                                        "machdep.cpu.brand_string"),
                              command("pmset", "-g", "batt"), command("pmset", "-g", "therm"),
                              command("otool", "-L", str(binary))])
    record.write_text(json.dumps(provenance, indent=2) + "\n")
    with (run / "entropy-host-activity.jsonl").open("x") as activity:
        for mode, seed in phases:
            cmd = [str(binary), mode, str(run / "cases.tsv"), str(run), str(args.rounds),
                   "2", str(seed), "tokens"]
            entry = dict(argv=cmd, mode=mode, seed=seed, started=time.time())
            provenance["commands"].append(entry)
            with (run / f"{mode}-{seed}.log").open("x") as log:
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
                    raise RuntimeError(f"Failed {mode}-{seed}; see log")
            print(f"Finished {mode}-{seed}", flush=True)
    if sha(binary) != provenance["binary_sha256"] or any(sha(Path(p)) != h for p, h in files.items()):
        raise RuntimeError("Capture input, expected output or binary changed")
    provenance["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
