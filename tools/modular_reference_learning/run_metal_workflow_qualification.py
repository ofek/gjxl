#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Freeze and serially capture learned Metal qualification (no third-party Python)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def command(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def environment():
    processes = command(["ps", "-Ao", "pid,pcpu,comm", "-r"])
    processes["stdout"] = "\n".join(processes["stdout"].splitlines()[:30]) + "\n"
    return {"time_unix": time.time(), "power": command(["pmset", "-g", "batt"]),
            "thermal": command(["pmset", "-g", "therm"]), "processes": processes}


def clone(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Darwin":
        subprocess.run(["/bin/cp", "-c", str(source), str(dest)], check=True)
    else:
        shutil.copy2(source, dest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="JSON with inputs [{name,path,sha256,modes:[encode,features]}]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encode-repeats", type=int, default=4)
    parser.add_argument("--feature-repeats", type=int, default=6)
    parser.add_argument("--cpu-participants", type=int, default=1)
    parser.add_argument("--public-only", action="store_true",
                        help="Encode only public CPU and explicit Metal; omit forced producers")
    args = parser.parse_args()
    if args.encode_repeats < 1 or args.feature_repeats < 1:
        parser.error("Repetitions must be positive")
    if not 1 <= args.cpu_participants <= 256:
        parser.error("CPU participants must be in 1..256")
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    requested = json.loads(args.inputs.read_text())["inputs"]
    names = [x["name"] for x in requested]
    if len(set(names)) != len(names) or any(Path(n).name != n for n in names):
        raise RuntimeError("Input names must be unique path components")
    binary = out / "gjxl_modular_metal_workflow_profile"
    clone(args.binary.resolve(), binary)
    captures = []
    manifest = {
        "schema": 2, "complete": False,
        "cpu_ceiling": args.cpu_participants,
        "encode_variants": ["cpu", "metal-auto"] if args.public_only else
                           ["cpu", "metal-auto", "metal-cpu", "metal-gpu"],
        "source_revision": command(["git", "rev-parse", "HEAD"], cwd=root),
        "merge_parent": command(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd=root),
        "source_status": command(["git", "status", "--short", "--branch"], cwd=root),
        "platform": platform.platform(),
        "system": command(["sysctl", "hw.model", "machdep.cpu.brand_string", "hw.memsize",
                           "hw.physicalcpu", "hw.logicalcpu"]),
        "os": command(["sw_vers"]), "binary_sha256": digest(binary),
        "dynamic_libraries": command(["otool", "-L", str(binary)]),
        "input_manifest": str(args.inputs.resolve()), "input_manifest_sha256": digest(args.inputs),
        "encode_repeats": args.encode_repeats, "feature_repeats": args.feature_repeats,
        "scope": "Serial processes, one input per process. Public CPU/Metal-auto calls; forced "
                 "producers mirror the public wrapper. First public CPU/Metal calls are cold "
                 "controls, then one warmup and rotating-order measured rounds. Encoder timer "
                 "includes admission, source preparation, selection, features, fitting, tokens, "
                 "entropy, frame cleanup and output publication. Source IO, domain construction, "
                 "verification and output IO excluded. Operation timer includes allocation, "
                 "copy, predictor computation, dispatch, wait and publication; selected channels, "
                 "backend setup, admission and output destruction outside. Managed peaks exclude "
                 "borrowed inputs, published output, immutable backend, driver and oracle storage. "
                 "Bounded feature batches; standalone feature trials retain all batch owners "
                 "through comparison. Explicit CPU ceiling recorded per call. "
                 "Active desktop; no concurrent qualification/build/test processes launched here.",
        "inputs": [], "sources": [], "captures": captures,
    }
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "include", "cmake", "CMakeLists.txt",
         "tools/modular_reference_learning", "tests/modular_reference.cpp",
         "tests/modular_reference.h"], cwd=root, text=True).splitlines()
    # Include this qualification's new untracked sources, never unrelated files.
    tracked += ["tools/modular_reference_learning/metal_workflow_profile.cpp",
                "tools/modular_reference_learning/run_metal_workflow_qualification.py",
                "tools/modular_reference_learning/report_metal_workflow_qualification.py"]
    for relative in sorted(set(tracked)):
        source = root / relative
        clone(source, out / "source" / relative)
        manifest["sources"].append({"path": relative, "sha256": digest(source)})
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    (out / "source.patch").write_bytes(patch)
    manifest["source_patch_sha256"] = digest(out / "source.patch")
    for filename in ["CMakeCache.txt", "compile_commands.json", "build.ninja"]:
        source = args.binary.resolve().parent / filename
        if source.exists():
            clone(source, out / "build-config" / filename)
    for item in requested:
        source = Path(item["path"]).resolve()
        actual = digest(source)
        if actual != item["sha256"]:
            raise RuntimeError(f"Changed input: {source}")
        frozen = out / "inputs" / (item["name"] + source.suffix)
        clone(source, frozen)
        manifest["inputs"].append(dict(item, frozen_path=str(frozen)))
    write(out / "manifest.json", manifest)
    try:
        # Complete full-encoder evidence first; feature isolation follows without
        # overlapping workloads. Every child independently checks its results.
        for mode in ["encode", "features"]:
            for item in manifest["inputs"]:
                if mode not in item["modes"]:
                    continue
                name = item["name"]
                directory = out / mode / name
                directory.parent.mkdir(exist_ok=True)
                repeats = args.encode_repeats if mode == "encode" else args.feature_repeats
                selected_mode = "encode-public" if mode == "encode" and args.public_only else mode
                argv = [str(binary), item["frozen_path"], str(directory), str(repeats),
                        selected_mode, str(args.cpu_participants)]
                capture = {"name": name, "mode": mode, "argv": argv, "before": environment()}
                start = time.monotonic()
                print(f"START {mode} {name}", flush=True)
                with (directory.parent / (name + ".stdout")).open("w") as stdout, \
                     (directory.parent / (name + ".stderr")).open("w") as stderr:
                    child = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
                capture.update(returncode=child.returncode, wall_seconds=time.monotonic() - start,
                               after=environment())
                captures.append(capture)
                if child.returncode == 0:
                    result = directory / ("encodes.jsonl" if mode == "encode" else "features.jsonl")
                    rows = [json.loads(line) for line in result.read_text().splitlines()]
                    expected = (2 + len(manifest["encode_variants"]) * (1 + repeats)
                                if mode == "encode" else 3 * (1 + repeats))
                    if len(rows) != expected or not all(r["released"] for r in rows):
                        raise RuntimeError(f"Incomplete capture: {mode} {name}")
                    capture.update(rows=len(rows), result_sha256=digest(result))
                    if mode == "encode":
                        capture["output_sha256"] = digest(directory / "output.jxl")
                write(out / "manifest.json", manifest)
                if child.returncode:
                    raise RuntimeError(f"Failed capture: {mode} {name}; see retained stderr")
                print(f"DONE {mode} {name} {capture['wall_seconds']:.2f}s", flush=True)
        manifest["complete"] = True
    finally:
        manifest["finished_environment"] = environment()
        write(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
