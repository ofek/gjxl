#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Serial, counterbalanced complete-workflow timing against qualified bytes."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from compare_reference_workflow import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.binary, args.validation, args.out = (p.resolve() for p in (args.binary, args.validation, args.out))
    args.out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((args.validation / "manifest.json").read_text())
    study = json.loads((args.validation / "corpus/manifest.json").read_text())
    controls = {}
    for image in study["inputs"]:
        name = image["name"]
        for variant in ("reference", "best", "all"):
            path = (args.validation / "controls" / name / "stock-e9.jxl" if variant == "reference" else
                    args.validation / "corpus" / (name + "-" + variant) / "output.jxl")
            require(digest(path) == previous["artifacts"][str(path)], "Changed qualified file")
            controls[name, variant] = digest(path)
        path = args.validation / "controls" / name / "input.pam"
        require(digest(path) == image["sha256"], "Changed original input")
    artifacts = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "Refusing to overwrite frozen timing file")
        subprocess.run(["/bin/cp", "-c", str(source), str(target)], check=True)
        artifacts[str(target)] = digest(target)
    reference = args.validation / "dependencies/reference-policy/reference_learning"
    require(digest(reference) == previous["artifacts"][str(reference)], "Changed reference executable")
    binaries = {}
    for label, source in (("reference", reference), ("native", args.binary)):
        target = args.out / (label + "_workflow")
        copy(source, target)
        binaries[label] = str(target)
    repo = Path(__file__).resolve().parents[2]
    # All encoding sources except the opt-in profiler and driver are the
    # qualified integration snapshot. Store the complete current source too.
    for folder in ("src", "include", "cmake", "tools/modular_reference_learning"):
        for source in sorted((repo / folder).rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                copy(source, args.out / "source" / source.relative_to(repo))
    copy(repo / "CMakeLists.txt", args.out / "source/CMakeLists.txt")
    for name in ("CMakeCache.txt", "compile_commands.json", "build.ninja", "libgjxl_codestream.a",
                 "libgjxl_codec.a", "libgjxl_gpu.a", "libgjxl_gpu_ops.a", "libgjxl_modular_reference.a"):
        copy(args.binary.parent / name, args.out / "build" / name)
    copy(args.binary.parent / "Testing/Temporary/LastTest.log", args.out / "build/LastTest.log")
    stages = ["input", "transforms", "training", "tokens", "model", "emission", "assembly"]
    manifest = {"schema": 1, "kind": "native-reference-workflow-timing",
        "validation_manifest": str(args.validation / "manifest.json"),
        "validation_manifest_sha256": digest(args.validation / "manifest.json"),
        "platform": platform.platform(), "source_revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True),
        "source_snapshot_is_authoritative": True, "binaries": binaries, "inputs": study["inputs"],
        "variants": ["reference", "best", "all"], "warmups": 1, "repeats": 3, "stages": stages,
        "scope": "Fresh serial processes, one warmup then three rotating-order repetitions per input/variant. "
                 "Native total includes selection, preparation, learning, coding, admission and prepared-frame teardown; "
                 "reference total is the stock e9 encoder API interval. Learning includes all scope preparation and fits, "
                 "plus native stage admission, validation and compaction. No input IO, independent source unpack, decode, "
                 "trace or output IO in measured intervals. No thread pools. Uncontrolled desktop, no concurrent agent "
                 "builds/tests/encodes. Per-stage native CPU is process CPU with nested time removed.",
        "artifacts": artifacts}
    (args.out / "manifest-start.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out / "results.jsonl").open("x", buffering=1) as ledger:
        for index, image in enumerate(study["inputs"]):
            name = image["name"]
            source = args.validation / "controls" / name / "input.pam"
            for iteration in range(4):
                offset = (index + iteration) % 3
                variants = manifest["variants"][offset:] + manifest["variants"][:offset]
                for variant in variants:
                    directory = args.out / f"{name}-{variant}-{iteration}"
                    stdout = args.out / (directory.name + ".stdout")
                    stderr = args.out / (directory.name + ".stderr")
                    if variant == "reference":
                        command = [binaries["reference"], "reference-policy", str(source), str(directory), "timing", "auto"]
                    else:
                        command = [binaries["native"], str(source), str(directory), "auto", variant, "timing"]
                    start = time.time()
                    with stdout.open("x") as out, stderr.open("x") as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True)
                    finish = time.time()
                    result = json.loads((stdout if variant == "reference" else directory / "result.json").read_text())
                    require(digest(directory / "output.jxl") == controls[name, variant], "Timed output differs from qualified bytes")
                    if variant == "reference":
                        require(result["decoded_exact"] and not result["learning"]["capture"] and
                                not result["learning"]["stream_trace"], "Invalid reference timing mode")
                    else:
                        require(result["schema"] == 2 and result["exact_decode"] and
                                result["total_includes_prepared_teardown"] and result["total"] and
                                not (directory / "workflow.json").exists(), "Invalid native timing mode")
                        expected = json.loads((args.validation / "corpus" / (name + "-" + variant) / "result.json").read_text())
                        require(result["stats"] == expected["stats"], "Timed workflow changed its learning work")
                        require(len(result["profile_cpu"]) == len(stages), "Incomplete native stage profile")
                        for kind in ("cpu", "wall"):
                            values = result["profile_" + kind]
                            require(min(values) >= -1e-6 and sum(values) <= result["total"][kind] + 1e-4,
                                    "Nested stage time exceeds complete interval")
                    row = {"input": name, "variant": variant, "iteration": iteration, "warmup": iteration == 0,
                           "command": command, "started_unix": start, "finished_unix": finish,
                           "output_sha256": controls[name, variant], "result": result}
                    ledger.write(json.dumps(row, sort_keys=True) + "\n")
                    elapsed = result["reference_call_cpu"] if variant == "reference" else result["total"]["cpu"]
                    print(name, variant, iteration, f"{elapsed:.6f} CPU seconds", flush=True)
    for path in sorted(args.out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[str(path)] = digest(path)
    with (args.out / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
