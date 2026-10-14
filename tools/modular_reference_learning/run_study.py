#!/usr/bin/env python3
"""Run serial reference/bridge controls with immutable inputs and artifact hashes."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--ans-study", type=Path, required=True)
    parser.add_argument("--reference-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", nargs="*", help="Default: full saved 16-input corpus")
    parser.add_argument("--variants", nargs="+", choices=["reference", "best", "all"], default=["reference", "best"])
    parser.add_argument("--timing-repeats", type=int, default=0)
    args = parser.parse_args()
    if args.timing_repeats < 0 or args.timing_repeats > 20:
        parser.error("timing-repeats must be in [0, 20]")
    for name in ("binary", "generated", "ans_study", "reference_study", "output"):
        setattr(args, name, getattr(args, name).resolve())
    source = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    # Preserve the executable used by this ledger even if the build is rebuilt.
    frozen_binary = args.output / "reference_learning"
    if frozen_binary.exists():
        if sha(frozen_binary) != sha(args.binary):
            raise ValueError("Changed binary: use a new output directory")
    else:
        shutil.copy2(args.binary, frozen_binary)
    old_manifest = args.ans_study / "search/manifest.json"
    inputs = [c["input"] for c in json.loads(old_manifest.read_text())["cases"]]
    if args.inputs:
        requested = set(args.inputs)
        if requested - {item["name"] for item in inputs}:
            parser.error("Unknown input name")
        inputs = [item for item in inputs if item["name"] in requested]
    baseline = {r["input"]: r["estimated"] for r in json_lines(args.ans_study / "search/results.jsonl")}
    reference = {r["input"]: r for r in json_lines(args.reference_study / "results.jsonl") if r["config"] == "base"}
    policies = {item["name"]: args.ans_study / "fixed" / (item["name"] + "-learned") / "policy.txt" for item in inputs}
    for item in inputs:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Changed input: {item['name']}")
        old = baseline[item["name"]]
        if sha(old["path"]) != old["sha256"]:
            raise ValueError("Changed native control")
        if item["name"] in reference:
            old_ref = reference[item["name"]]
            if sha(old_ref["output"]) != old_ref["output_sha256"]:
                raise ValueError("Changed reference control")
    artifacts = [frozen_binary, args.generated / "provenance.json", old_manifest,
                 args.ans_study / "search/results.jsonl", args.reference_study / "results.jsonl"]
    artifacts += sorted(source.glob("*.cpp")) + sorted(source.glob("*.h")) + sorted(source.glob("*.py")) + [source / "CMakeLists.txt"]
    artifacts += list(policies.values())
    # Bind the linked archive contents, not just their directory names.
    cmake_cache = args.binary.parent / "CMakeCache.txt"
    cache = dict(line.split("=", 1) for line in cmake_cache.read_text().splitlines() if line and not line.startswith(("#", "//")) and "=" in line)
    for prefix, names in {
        "GJXL_BUILD": ["libgjxl_codestream.a", "libgjxl_codec.a", "libgjxl_gpu_ops.a", "libgjxl_gpu.a"],
        "REFERENCE_BUILD": ["lib/libjxl-internal.a", "third_party/highway/libhwy.a"],
    }.items():
        root = Path(next(value for key, value in cache.items() if key.split(":")[0] == prefix))
        artifacts += [root / name for name in names]
    manifest = {
        "schema": 1, "inputs": inputs, "variants": args.variants,
        "timing_repeats": args.timing_repeats, "timing_warmups": int(args.timing_repeats > 0),
        "source_revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "artifacts": {str(path): sha(path) for path in artifacts},
        "native_controls": {i["name"]: baseline[i["name"]] for i in inputs},
        "reference_controls": {i["name"]: reference[i["name"]] for i in inputs if i["name"] in reference},
        "timing_scope": "Serial diagnostic process CPU and wall; uncontrolled desktop. Capture calls excluded from learning-time comparisons.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Changed manifest: use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    ledger = args.output / "results.jsonl"
    done = {}
    if ledger.exists():
        for row in json_lines(ledger):
            for path, digest in row["artifacts"].items():
                if sha(path) != digest:
                    raise ValueError(f"Changed completed artifact: {path}")
            done[(row["input"], row["variant"], row["iteration"])] = row
    with ledger.open("a", buffering=1) as log:
        for index, item in enumerate(inputs):
            # Capture, then interleaved warmup/measurement rounds. All processes serial.
            for iteration in range(-1, args.timing_repeats + 1 if args.timing_repeats else 0):
                variants = args.variants[index % len(args.variants):] + args.variants[:index % len(args.variants)]
                for variant in variants:
                    key = (item["name"], variant, iteration)
                    if key in done:
                        continue
                    case = args.output / f"{item['name']}-{variant}-{iteration}"
                    case.mkdir(exist_ok=True)
                    # A partial attempt must be inspected, never silently overwritten.
                    if any(case.iterdir()):
                        raise ValueError(f"Incomplete case requires inspection: {case}")
                    recording = "capture" if iteration == -1 else "timing"
                    cmd = [str(frozen_binary), "reference", item["path"], str(case), recording] if variant == "reference" else [
                        str(frozen_binary), "bridge", item["path"], str(policies[item["name"]]), str(case), recording, variant]
                    started = time.time()
                    with (case / "stdout.json").open("w") as stdout, (case / "stderr.log").open("w") as stderr:
                        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)
                    result = json.loads((case / "stdout.json").read_text())
                    if not result["decoded_exact"] or (variant != "reference" and not result["tokens_exact"]):
                        raise ValueError("Failed bridge correctness control")
                    output = case / "output.jxl"
                    digest = sha(output)
                    if result["bytes"] != output.stat().st_size:
                        raise ValueError("Size mismatch")
                    if variant == "reference" and item["name"] in reference and digest != reference[item["name"]]["output_sha256"]:
                        raise ValueError("Reference file differs from saved stock e9")
                    if iteration >= 0 and digest != done[(item["name"], variant, -1)]["output_sha256"]:
                        raise ValueError("Timing and capture files differ")
                    if iteration == -1 and not all(fit["replayed"] for fit in result["learning"]["fits"]):
                        raise ValueError("Missing capture replay")
                    row = {"input": item["name"], "variant": variant, "iteration": iteration,
                           "warmup": iteration == 0, "output_sha256": digest,
                           "command": cmd, "started_unix": started, "finished_unix": time.time(),
                           "result": result, "artifacts": {str(p): sha(p) for p in sorted(case.rglob("*")) if p.is_file()}}
                    log.write(json.dumps(row) + "\n")
                    done[key] = row
                    print(item["name"], variant, iteration, result["bytes"], flush=True)


if __name__ == "__main__":
    main()
