#!/usr/bin/env python3
"""Bounded, paired complete-call check of entropy, selection or CPU tokenization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment():
    result = {"unix": time.time(), "platform": platform.platform(), "load": os.getloadavg()}
    for key, command in {"power": ["pmset", "-g", "batt"],
                         "thermal": ["pmset", "-g", "therm"],
                         "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"]}.items():
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        result[key] = run.stdout.strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--route", choices=["cpu", "shared-cpu", "hybrid", "auto"], default="cpu")
    parser.add_argument("--phase", choices=["entropy", "selection", "tokenization"], default="entropy")
    args = parser.parse_args()
    if not 2 <= args.threads <= 256 or args.repetitions < 1:
        parser.error("threads must be 2..256 and repetitions positive")
    binary, source, output = args.binary.resolve(), args.input.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    manifest = {"schema": 1, "status": "running", "input": str(source),
                "input_sha256": digest(source), "binary": str(binary),
                "binary_sha256": digest(binary), "threads": args.threads,
                "repetitions": args.repetitions, "route": args.route, "phase": args.phase,
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "stages": ["input", "transforms", "training", "tokens", "model", "emission",
                           "assembly", "gpu_input_tokens", "stream_encoding"],
                "boundary": "Complete encoder call; input loading, backend setup, independent decode and file I/O excluded. "
                            f"Learning and CPU ceiling fixed; only {args.phase} participation changes. "
                            "One unmeasured warmup per mode; alternating measured order; active desktop.",
                "runs": []}
    patch = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=repo)
    for name in subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True).splitlines():
        diff = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", name], cwd=repo, capture_output=True)
        if diff.returncode not in (0, 1):
            raise RuntimeError(diff.stderr.decode())
        patch += diff.stdout
    (output / "source.diff").write_bytes(patch)
    manifest["source_diff_sha256"] = digest(output / "source.diff")
    representative = None
    try:
        for repetition in range(-1, args.repetitions):
            modes = [1, args.threads] if repetition % 2 == 0 else [args.threads, 1]
            for participants in modes:
                name = f"{'warmup' if repetition < 0 else f'rep-{repetition}'}-{args.phase}-{participants}"
                directory = output / name
                command = [str(binary), str(source), str(directory), args.route, "auto", "best", "fit-off",
                           str(args.threads), str(args.threads),
                           str(participants if args.phase == "entropy" else args.threads)]
                if args.phase in ("selection", "tokenization"):
                    command.append(str(participants if args.phase == "selection" else args.threads))
                if args.phase == "tokenization":
                    command.append(str(participants))
                before = environment()
                with open(output / f"{name}.log", "w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                after = environment()
                result = json.loads((directory / "result.json").read_text())
                assert result["schema"] == {"entropy": 3, "selection": 4, "tokenization": 5}[args.phase] and result["exact_decode"]
                assert len(result["profile_wall"]) == 9
                assert all(value >= -1e-6 for value in result["profile_wall"])
                encoded = directory / "output.jxl"
                encoded_hash = digest(encoded)
                if representative is None:
                    representative = encoded
                    manifest["output_sha256"] = encoded_hash
                    manifest["output_artifact"] = str(encoded.relative_to(output))
                else:
                    assert encoded_hash == manifest["output_sha256"], "Parallelism changed complete bytes"
                    # Each run already independently decoded its own output.
                    # Retain one identical compressed artifact to save storage.
                    encoded.unlink()
                record = {"repetition": repetition, "measured": repetition >= 0, "participants": participants,
                          "command": command, "environment_before": before, "environment_after": after,
                          "output_sha256": encoded_hash, "result": result}
                manifest["runs"].append(record)
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(name, result["total"]["wall"], "seconds; identical bytes and exact decode", flush=True)
        manifest["medians"] = {}
        for participants in [1, args.threads]:
            runs = [r["result"] for r in manifest["runs"] if r["measured"] and r["participants"] == participants]
            manifest["medians"][str(participants)] = {
                "complete_seconds": statistics.median(r["total"]["wall"] for r in runs),
                {"selection": "selection_seconds", "tokenization": "tokenization_seconds",
                 "entropy": "model_and_emission_seconds"}[args.phase]: statistics.median(
                    r["profile_wall"][1] if args.phase == "selection" else
                    r["profile_wall"][3] if args.phase == "tokenization" else
                    r["profile_wall"][4] + r["profile_wall"][5] + r["profile_wall"][8] for r in runs),
                "peak_backing_bytes": max(r["memory"]["peak_backing_bytes"] for r in runs),
                "peak_committed_bytes": max(r["memory"]["peak_committed_bytes"] for r in runs)}
        assert digest(binary) == manifest["binary_sha256"]
        assert digest(source) == manifest["input_sha256"]
        manifest["status"] = "complete"
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
