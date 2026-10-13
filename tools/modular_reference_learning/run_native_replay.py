#!/usr/bin/env python3
"""Verify native trees or time paired fitters on the immutable reference corpus."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from analyze import load_study, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["verify", "paired"], default="verify")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--variants", nargs="+", choices=["reference", "best", "all"])
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be in [1, 20]")
    args.binary, args.corpus, args.output = (
        path.resolve() for path in (args.binary, args.corpus, args.output))
    _, rows, corpus_artifacts = load_study(args.corpus)
    if args.inputs:
        if set(args.inputs) - {r["input"] for r in rows}:
            parser.error("Unknown input")
        rows = [r for r in rows if r["input"] in args.inputs]
    if args.variants:
        rows = [r for r in rows if r["variant"] in args.variants]
    fixtures = []
    for row in rows:
        if row["iteration"] != -1:
            continue
        paths = sorted((p for p in row["artifacts"] if p.endswith(".bin")),
                       key=lambda p: int(Path(p).stem.split("-")[-1]))
        for index, path in enumerate(paths):
            fixtures.append({"input": row["input"], "variant": row["variant"],
                             "fit": index, "path": path,
                             "sha256": row["artifacts"][path]})
    if not fixtures:
        parser.error("No selected fixtures")
    # New output only: partial failures remain available for inspection.
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / "reference_learning"
    shutil.copy2(args.binary, binary)
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    sources = sorted(source.glob("*.cpp")) + sorted(source.glob("*.h")) + sorted(source.glob("*.py"))
    sources += [source / "CMakeLists.txt", repo / "CMakeLists.txt", repo / "LICENSE",
                repo / "src/codec/modular/histogram_tree_learning.h",
                repo / "src/codec/modular/histogram_tree_learning.cpp",
                repo / "tests/modular_histogram_learning_test.cpp"]
    artifacts = {str(binary): sha(binary), **corpus_artifacts}
    for path in sources:
        target = args.output / "source" / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        artifacts[str(target)] = sha(target)
    for name in ("CMakeCache.txt", "compile_commands.json"):
        target = args.output / name
        shutil.copy2(args.binary.parent / name, target)
        artifacts[str(target)] = sha(target)
    cache = dict(line.split("=", 1) for line in (args.binary.parent / "CMakeCache.txt").read_text().splitlines()
                 if line and not line.startswith(("#", "//")) and "=" in line)
    for prefix, names in {
        "GJXL_BUILD": ["libgjxl_codestream.a", "libgjxl_codec.a", "libgjxl_gpu_ops.a", "libgjxl_gpu.a"],
        "REFERENCE_BUILD": ["lib/libjxl-internal.a", "third_party/highway/libhwy.a"],
        "GENERATED_SOURCE": ["provenance.json", "lib/jxl/modular/encoding/enc_ma.cc",
                             "lib/jxl/modular/encoding/enc_ma.h"],
    }.items():
        root = Path(next(v for k, v in cache.items() if k.split(":")[0] == prefix))
        for name in names:
            artifacts[str(root / name)] = sha(root / name)
    manifest = {
        "schema": 1, "mode": args.mode, "fixtures": fixtures, "artifacts": artifacts,
        "repeats": args.repeats if args.mode == "paired" else 1,
        "warmups_per_fitter": int(args.mode == "paired"),
        "source_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True),
        "source_snapshot_is_authoritative": True,
        "timing_scope": "Same process, identical quantized rows and parameters; serial fits, alternating reference/native order after one warmup each. Copies/conversions excluded. Native validation and fitting allocations included. Uncontrolled desktop.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "results.jsonl").open("x", buffering=1) as ledger:
        for fixture in fixtures:
            stem = f"{fixture['input']}-{fixture['variant']}-{fixture['fit']}"
            stdout, stderr = args.output / (stem + ".jsonl"), args.output / (stem + ".log")
            command = [str(binary), "paired-replay" if args.mode == "paired" else "native-verify", fixture["path"]]
            if args.mode == "paired":
                command.append(str(args.repeats))
            started = time.time()
            with stdout.open("x") as out, stderr.open("x") as err:
                subprocess.run(command, stdout=out, stderr=err, check=True)
            runs = [json.loads(line) for line in stdout.read_text().splitlines()]
            if len(runs) != manifest["repeats"] or not all(r["tree_exact"] for r in runs):
                raise ValueError("Missing native exact-tree replay")
            row = {**fixture, "runs": runs, "command": command,
                   "started_unix": started, "finished_unix": time.time(),
                   "artifacts": {str(p): sha(p) for p in (stdout, stderr)}}
            ledger.write(json.dumps(row) + "\n")
            print(stem, "exact", runs[0]["samples"], runs[0]["nodes"], flush=True)


if __name__ == "__main__":
    main()
