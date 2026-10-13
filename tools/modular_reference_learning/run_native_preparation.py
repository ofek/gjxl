#!/usr/bin/env python3
"""Qualify native preparation against frozen reference samples and encoded bytes."""
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
    parser.add_argument("--inputs", nargs="+")
    args = parser.parse_args()
    args.binary, args.corpus, args.output = (p.resolve() for p in (args.binary, args.corpus, args.output))
    corpus, rows, artifacts = load_study(args.corpus)
    inputs = {item["name"]: item for item in corpus["inputs"]}
    if args.inputs and set(args.inputs) - inputs.keys():
        parser.error("Unknown input")
    cases = []
    for row in rows:
        if row["iteration"] != -1 or row["variant"] == "reference":
            continue
        if args.inputs and row["input"] not in args.inputs:
            continue
        fixtures = [p for p in row["artifacts"] if p.endswith(".bin")]
        if len(fixtures) != 1:
            raise ValueError("Native canonical-stream bridge requires one global fixture")
        directory = Path(fixtures[0]).parents[1]
        case = {"input": row["input"], "variant": row["variant"], "fixture": fixtures[0],
                "policy": str(directory / "policy.txt"), "path": inputs[row["input"]]["path"],
                "expected_output": str(directory / "output.jxl"), "expected_sha256": row["output_sha256"],
                "expected_bytes": row["result"]["bytes"]}
        cases.append(case)
        for field in ("fixture", "policy", "path", "expected_output"):
            artifacts[case[field]] = sha(case[field])
    if not cases:
        parser.error("No cases")
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / "reference_learning"
    shutil.copy2(args.binary, binary)
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    paths = [p for p in source.iterdir() if p.suffix in {".h", ".cpp", ".py"} or p.name == "CMakeLists.txt"]
    paths += [repo / "CMakeLists.txt", repo / "LICENSE"]
    paths += list((repo / "src/codec/modular").glob("histogram_*.h"))
    paths += list((repo / "src/codec/modular").glob("histogram_*.cpp"))
    paths += list((repo / "tests").glob("modular_histogram*_test.cpp"))
    for path in paths:
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
        "REFERENCE_BUILD": ["lib/libjxl-internal.a", "third_party/highway/libhwy.a", "lib/libjxl_cms.dylib",
                            "third_party/brotli/libbrotlienc.dylib", "third_party/brotli/libbrotlidec.dylib",
                            "third_party/brotli/libbrotlicommon.dylib"],
        "GENERATED_SOURCE": ["provenance.json", "lib/jxl/modular/encoding/enc_ma.cc",
                             "lib/jxl/modular/encoding/enc_ma.h", "lib/jxl/modular/encoding/enc_encoding.cc",
                             "reference_codec.cpp"],
    }.items():
        root = Path(next(v for k, v in cache.items() if k.split(":")[0] == prefix))
        for name in names:
            artifacts[str(root / name)] = sha(root / name)
    artifacts[str(binary)] = sha(binary)
    manifest = {"schema": 1, "cases": cases, "artifacts": artifacts,
                "source_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "source_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True),
                "source_snapshot_is_authoritative": True,
                "timing_scope": "Single storage-tracked correctness run per case; diagnostic only. Preparation includes admission RNG, quantization, sample collection, deduplication and allocation. Fitting, transforms, stream planning, verification and encoding excluded."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "results.jsonl").open("x") as ledger:
        for case in cases:
            directory = args.output / f"{case['input']}-{case['variant']}"
            directory.mkdir()
            encoded = directory / "output.jxl"
            command = [str(binary), "prepare-check", case["path"], case["policy"], case["fixture"], case["variant"], str(encoded)]
            start = time.monotonic()
            result = subprocess.run(command, text=True, capture_output=True)
            (directory / "stdout.json").write_text(result.stdout)
            (directory / "stderr.log").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"Failed {case['input']} {case['variant']}: {result.stderr}")
            data = json.loads(result.stdout)
            if not all(data[k] is True for k in ("samples_exact", "tree_exact", "decoded_exact")):
                raise ValueError("Missing exact preparation/fit/decode check")
            if sha(encoded) != case["expected_sha256"] or data["bytes"] != case["expected_bytes"]:
                raise ValueError("Native preparation changed encoded bytes")
            record = {"input": case["input"], "variant": case["variant"], "command": command,
                      "elapsed_seconds": time.monotonic() - start, "result": data,
                      "output_sha256": sha(encoded), "artifacts": {str(p): sha(p) for p in directory.iterdir()}}
            ledger.write(json.dumps(record) + "\n"); ledger.flush()
            print(case["input"], case["variant"], data["bytes"], "exact", flush=True)


if __name__ == "__main__":
    main()
