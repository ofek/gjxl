#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Freeze and run the conditional uniform-WP control on the qualified corpus."""
import argparse
import json
import platform
import subprocess
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
    prior_path = args.validation / "manifest.json"
    prior = json.loads(prior_path.read_text())
    corpus_path = args.validation / "corpus/manifest.json"
    require(digest(corpus_path) == prior["artifacts"][str(corpus_path)], "Changed corpus manifest")
    corpus = json.loads(corpus_path.read_text())
    require(len({im["name"] for im in corpus["inputs"]}) == len(corpus["inputs"]), "Duplicate corpus input")
    artifacts = {}

    def copy(source, target, expected=None):
        if expected is not None:
            require(digest(source) == expected, "Changed control: " + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "Refusing to replace artifact")
        subprocess.run(["/bin/cp", "-c", str(source), str(target)], check=True)
        artifacts[str(target)] = digest(target)
        require(artifacts[str(target)] == digest(source), "Copy differs")

    binary = args.out / "reference_wp_sweep"
    copy(args.binary, binary)
    repo = Path(__file__).resolve().parents[2]
    for folder in ("src", "include", "cmake", "tools/modular_reference_learning"):
        for source in sorted((repo / folder).rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                copy(source, args.out / "source" / source.relative_to(repo))
    for name in ("CMakeLists.txt", "tests/modular_reference.cpp", "tests/modular_reference.h"):
        copy(repo / name, args.out / "source" / name)
    for name in ("CMakeCache.txt", "compile_commands.json", "build.ninja", "libgjxl_codestream.a",
                 "libgjxl_codec.a", "libgjxl_gpu.a", "libgjxl_gpu_ops.a", "libgjxl_modular_reference.a",
                 "Testing/Temporary/LastTest.log"):
        copy(args.binary.parent / name, args.out / "build" / name)
    for source in sorted((args.binary.parent / "reference-wp-sweep-test").rglob("*")):
        if source.is_file():
            copy(source, args.out / "tests" / source.relative_to(args.binary.parent / "reference-wp-sweep-test"))
    cache = dict(line.split("=", 1) for line in (args.binary.parent / "CMakeCache.txt").read_text().splitlines()
                 if line and not line.startswith(("#", "//")) and "=" in line)
    reference_build = Path(cache["GJXL_MODULAR_ORACLE_BUILD:PATH"])
    for name in ("lib/libjxl-internal.a", "lib/libjxl.dylib", "lib/libjxl_cms.dylib",
                 "third_party/highway/libhwy.a", "third_party/brotli/libbrotlienc.dylib",
                 "third_party/brotli/libbrotlidec.dylib", "third_party/brotli/libbrotlicommon.dylib"):
        previous = args.validation / "reference/build" / name
        copy(reference_build / name, args.out / "reference" / name, prior["artifacts"][str(previous)])
    for image in corpus["inputs"]:
        name = image["name"]
        for filename in ("input.pam", "stock-e9.jxl", "legacy-native.jxl"):
            source = args.validation / "controls" / name / filename
            copy(source, args.out / "controls" / name / filename, prior["artifacts"][str(source)])
        source = args.validation / "corpus" / (name + "-best/output.jxl")
        copy(source, args.out / "controls" / name / "qualified-best.jxl", prior["artifacts"][str(source)])
        require(digest(args.out / "controls" / name / "input.pam") == image["sha256"], "Wrong original input")
    manifest = {"schema": 1, "kind": "uniform-wp-ablation", "inputs": corpus["inputs"],
        "buffering": "auto", "predictors": "best", "modes": list(range(-1, 5)),
        "binary": str(binary), "prior_manifest": str(prior_path), "prior_manifest_sha256": digest(prior_path),
        "source_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True),
        "source_snapshot_is_authoritative": True, "platform": platform.platform(),
        "timing_qualified": False, "exhaustive_search_qualified": False,
        "scope": "Fixed selected palette/RCT/stream scope; Best learner and direct estimated ANS64 unchanged. "
                 "Replay selected per-stream WP vector, then override every WP-eligible stream uniformly with "
                 "each of five presets and refit every original learning scope. Selected vector is retained "
                 "as a candidate. Complete native files and independent exact decodes determine rate. "
                 "This does not enumerate independent per-stream assignments or other transforms/predictors/scopes. "
                 "Single-pass timing is diagnostic only; managed memory includes retained baseline and candidate.",
        "artifacts": artifacts}
    (args.out / "manifest-start.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out / "results.jsonl").open("x", buffering=1) as ledger:
        for image in corpus["inputs"]:
            name = image["name"]
            source = args.out / "controls" / name / "input.pam"
            output = args.out / name
            command = [str(binary), str(source), str(output), "auto", "best"]
            with (args.out / (name + ".log")).open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output / "result.json").read_text())
            require(result["exact_decode"] and result["selected_replay_exact"], "Incomplete sweep")
            require(digest(output / "baseline.jxl") == digest(args.out / "controls" / name / "qualified-best.jxl"),
                    "Baseline changed qualified bytes: " + name)
            ledger.write(json.dumps({"input": name, "command": command, "result": result}, sort_keys=True) + "\n")
            best = min(result["candidates"], key=lambda row: row["bytes"])
            print(name, result["baseline"]["bytes"], "->", best["bytes"], best["name"], flush=True)
    for source in sorted(args.out.rglob("*")):
        if source.is_file() and source.name != "manifest.json":
            artifacts[str(source)] = digest(source)
    with (args.out / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
