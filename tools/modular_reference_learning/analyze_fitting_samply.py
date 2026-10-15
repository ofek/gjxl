#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Resolve native fitting stacks and inline hotspots from a Samply capture.

Requires a presymbolicated sidecar and a matching binary/dSYM. Attribution is
sampled main-thread CPU, never wall time. Current/previous delta assignments
are both reported. Inline source symbols come from llvm-symbolizer.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import subprocess

from run_hybrid_qualification import digest, require


def resolve(profile, symbols, thread):
    maps = {}
    for module in symbols["data"]:
        maps[str(module.get("code_id", "")).upper()] = {
            int(address): symbols["string_table"][module["symbol_table"][index]["symbol"]]
            for address, index in module["known_addresses"]}
    frames = []
    for i in range(thread["frameTable"]["length"]):
        func = thread["frameTable"]["func"][i]
        name = thread["stringArray"][thread["funcTable"]["name"][func]]
        resource = thread["funcTable"]["resource"][func]
        lib = thread["resourceTable"]["lib"][resource] if resource is not None and resource >= 0 else None
        address = thread["frameTable"]["address"][i]
        if lib is not None and lib >= 0:
            name = maps.get(str(profile["libs"][lib].get("codeId", "")).upper(), {}).get(address, name)
        frames.append({"name": name, "lib": lib, "address": address})
    return frames


def sample_weights(deltas, previous):
    values = [v or 0 for v in deltas]
    require(all(v >= 0 for v in values), "Negative sample CPU delta")
    if not previous or not values:
        return values
    return [values[0] + (values[1] if len(values) > 1 else 0)] + values[2:] + ([0] if len(values) > 1 else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbolizer", default="llvm-symbolizer")
    args = parser.parse_args()
    with gzip.open(args.profile, "rt") as stream:
        profile = json.load(stream)
    sidecar = Path(str(args.profile).removesuffix(".json.gz") + ".json.syms.json")
    symbols = json.loads(sidecar.read_text())
    require(profile["meta"]["sampleUnits"]["threadCPUDelta"] == "µs", "Unknown CPU sample units")
    threads = [t for t in profile["threads"] if t.get("isMainThread")]
    require(len(threads) == 1, "Expected one captured main thread")
    thread = threads[0]
    frames = resolve(profile, symbols, thread)
    stack_table = thread["stackTable"]
    stacks = {}
    for index in set(thread["samples"]["stack"]):
        stack, seen, current = [], set(), index
        while current is not None:
            require(current not in seen, "Cyclic stack table")
            seen.add(current)
            stack.append(frames[stack_table["frame"][current]])
            current = stack_table["prefix"][current]
        stacks[index] = stack
    inside = {index: any("gjxl::modular_internal::" in f["name"] and
                        ("LearnHistogramTree(" in f["name"] or "FindBestSplit<" in f["name"])
                        for f in stack) for index, stack in stacks.items()}
    addresses = sorted({f["address"] for index, stack in stacks.items() if inside[index]
                        for f in stack if f["lib"] is not None and f["lib"] >= 0 and
                        profile["libs"][f["lib"]]["name"] == args.binary.name})
    require(addresses, "No native fitter addresses resolved")
    # Samply stores Mach-O image-relative addresses. llvm-symbolizer's
    # --relative-address does not apply the Mach-O __TEXT slide here.
    load_commands = subprocess.check_output(["otool", "-l", str(args.binary)], text=True)
    match = re.search(r"segname __TEXT\s+vmaddr (0x[0-9a-fA-F]+)", load_commands)
    require(match is not None, "Cannot identify Mach-O image base")
    image_base = int(match.group(1), 16)
    command = [args.symbolizer, "--inlines", "--output-style=JSON",
               "--obj=" + str(args.binary.resolve())]
    output = subprocess.check_output(command, input="".join(f"0x{a+image_base:x}\n" for a in addresses), text=True)
    resolved = [json.loads(line) for line in output.splitlines()]
    require(len(resolved) == len(addresses), "Symbolizer lost addresses")
    inline = dict(zip(addresses, resolved))
    result = {"scope": "Sampled main-thread CPU inside native LearnHistogramTree/FindBestSplit only. "
                       "Flat rows are disjoint; inclusive inline-function rows overlap. "
                       "Both current and previous CPU-delta assignment are retained.",
              "profile_sha256": digest(args.profile), "sidecar_sha256": digest(sidecar),
              "binary_sha256": digest(args.binary), "symbolizer_command": command, "image_base": image_base,
              "inline_symbols": resolved, "attribution": {}}
    for previous in [False, True]:
        weights = sample_weights(thread["samples"]["threadCPUDelta"], previous)
        require(len(weights) == thread["samples"]["length"], "Wrong sample weights")
        flat, inclusive, lines = Counter(), Counter(), Counter()
        cpu = samples = resolved_cpu = entropy_cpu = native_cpu = inline_cpu = 0
        for index, weight in zip(thread["samples"]["stack"], weights):
            if not inside[index] or weight <= 0:
                continue
            cpu += weight
            samples += 1
            stack = stacks[index]
            names = set(f["name"] for f in stack)
            leaf = stack[0]
            native = (leaf["lib"] is not None and leaf["lib"] >= 0 and
                      profile["libs"][leaf["lib"]]["name"] == args.binary.name)
            info = inline.get(leaf["address"], {}).get("Symbol", []) if native else []
            if native:
                native_cpu += weight
                if info and info[0].get("FileName") and info[0].get("Line", 0) > 0:
                    inline_cpu += weight
            leaf_name = (info[0].get("FunctionName") if info else None) or leaf["name"]
            flat[leaf_name or leaf["name"]] += weight
            if info:
                for item in info:
                    if item.get("FunctionName"):
                        names.add(item["FunctionName"])
                loc = info[0]
                lines[f"{loc.get('FileName', '')}:{loc.get('Line', 0)}"] += weight
            if leaf_name and not leaf_name.startswith("0x"):
                resolved_cpu += weight
            if any("EstimateBits(" in name or "FastLog2(" in name for name in names):
                entropy_cpu += weight
            for name in names:
                inclusive[name] += weight
        require(cpu > 0 and sum(flat.values()) == cpu, "Fitting sample conservation failed")
        require(resolved_cpu / cpu >= 0.95, "Insufficient resolved fitting CPU")
        require(native_cpu and inline_cpu / native_cpu >= 0.95,
                "Insufficient inline/source resolution; check matching dSYM and image base")
        rows = lambda counter: [{"name": name, "cpu_ms": weight/1000, "percent": 100*weight/cpu}
                                for name, weight in counter.most_common(50)]
        result["attribution"]["previous" if previous else "current"] = {
            "fitter_cpu_ms": cpu/1000, "positive_samples": samples,
            "resolved_leaf_percent": 100*resolved_cpu/cpu,
            "inline_native_resolution_percent": 100*inline_cpu/native_cpu,
            "entropy_arithmetic_inclusive_percent": 100*entropy_cpu/cpu,
            "flat": rows(flat), "inclusive": rows(inclusive), "leaf_source_lines": rows(lines)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {n: v for n, v in x.items() if not isinstance(v, list)}
                      for k, x in result["attribution"].items()}, indent=2))


if __name__ == "__main__":
    main()
