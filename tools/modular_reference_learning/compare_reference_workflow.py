#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Compare original-input native decisions to frozen stock e9 stream traces.

Compare samples directly, not just their hashes. Metadata shift -1 is normalized
by the driver from the native metadata role. Reference component labels are not
used by the learner and are deliberately outside this comparison. Coding event
order is irrelevant: streaming libjxl visits DC chunks, while native records are
in canonical stream-ID order.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compare(native_dir, reference_dir):
    native = json.loads((native_dir / "workflow.json").read_text())
    result = json.loads((native_dir / "result.json").read_text())
    require(result["exact_decode"], "Native file was not independently verified")
    require(result["stats"] == native["stats"], "Native result and trace disagree")
    records = {s["stream_id"]: s for s in native["streams"]}
    require(len(records) == len(native["streams"]), "Duplicate native stream IDs")
    source = reference_dir / "streams"
    coding = [json.loads(p.read_text()) for p in sorted(source.glob("coding-*.json"))]
    require(coding, "No reference coding streams")
    require(len({s["stream_id"] for s in coding}) == len(coding), "Duplicate reference coding IDs")
    for r in coding:
        n = records[r["stream_id"]]
        label = f"coding stream {r['stream_id']}"
        for field in ("width", "height", "metadata_channels", "local_tree", "transforms"):
            require(n[field] == r[field], f"{label}: {field} differs")
        require(len(n["channels"]) == r["channels"], f"{label}: channel count differs")
        for field in ("wp_mode", "wp_coefficients", "wp_weights"):
            require(n[field] == r["options"][field], f"{label}: {field} differs")
    require({r["stream_id"] for r in coding} == {k for k, n in records.items() if n["channels"]},
            "Active stream IDs differ")
    calls, fits, nodes, compared_bytes, compared_channels = 0, 0, 0, 0, 0
    native_learned = {k for k, n in records.items() if n["learning"]["stop"]}
    seen = set()
    native_pixels = native_dir / "pixels.i32le"
    require(native_pixels.stat().st_size == native["pixel_bytes"], "Truncated native pixel data")
    with native_pixels.open("rb") as pixels:
        for path in sorted(source.glob("learning-*.json")):
            r = json.loads(path.read_text())
            key = 0 if r["local_stream_id"] == -1 else r["local_stream_id"]
            require(key not in seen, "Repeated learning scope")
            seen.add(key)
            n = records[key]
            label = f"learning scope {key} ({path.name})"
            for field in ("start", "stop"):
                require(n["learning"][field] == r[field], f"{label}: {field} differs")
            require(n["tree"] == r["tree"], f"{label}: raw learned tree differs")
            fit_count = r["fit_end"] - r["fit_begin"]
            require(fit_count in (0, 1), "Reference scope did not use zero or one fit")
            require(fit_count == int(n["learning"]["selected_samples"] != 0),
                    f"{label}: fit count differs")
            calls += 1
            fits += fit_count
            nodes += len(r["tree"])
            raw_path = path.with_suffix(".i32le")
            require(raw_path.stat().st_size == r["pixel_bytes"], "Truncated reference pixels")
            with raw_path.open("rb") as raw:
                for im in r["images"]:
                    sid = im["index"] if r["local_stream_id"] == -1 else key
                    if sid not in records:
                        require(not im["channels"] and not im["width"] and not im["height"],
                                "Missing nonempty reference image")
                        continue
                    stream = records[sid]
                    for field in ("width", "height", "metadata_channels", "transforms"):
                        require(stream[field] == im[field], f"{label}, image {sid}: {field} differs")
                    for field in ("wp_mode", "wp_coefficients", "wp_weights"):
                        require(stream[field] == im["options"][field], f"{label}, image {sid}: {field} differs")
                    require(len(stream["channels"]) == len(im["channels"]), "Learning channel count differs")
                    for c, (nc, rc) in enumerate(zip(stream["channels"], im["channels"])):
                        for field in ("width", "height", "hshift", "vshift", "included", "bytes"):
                            require(nc[field] == rc[field], f"{label}, image {sid}, channel {c}: {field} differs")
                        raw.seek(rc["offset"])
                        pixels.seek(nc["offset"])
                        require(raw.read(rc["bytes"]) == pixels.read(nc["bytes"]),
                                f"{label}, image {sid}, channel {c}: transformed pixels differ")
                        compared_bytes += rc["bytes"]
                        compared_channels += 1
    require(seen == native_learned, "Native and reference learning scopes differ")
    s = native["stats"]
    require((calls, fits, calls - fits, nodes) ==
            (s["learning_calls"], s["fits"], s["empty_learning_calls"], s["tree_nodes"]),
            "Aggregated learning counters differ")
    require(compared_bytes == native["pixel_bytes"], "Not all native included pixels were compared")
    files = sorted(source.glob("*")) + [reference_dir / "output.jxl"] + sorted(native_dir.glob("*"))
    return {"native": str(native_dir.resolve()), "reference": str(reference_dir.resolve()),
            "learning_calls": calls, "fits": fits, "tree_nodes": nodes,
            "coding_streams": len(coding), "compared_channels": compared_channels,
            "compared_samples": compared_bytes // 4, "decisions_and_trees_exact": True,
            "native_bytes": result["bytes"], "reference_bytes": (reference_dir / "output.jxl").stat().st_size,
            "artifacts": {str(p.resolve()): digest(p) for p in files if p.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = compare(args.native, args.reference)
    if args.out:
        with args.out.open("x") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps({k: v for k, v in data.items() if k != "artifacts"}, sort_keys=True))


if __name__ == "__main__":
    main()
