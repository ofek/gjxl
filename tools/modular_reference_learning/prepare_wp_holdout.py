#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Freeze the predeclared high-depth WP corpus, without running an encoder."""
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np

from compare_reference_workflow import digest, require


def synthetic(formula, width, height, channels, seed):
    y, x = np.indices((height, width), dtype=np.int64)
    output = np.empty((height, width, channels), dtype=np.uint16)
    if formula == "noise":
        output[:] = np.random.Generator(np.random.PCG64(seed)).integers(
            0, 65536, output.shape, dtype=np.uint16)
    else:
        for c in range(channels):
            if formula == "affine":
                a, b = ((3, 1), (1, 4), (2, 3), (4, 1))[c]
                value = (a * x * 65535 // (width - 1) + b * y * 65535 // (height - 1)) // (a + b)
            elif formula == "wrapped":
                value = (x * 701 + y * 389 + c * 12011 + x * y * (c + 1)) & 65535
            elif formula == "quadratic":
                value = ((x * x * 65535 // ((width - 1) ** 2) +
                          y * y * 65535 // ((height - 1) ** 2) + (c * 97 * x) % 1024) // 2) & 65535
            elif formula == "edges":
                value = np.where((x // 47 + y // 31) % 2,
                                 46000 + (x % 47) * 83 + c * 257,
                                 1700 + (y % 31) * 137 + c * 401)
            elif formula == "palette":
                index = (x // 11 + 3 * (y // 7)) % 193
                value = (index * (c + 1) * 313 + c * 5003) & 65535
                if c == 3:
                    value = np.where(index % 7 == 0, 0, value)
            else:
                raise ValueError("Unknown synthetic formula: " + formula)
            output[:, :, c] = value
    if channels == 4 and formula != "palette":
        output[:, :, 3] = np.where((x // 23 + y // 29) % 7 == 0, 0, output[:, :, 3])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-testdata", type=Path, required=True)
    parser.add_argument("--original-corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    spec_path = source / "wp_holdout_spec.json"
    spec = json.loads(spec_path.read_text())
    original = json.loads((args.original_corpus / "manifest.json").read_text())
    old_hashes = {im["sha256"] for im in original["inputs"]}
    old_sources = {im.get("source_path") for im in original["inputs"]}
    old_source_hashes = {im.get("source_sha256") for im in original["inputs"]}
    old_sample_hashes = {im.get("samples_sha256") for im in original["inputs"]}
    images, inputs = {}, []
    shutil.copy2(spec_path, root / "spec.json")
    shutil.copy2(Path(__file__), root / "prepare_wp_holdout.py")
    shutil.copy2(source / "compare_reference_workflow.py", root / "compare_reference_workflow.py")

    def save(name, pixels, category, description):
        require(pixels.dtype == np.uint16 and pixels.ndim == 3 and pixels.shape[2] in (1, 3, 4),
                "Invalid source profile")
        height, width, channels = pixels.shape
        raw = pixels.astype(">u2").tobytes()
        header = (f"P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH {channels}\nMAXVAL 65535\n"
                  f"TUPLTYPE { {1: 'GRAYSCALE', 3: 'RGB', 4: 'RGB_ALPHA'}[channels]}\nENDHDR\n").encode()
        path = root / "inputs" / (name + ".pam")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(header + raw)
        require(digest(path) not in old_hashes, "Input was in original rate corpus")
        require(np.count_nonzero(pixels % 257) > 0, "Merely expanded 8-bit samples")
        # Verify serialization numerically before any encoding uses it.
        readback = np.frombuffer(path.read_bytes()[len(header):], dtype=">u2").reshape(pixels.shape)
        require(np.array_equal(readback, pixels), "PAM conversion changed integers")
        require(hashlib.sha256(raw).hexdigest() not in old_sample_hashes, "Samples were in original rate corpus")
        inputs.append({"name": name, "path": str(path), "sha256": digest(path),
            "samples_sha256": hashlib.sha256(raw).hexdigest(), "width": width, "height": height,
            "channels": channels, "bits": 16, "category": category,
            "groups": ((width + 255) // 256) * ((height + 255) // 256),
            "non_8bit_expansion_samples": int(np.count_nonzero(pixels % 257)),
            "transparent_pixels": int(np.count_nonzero(pixels[:, :, 3] == 0)) if channels == 4 else 0,
            "preparation": description})

    for entry in spec["natural_sources"]:
        path = (args.reference_testdata / entry["path"]).resolve()
        require(str(path) not in old_sources, "Natural source was in original rate corpus")
        require(digest(path) not in old_source_hashes, "Natural source duplicates original corpus")
        header = path.read_bytes()[:33]
        require(header[:8] == b"\x89PNG\r\n\x1a\n", "Expected PNG")
        width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
        require(bits == 16 and color == 2 and not interlace, "Expected noninterlaced RGB16 source")
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        require(decoded is not None and decoded.dtype == np.uint16 and decoded.shape == (height, width, 3),
                "PNG decoder changed depth/shape")
        pixels = np.ascontiguousarray(decoded[:, :, ::-1])
        independent = subprocess.check_output(["magick", str(path), "-depth", "16", "-endian", "MSB", "rgb:-"])
        require(independent == pixels.astype(">u2").tobytes(), "Independent PNG sample reads differ")
        target = root / "sources" / (entry["name"] + ".png")
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        images[entry["name"]] = pixels
        save(entry["name"], pixels, "natural-rgb16", {"source": str(path), "source_sha256": digest(path),
             "operation": "OpenCV unchanged RGB sample unpack, independently matched by ImageMagick Q16"})
    for entry in spec["derived_natural_cases"]:
        original_pixels = images[entry["source"]]
        height, width, _ = original_pixels.shape
        if entry["operation"] == "center-256-rgb":
            pixels = original_pixels[(height - 256) // 2:(height - 256) // 2 + 256,
                                     (width - 256) // 2:(width - 256) // 2 + 256]
        elif entry["operation"] == "red-plane-gray":
            pixels = original_pixels[:, :, :1]
        elif entry["operation"] == "append-generated-alpha":
            y, x = np.indices((height, width), dtype=np.int64)
            alpha = (x * 239 + y * 607) & 65535
            alpha[(x // 23 + y // 29) % 7 == 0] = 0
            pixels = np.concatenate((original_pixels, alpha[:, :, None].astype(np.uint16)), axis=2)
        else:
            raise ValueError("Unknown natural derivation")
        save(entry["name"], pixels, "derived-natural16", entry)
    for family in spec["synthetic_families"]:
        for extent in spec["synthetic_extents"]:
            pixels = synthetic(family["formula"], extent["width"], extent["height"], family["channels"], spec["noise_seed"])
            save(family["name"] + "-" + extent["name"], pixels, "synthetic16",
                 {"family": family, "extent": extent, "seed": spec["noise_seed"]})
    require(len(inputs) == spec["expected_inputs"] and len({im["name"] for im in inputs}) == len(inputs),
            "Unexpected corpus membership")
    manifest = {"schema": 1, "kind": "wp-heldout-corpus", "spec": spec, "inputs": inputs,
        "encodes_run": False, "original_corpus_manifest": str((args.original_corpus / "manifest.json").resolve()),
        "original_corpus_manifest_sha256": digest(args.original_corpus / "manifest.json"),
        "tools": {"numpy": np.__version__, "opencv": cv2.__version__,
                  "imagemagick": subprocess.check_output(["magick", "-version"], text=True)},
        "artifacts": {str(p): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(len(inputs), "inputs; manifest", digest(root / "manifest.json"))


if __name__ == "__main__":
    main()
