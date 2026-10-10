#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create the deterministic unpack/RCT pilot matrix (requires numpy/Pillow).

Synthetic inputs are scalability probes, not a photographic corpus. Optional
500px wesaturate inputs preserve their decoded RGB samples without resizing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wesaturate", type=Path)
    parser.add_argument("--token-slice", action="store_true",
                        help="Add prescribed split trees and full-resolution photograph cases")
    parser.add_argument("--large-photos", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    if args.token_slice and len(args.large_photos) < 3:
        parser.error("Token qualification requires at least three larger natural RGB photographs")
    root = args.output.resolve()
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    records, cases = {}, []

    def save(name, data, bits, provenance):
        h, w, channels = data.shape
        path = inputs / (name + ".pam")
        kind = {1: "GRAYSCALE", 3: "RGB", 4: "RGB_ALPHA"}[channels]
        header = (f"P7\nWIDTH {w}\nHEIGHT {h}\nDEPTH {channels}\n"
                  f"MAXVAL {(1 << bits) - 1}\nTUPLTYPE {kind}\nENDHDR\n")
        path.write_bytes(header.encode() + data.astype("u1" if bits == 8 else ">u2").tobytes())
        records[name] = dict(path=str(path), sha256=digest(path), width=w, height=h,
                             channels=channels, bits=bits, **provenance)

    def synthetic(name, w, h, channels, bits, noise=False):
        if name in records:
            return
        y, x = np.indices((h, w), dtype=np.uint32)
        # Structured ramps, discontinuities and checkerboards exercise prediction
        # and signed RCT differences. Alpha includes zero with nonzero RGB.
        mask = (1 << bits) - 1
        base = (x * 17 + y * 29 + ((x // 53) ^ (y // 37)) * 511) & mask
        data = np.empty((h, w, channels), dtype=np.uint16)
        for c in range(channels):
            data[:, :, c] = (base + c * 107 + (x * (c + 1) ^ y * 3)) & mask
        if channels == 4:
            data[:, :, 3] = np.where((x // 19 + y // 23) % 7 == 0, 0, (x * 13 + y * 11) & mask)
        if noise:
            data = np.random.Generator(np.random.PCG64(20261010)).integers(
                0, mask + 1, data.shape, dtype=np.uint16)
        save(name, data, bits, dict(kind="synthetic-noise" if noise else "synthetic-structured",
                                   generator="prepare_gpu_performance.py", seed=20261010))

    def case(name, source, rct=6, coder="prefix", threads=1, jobs=1, order="little", padding=0,
             tree="gradient"):
        row = [name, records[source]["path"], rct, coder, threads, jobs, order, padding]
        if args.token_slice:
            row.append(tree)
        cases.append(row)

    for name, w, h in [("tiny", 1, 1), ("boundary", 257, 255), ("mp1", 1024, 1024),
                        ("hd", 1920, 1080), ("uhd", 3840, 2160)]:
        synthetic(name + "-rgb8", w, h, 3, 8)
        case(name + "-rgb8", name + "-rgb8", padding=5 if name == "boundary" else 0)
    for channels, label in [(1, "gray"), (3, "rgb"), (4, "rgba")]:
        for bits in (8, 16):
            name = f"mp1-{label}{bits}"
            synthetic(name, 1024, 1024, channels, bits)
            if name != "mp1-rgb8":
                case(name, name, rct=0 if channels == 1 else 6,
                     order="big" if bits == 16 else "little", padding=7)
    for label, channels, bits in [("gray16", 1, 16), ("rgba16", 4, 16)]:
        name = "uhd-" + label
        synthetic(name, 3840, 2160, channels, bits)
        case(name, name, rct=0 if channels == 1 else 6, padding=3)
    for size in ("mp1", "uhd"):
        source = size + "-rgb8"
        case(source + "-identity", source, rct=0)
        case(source + "-rct41", source, rct=41)
        case(source + "-ans", source, coder="ans")
        case(source + "-t4", source, threads=4)
    case("uhd-rgba16-t4", "uhd-rgba16", threads=4, padding=3)
    case("mp1-rgba16-le", "mp1-rgba16", padding=7)
    case("mp1-rgb8-b4", "mp1-rgb8", jobs=4)
    case("hd-rgb8-b4", "hd-rgb8", jobs=4)
    synthetic("mp1-noise-rgb16", 1024, 1024, 3, 16, noise=True)
    case("mp1-noise-rgb16", "mp1-noise-rgb16")
    case("mp1-noise-rgb16-ans", "mp1-noise-rgb16", coder="ans")
    if args.wesaturate:
        for stem in ("tmshre_riaphotographs_srgb8", "cvo9xd_keong_macan_srgb8", "u76c0g_bliznaca_srgb8"):
            source = args.wesaturate / (stem + ".png")
            with Image.open(source) as im:
                if im.mode != "RGB":
                    raise ValueError(f"Expected existing RGB samples: {source}: {im.mode}")
                data = np.array(im)
            name = "photo-" + stem.split("_")[0]
            save(name, data, 8, dict(kind="photograph", source=str(source.resolve()),
                                    source_sha256=digest(source), conversion="Pillow RGB samples; no resize/color conversion"))
            case(name, name)
            case(name + "-ans", name, coder="ans")
    if args.token_slice:
        for source in ("mp1-rgb8", "uhd-rgb8", "uhd-rgba16"):
            for threads in (1, 4):
                case(source + f"-split7-t{threads}", source, threads=threads, tree="split7")
        for path in args.large_photos:
            with Image.open(path) as im:
                if im.mode != "RGB" or im.width * im.height < 4_000_000:
                    raise ValueError(f"Expected natural RGB photograph >=4MP: {path}: {im.mode} {im.size}")
                data = np.array(im)
            name = "photo-large-" + path.stem.split(".")[0].lower().replace("_", "-")
            save(name, data, 8, dict(kind="photograph", source=str(path.resolve()),
                source_sha256=digest(path), conversion="Pillow RGB samples; no resize/color conversion"))
            for tree in ("gradient", "split7"):
                for threads in (1,4):
                    case(name + f"-{tree}-t{threads}", name, threads=threads, tree=tree)
            case(name + "-gradient-ans-t4", name, threads=4, coder="ans")
            case(name + "-gradient-b4", name, jobs=4)
    manifest = root / "cases.tsv"
    manifest.write_text("# name\tpath\trct\tcoder\tthreads\tjobs\torder\tpadding" +
                        ("\ttree\n" if args.token_slice else "\n") +
                        "".join("\t".join(map(str, c)) + "\n" for c in cases))
    (root / "inputs.json").write_text(json.dumps(dict(inputs=records, cases=len(cases),
                                                    manifest_sha256=digest(manifest)), indent=2) + "\n")
    print(f"Created {len(cases)} cases from {len(records)} inputs in {root}")


if __name__ == "__main__":
    main()
