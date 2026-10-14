#!/usr/bin/env python3
"""Generate research-only hooks without changing the pinned learning algorithm."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PIN = "e8ff09762481785938d8e4e01333ed3917571161"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one hook: {old!r}")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise ValueError(f"Reference must be {PIN}, got {revision}")
    hashes = {}

    def read(path):
        # Reject local edits to every reference source used by a generated hook.
        if path.is_relative_to(args.reference):
            relative = str(path.relative_to(args.reference))
            pristine = subprocess.check_output(
                ["git", "-C", str(args.reference), "show", f"{PIN}:{relative}"]
            )
            if pristine != path.read_bytes():
                raise ValueError(f"Modified reference source: {path}")
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path.read_text()

    def write(relative, content):
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    root = Path("lib/jxl/modular/encoding")
    header = read(args.reference / root / "enc_ma.h")
    # A friend changes neither layout nor the production API. Only this research
    # include directory sees it; the checked-out reference remains untouched.
    header = replace_once(header, " private:\n", " private:\n  friend struct ReferenceLearningSampleAccess;\n")
    write(root / "enc_ma.h", header)
    fitting = read(args.reference / root / "enc_ma.cc")
    fitting = replace_once(fitting, "Status ComputeBestTree(", "Status ComputeBestTreeUncaptured(")
    write(root / "enc_ma.cc", fitting)
    encoding = read(args.reference / root / "enc_encoding.cc")
    encoding = '#include "reference_capture.h"\n' + encoding
    encoding = '#include "policy_trace.h"\n' + encoding
    encoding = replace_once(encoding, "  TreeSamples tree_samples;\n",
                            '  reference_learning::LearningInputTrace trace(images, options, start, stop, multiplier_info);\n'
                            '  reference_learning::LearningTimer timer;\n  TreeSamples tree_samples;\n')
    encoding = replace_once(encoding, "                                 options[start], multiplier_info, range));\n  return tree;",
                            "                                 options[start], multiplier_info, range));\n  trace.Finish(tree);\n  return tree;")
    encoding = replace_once(encoding, "  ModularOptions options = opts;  // Make a copy to modify it.",
                            "  reference_learning::ScopedLocalLearning local_scope(group_id);\n"
                            "  ModularOptions options = opts;  // Make a copy to modify it.")
    encoding = replace_once(encoding, "  header.use_global_tree = true;\n",
                            "  header.use_global_tree = true;\n"
                            "  reference_learning::TraceCodingStream(image, options, group_id);\n")
    write(root / "enc_encoding.cc", encoding)
    adapter = read(source / "tests/modular_reference.cpp")
    adapter = replace_once(adapter, "                                               1) == JXL_ENC_SUCCESS,\n",
                            "                                               9) == JXL_ENC_SUCCESS,\n")
    adapter = '#include "policy_trace.h"\n' + adapter
    adapter = replace_once(adapter, '          "Reference lossless settings failed");\n',
                            '          "Reference lossless settings failed");\n'
                            '  reference_learning::ConfigureReferencePolicy(settings);\n')
    write("reference_codec.cpp", adapter)
    write("provenance.json", json.dumps({"reference_revision": PIN, "inputs": hashes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
