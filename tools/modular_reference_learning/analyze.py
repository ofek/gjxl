#!/usr/bin/env python3
"""Verify the saved bridge controls and summarize rate and diagnostic timings."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_study(root):
    manifest_path, ledger = root / "manifest.json", root / "results.jsonl"
    manifest = json.loads(manifest_path.read_text())
    for path, digest in manifest["artifacts"].items():
        if sha(path) != digest:
            raise ValueError(f"Manifest artifact changed: {path}")
    for item in manifest["inputs"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Input changed")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    seen = set()
    for row in rows:
        key = row["input"], row["variant"], row["iteration"]
        if key in seen:
            raise ValueError(f"Duplicate case: {key}")
        seen.add(key)
        for path, digest in row["artifacts"].items():
            if sha(path) != digest:
                raise ValueError(f"Result artifact changed: {path}")
        result = row["result"]
        if not result["decoded_exact"] or (row["variant"] != "reference" and not result["tokens_exact"]):
            raise ValueError("Missing correctness check")
        fixtures = [path for path in row["artifacts"] if Path(path).suffix == ".bin"]
        fits = result["learning"]["fits"]
        if row["iteration"] == -1:
            if len(fixtures) != len(fits) or not all(fit["replayed"] for fit in fits):
                raise ValueError("Missing exact replay fixture")
        elif fixtures or result["learning"]["capture"] or result["learning"]["learning_includes_capture_overhead"]:
            raise ValueError("Timing run includes capture work")
    iterations = range(-1, manifest["timing_repeats"] + 1 if manifest["timing_repeats"] else 0)
    expected = {(item["name"], variant, iteration) for item in manifest["inputs"]
                for variant in manifest["variants"] for iteration in iterations}
    if seen != expected:
        raise ValueError(f"Incomplete ledger: missing {expected - seen}, unexpected {seen - expected}")
    return manifest, rows, {str(manifest_path): sha(manifest_path), str(ledger): sha(ledger)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--sanitizer-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, corpus, artifacts = load_study(args.corpus.resolve())
    timing_manifest, timing_rows, timing_artifacts = load_study(args.timings.resolve())
    artifacts.update(timing_artifacts)
    if set(manifest["variants"]) != {"reference", "best", "all"}:
        raise ValueError("Expected all three corpus controls")
    rates = {(row["input"], row["variant"]): row for row in corpus if row["iteration"] == -1}
    cases = []
    total = {variant: 0 for variant in ("reference", "native", "best", "all", "hindsight_best_or_all")}
    for item in manifest["inputs"]:
        name = item["name"]
        size = {variant: rates[name, variant]["result"]["bytes"] for variant in ("reference", "best", "all")}
        control = manifest["native_controls"][name]
        if sha(control["path"]) != control["sha256"]:
            raise ValueError("Changed native baseline")
        size["native"] = control["bytes"]
        size["hindsight_best_or_all"] = min(size["best"], size["all"])
        for variant, value in size.items():
            total[variant] += value
        saved = manifest["reference_controls"].get(name)
        if saved and rates[name, "reference"]["output_sha256"] != saved["output_sha256"]:
            raise ValueError("Saved stock e9 mismatch")
        cases.append({
            "input": name, "bytes": size,
            "change_from_native_percent": {v: 100 * (size[v] / size["native"] - 1) for v in ("best", "all")},
            "saved_reference_sha256_matched": bool(saved),
            "learning": {v: rates[name, v]["result"]["learning"] for v in ("reference", "best", "all")},
            "capture_process_peak_rss_bytes": {v: rates[name, v]["result"]["process_peak_rss_bytes"] for v in ("reference", "best", "all")},
        })
    timings = []
    for item in timing_manifest["inputs"]:
        name = item["name"]
        for variant in timing_manifest["variants"]:
            measured = [r for r in timing_rows if r["input"] == name and r["variant"] == variant and r["iteration"] > 0]
            if len(measured) != timing_manifest["timing_repeats"] or len(measured) < 2:
                raise ValueError("Insufficient measured timing repetitions")
            for row in timing_rows:
                if row["input"] == name and row["variant"] == variant and row["output_sha256"] != rates[name, variant]["output_sha256"]:
                    raise ValueError("Repeated or cross-study file mismatch")
            record = {"input": name, "variant": variant, "repeats": len(measured)}
            for field in ("cpu", "wall"):
                fit = [sum(f[field] for f in r["result"]["learning"]["fits"]) for r in measured]
                learning = [sum(t[field] for t in r["result"]["learning"]["learning"]) for r in measured]
                record[f"fit_{field}_seconds"] = {"median": statistics.median(fit), "runs": fit}
                record[f"learning_{field}_seconds"] = {"median": statistics.median(learning), "runs": learning}
            record["process_peak_rss_bytes"] = [r["result"]["process_peak_rss_bytes"] for r in measured]
            timings.append(record)
    result = {
        "schema": 1,
        "reference_revision": "e8ff09762481785938d8e4e01333ed3917571161",
        "harness_revision": manifest["source_revision"],
        "scope": "Frozen native winning transforms; imported pinned-reference learner; unchanged native estimated ANS. No legacy-tree fallback.",
        "totals": total,
        "gap_vs_reference_percent": {v: 100 * (value / total["reference"] - 1) for v, value in total.items()},
        "change_vs_native_percent": {v: 100 * (value / total["native"] - 1) for v, value in total.items()},
        "hindsight_is_not_a_selection_policy": True,
        "corpus_cases": len(cases), "corpus_encodes": len(corpus),
        "captured_fit_events": sum(len(r["result"]["learning"]["fits"]) for r in corpus),
        "timing_encodes_including_captures_and_warmups": len(timing_rows),
        "timing_scope": "Median of serial interleaved fresh-process runs, with one warmup. Diagnostic CPU/wall times on an uncontrolled desktop; learners use different samples/trees. Capture work excluded.",
        "memory_scope": "Process peak RSS includes image copies, full encoding and validation. Capture runs also include copied samples and replay. Not isolated learner scratch.",
        "cases": cases, "timings": timings, "ledgers": artifacts,
        "analyzer_sha256": sha(Path(__file__).resolve()),
    }
    validation = args.sanitizer_validation.resolve()
    capture_log = (validation / "capture-tests.log").read_text()
    if "Fixture persistence/replay passed; 10 malformed fixtures and one false tree rejected." not in capture_log:
        raise ValueError("Missing sanitizer rejection-test completion")
    import_checks = json.loads(capture_log.splitlines()[0])
    if import_checks != {"import_checks": 10, "passed": True}:
        raise ValueError("Missing sanitizer import checks")
    if not json.loads((validation / "photo-replay.log").read_text())["tree_exact"]:
        raise ValueError("Missing sanitized photographic replay")
    alpha = json.loads((validation / "alpha16.log").read_text())
    if not alpha["decoded_exact"] or not alpha["tokens_exact"] or not all(f["replayed"] for f in alpha["learning"]["fits"]):
        raise ValueError("Missing sanitized alpha16 bridge checks")
    if sha(validation / "alpha16.jxl") != rates["alpha16-ramp", "all"]["output_sha256"]:
        raise ValueError("Sanitized and Release files differ")
    cache = (validation / "CMakeCache.txt").read_text()
    if "-fsanitize=address,undefined -fno-sanitize=vptr" not in cache:
        raise ValueError("Missing sanitizer configuration")
    result["sanitizer_validation"] = {
        "scope": "Harness and generated reference fitting/preparation translation units instrumented; linked native/reference archives are uninstrumented. Vptr excluded for pinned no-RTTI archive interoperability. UBSAN_OPTIONS=halt_on_error=1.",
        "import_checks": 10, "malformed_fixtures": 10, "false_expected_trees": 1,
        "photographic_replay_exact": True, "alpha16_tokens_decode_and_release_bytes_exact": True,
        "artifacts": {str(p): sha(p) for p in sorted(validation.iterdir()) if p.is_file()},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"totals": total, "gap_vs_reference_percent": result["gap_vs_reference_percent"],
                      "captured_fit_events": result["captured_fit_events"]}, indent=2))


if __name__ == "__main__":
    main()
