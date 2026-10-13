#!/usr/bin/env python3
"""Exercise replay qualification failures without running encodes or benchmarks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from analyze import sha
from analyze_native_replay import analyze


class QualificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.binary = self.root / "reference_learning"
        self.binary.write_bytes(b"synthetic analyzer fixture, not executable")
        fixture = self.root / "fixture.bin"
        fixture.write_bytes(b"synthetic analyzer input")
        self.fixture = {"input": "test", "variant": "best", "fit": 0,
                        "path": str(fixture), "sha256": sha(fixture)}
        self.stdout = self.root / "stdout.jsonl"
        self.runs = [{"iteration": i, "reference_first": i % 2 == 0,
                      "tree_exact": True, "nodes": 3, "samples": 128, "distinct": 64,
                      "predictors": 2, "properties": 16, "cpu": cpu, "wall": cpu + 0.1,
                      "reference_cpu": 1.0, "reference_wall": 1.1,
                      "sample_backing_bytes": 100, "scratch_bound_bytes": 1000,
                      "scratch_peak_bytes": 900} for i, cpu in enumerate([1.2, 1.7, 1.6])]
        self.row = {**self.fixture, "runs": self.runs,
                    "command": [str(self.binary), "paired-replay", str(fixture), "3"]}
        self.manifest = {"mode": "paired", "repeats": 3,
                         "fixtures": [self.fixture], "artifacts": {str(self.binary): sha(self.binary)},
                         "timing_scope": "synthetic analyzer test"}

    def save(self):
        self.stdout.write_text("".join(json.dumps(r) + "\n" for r in self.runs))
        self.row["artifacts"] = {str(self.stdout): sha(self.stdout)}
        (self.root / "results.jsonl").write_text(json.dumps(self.row) + "\n")
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_valid_and_gate_outlier(self):
        self.save()
        result = analyze(self.root)
        self.assertEqual(result["cpu_ratio_of_summed_case_medians"], 1.6)
        self.assertEqual(len(result["cpu_gate_1_5_exceeded"]), 1)
        self.assertEqual(result["cases"][0]["cpu"]["runs"], [1.2, 1.7, 1.6])

    def test_missing_fixture(self):
        self.save()
        (self.root / "results.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analyze(self.root)

    def test_changed_binary(self):
        self.save()
        self.binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Manifest artifact changed"):
            analyze(self.root)

    def test_changed_input(self):
        self.save()
        Path(self.fixture["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed paired input"):
            analyze(self.root)

    def test_changed_raw_output(self):
        self.save()
        self.stdout.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Replay result artifact changed"):
            analyze(self.root)

    def test_ledger_disagrees_with_raw_output(self):
        self.row["runs"] = copy.deepcopy(self.runs)
        self.row["runs"][0]["cpu"] = 0.1
        self.save()
        with self.assertRaisesRegex(ValueError, "differs from raw stdout"):
            analyze(self.root)

    def test_semantic_failures(self):
        changes = [("tree_exact", False, "inexact"),
                   ("reference_first", False, "order"),
                   ("nodes", 5, "deterministic"),
                   ("scratch_peak_bytes", 1001, "admission"),
                   ("cpu", 0, "Nonpositive")]
        for field, value, reason in changes:
            with self.subTest(field=field):
                original = self.runs[0][field]
                if field == "scratch_peak_bytes":
                    for run in self.runs:
                        run[field] = value
                else:
                    self.runs[0][field] = value
                self.save()
                with self.assertRaisesRegex(ValueError, reason):
                    analyze(self.root)
                for run in self.runs if field == "scratch_peak_bytes" else self.runs[:1]:
                    run[field] = original

    def test_wrong_command(self):
        self.row["command"][1] = "native-replay"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unexpected replay command"):
            analyze(self.root)

    def test_too_few_repeats(self):
        self.manifest["repeats"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "three paired repeats"):
            analyze(self.root)


if __name__ == "__main__":
    unittest.main()
