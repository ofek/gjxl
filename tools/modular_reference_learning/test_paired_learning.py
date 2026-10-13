#!/usr/bin/env python3
"""Reject misleading timing claims and invalid matched-learning controls."""
import copy
import math
import unittest

from analyze_paired_learning import summarize_case


class PairedLearningEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.case = {"input": "image", "variant": "best", "expected_shape": {
            "samples": 128, "distinct": 100, "nodes": 3, "predictors": 2, "properties": 16}}
        self.row = {"input": "image", "variant": "best", "runs": []}
        for i in range(3):
            run = {"iteration": i, "reference_first": i % 2 == 0,
                   "samples_exact_in_warmup": True, "trees_exact": True,
                   **self.case["expected_shape"], "preparation_storage_bound_bytes": 10000,
                   "fitting_storage_bound_bytes": 1000}
            for metric in ("cpu", "wall"):
                run.update({k + "_" + metric: v for k, v in {
                    "native_total": 2, "native_preparation": 0.4, "native_fit": 1.5,
                    "reference_total": 3, "reference_fit": 2, "reference_non_fit": 1}.items()})
            self.row["runs"].append(run)

    def test_valid(self):
        case = summarize_case(self.case, self.row, 3)
        self.assertEqual(case["total_cpu_ratio"], 2 / 3)
        self.assertEqual(case["fit_cpu_ratio"], 0.75)

    def test_non_fit_is_difference_before_median(self):
        for run, total, fit in zip(self.row["runs"], (6, 7, 10), (1, 6, 3)):
            run["native_total_cpu"], run["native_fit_cpu"] = total, fit
            run["reference_total_cpu"], run["reference_fit_cpu"] = total, fit
            run["reference_non_fit_cpu"] = total - fit
        case = summarize_case(self.case, self.row, 3)
        self.assertEqual(case["native_non_fit_cpu"]["median"], 5)
        self.assertEqual(case["reference_non_fit_cpu"]["median"], 5)
        self.assertNotEqual(case["native_total_cpu"]["median"] - case["native_fit_cpu"]["median"], 5)

    def test_timing_failures(self):
        changes = (("native_total_cpu", 1, "nested"),
                   ("reference_non_fit_cpu", 0.8, "difference"),
                   ("reference_fit_cpu", 4, "difference"),
                   ("native_fit_cpu", math.nan, "nonfinite"),
                   ("native_fit_wall", math.inf, "nonfinite"),
                   ("native_total_cpu", True, "nonfinite"),
                   ("reference_total_cpu", 0, "Nonpositive"))
        for key, value, message in changes:
            with self.subTest(key=key, value=value):
                row = copy.deepcopy(self.row)
                row["runs"][0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    summarize_case(self.case, row, 3)

    def test_control_failures(self):
        changes = (("trees_exact", False, "exact checks"),
                   ("samples_exact_in_warmup", 1, "exact checks"),
                   ("reference_first", False, "order"),
                   ("iteration", 2, "order"),
                   ("samples", 129, "Shape"),
                   ("nodes", 5, "Shape"),
                   ("preparation_storage_bound_bytes", 9999, "storage"),
                   ("fitting_storage_bound_bytes", 0, "storage"))
        for key, value, message in changes:
            with self.subTest(key=key, value=value):
                row = copy.deepcopy(self.row)
                row["runs"][1][key] = value
                if key == "reference_first":
                    row["runs"][0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    summarize_case(self.case, row, 3)

    def test_incomplete(self):
        self.row["runs"].pop()
        with self.assertRaisesRegex(ValueError, "repeats"):
            summarize_case(self.case, self.row, 3)

    def test_wrong_identity(self):
        self.row["variant"] = "all"
        with self.assertRaisesRegex(ValueError, "identity"):
            summarize_case(self.case, self.row, 3)


if __name__ == "__main__":
    unittest.main()
