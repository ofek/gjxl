#!/usr/bin/env python3
"""Reject invalid claims even when the preparation ledger is parseable."""
import copy
import unittest
from analyze_native_preparation import suite_passed, validate_case


class PreparationEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.case = {"input": "image", "variant": "all", "expected_bytes": 99, "expected_sha256": "digest"}
        self.row = {"input": "image", "variant": "all", "output_sha256": "digest",
                    "result": {"samples_exact": True, "tree_exact": True, "decoded_exact": True,
                               "bytes": 99, "distinct": 40, "samples": 50, "scanned_pixels": 80,
                               "population": 81, "storage_peak_bytes": 100, "storage_bound_bytes": 105}}

    def test_valid(self):
        validate_case(self.case, self.row)

    def test_ctest_summary_versions(self):
        for text in ("100% tests passed out of 6", "100% tests passed, 0 tests failed out of 6"):
            self.assertTrue(suite_passed(text + "\n", 6))
            self.assertFalse(suite_passed(text + "\n", 5))
        self.assertFalse(suite_passed("83% tests passed, 1 tests failed out of 6", 6))

    def test_missing_checks(self):
        for field in ("samples_exact", "tree_exact", "decoded_exact"):
            for value in (False, None, 1, "true"):
                with self.subTest(field=field, value=value):
                    row = copy.deepcopy(self.row); row["result"][field] = value
                    with self.assertRaises(ValueError): validate_case(self.case, row)

    def test_wrong_bytes(self):
        self.row["result"]["bytes"] += 1
        with self.assertRaises(ValueError): validate_case(self.case, self.row)

    def test_wrong_hash(self):
        self.row["output_sha256"] = "changed"
        with self.assertRaises(ValueError): validate_case(self.case, self.row)

    def test_wrong_policy(self):
        self.row["variant"] = "best"
        with self.assertRaises(ValueError): validate_case(self.case, self.row)

    def test_invalid_population(self):
        for field, value in (("distinct", 0), ("distinct", 51), ("samples", 81), ("population", 80)):
            with self.subTest(field=field):
                row = copy.deepcopy(self.row); row["result"][field] = value
                with self.assertRaises(ValueError): validate_case(self.case, row)

    def test_invalid_storage(self):
        for value in (0, 106):
            with self.subTest(value=value):
                self.row["result"]["storage_peak_bytes"] = value
                with self.assertRaises(ValueError): validate_case(self.case, self.row)


if __name__ == "__main__":
    unittest.main()
