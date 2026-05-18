"""Tests for the host-metric `verify`.

Two branches matter:
  1. binary-string bypass: if `stored_answer` is `[01]+` we do exact string
     compare (no numeric parsing). Stops 11011 == 00011011 collisions.
  2. numeric: float-cast both sides, compare with rel_tol=1e-2, abs_tol=1e-5.
  3. fallback: case-insensitive string equality.
"""

from __future__ import annotations

import pytest

from nemotron_local_cv.host_metric import verify


class TestHostDocstring:
    def test_binary_exact(self):
        assert verify("10011000", "10011000") is True

    def test_binary_off_by_one_bit(self):
        assert verify("10011000", "10011001") is False

    def test_numeric_within_tolerance(self):
        assert verify("24.64", "24.6401") is True

    def test_case_insensitive_string(self):
        assert verify("XLVII", "xlvii") is True

    def test_binary_left_pad_not_equal(self):
        # The bit-manipulation tasks are very specific about width; '11011' is
        # not the same answer as '00011011'.
        assert verify("11011", "00011011") is False


class TestBinaryBypass:
    def test_zeros_and_ones_only_treated_as_binary(self):
        # "10" looks like a number but is also `[01]+` — bypass wins. The
        # bypass uses lowercase string-equality, so "10" == "10" passes.
        assert verify("10", "10") is True

    def test_binary_with_leading_zeros(self):
        assert verify("00010", "00010") is True

    def test_binary_vs_decimal_numeric_would_have_passed_but_doesnt(self):
        # Numeric parse would say "10" == "10.0"; the binary bypass blocks it.
        assert verify("10", "10.0") is False

    def test_non_binary_digit_falls_to_numeric(self):
        # "12" has non-binary digits, so it goes to numeric branch.
        assert verify("12", "12.0") is True


class TestNumericTolerance:
    def test_relative_one_percent(self):
        assert verify("250", "251.2") is True

    def test_outside_relative_tolerance(self):
        assert verify("250", "260") is False

    def test_negative_numbers(self):
        assert verify("-3.14", "-3.1401") is True

    def test_integer_vs_float_string(self):
        assert verify("42", "42.0") is True

    def test_scientific_notation(self):
        assert verify("1e3", "1000") is True

    def test_near_zero_uses_abs_tol(self):
        # Stored is non-binary "0.0" so we reach the numeric branch.
        assert verify("0.0", "0.000005") is True
        assert verify("0.0", "0.0001") is False


class TestBinaryBypassQuirks:
    """The binary bypass only inspects `stored_answer`'s digit set. Any
    all-[01] stored value is treated as binary, even when the puzzle is
    actually numeric. Pinning this so we notice if the host ever changes it."""

    def test_decimal_one_hundred_taken_as_binary(self):
        # Pure integer 100 is `[01]+`, so verify("100", "100.0") fails even
        # though numerically they're equal.
        assert verify("100", "100.0") is False

    def test_decimal_one_taken_as_binary(self):
        assert verify("1", "1.0") is False

    def test_decimal_with_two_in_it_is_numeric(self):
        assert verify("12", "12.0") is True


class TestStringFallback:
    def test_exact_string(self):
        assert verify("cat", "cat") is True

    def test_string_with_whitespace_stripped(self):
        assert verify("  cat  ", "cat") is True

    def test_string_inequality(self):
        assert verify("cat", "dog") is False

    def test_case_insensitive(self):
        assert verify("Cat", "CAT") is True

    def test_non_numeric_strings_with_digits(self):
        # "abc123" can't float-parse — falls to string branch.
        assert verify("abc123", "ABC123") is True


class TestEdgeCases:
    def test_unparseable_vs_string_equal(self):
        # If predicted has trailing junk that breaks float() the string branch
        # still saves us when they happen to string-equal.
        assert verify("foo", "FOO") is True

    def test_mixed_numeric_vs_string(self):
        # Stored is numeric, predicted is non-numeric. float() on predicted
        # raises → string fallback → unequal.
        assert verify("42", "forty-two") is False

    def test_empty_strings_equal(self):
        # Both empty after strip → string equality is True. This is a host
        # property worth pinning so we notice if it ever changes.
        assert verify("", "") is True
