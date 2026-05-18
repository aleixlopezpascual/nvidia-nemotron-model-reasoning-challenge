"""Tests for the host-metric `extract_final_answer`.

The function is byte-identical with the 2026-05-18 host kernel. These cases
pin the contract — if any of them break, our local CV no longer mirrors the
host scorer and downstream LB numbers will diverge from CV.
"""

from __future__ import annotations

import pytest

from nemotron_local_cv.host_metric import extract_final_answer


class TestHostDocstring:
    """Verbatim from the host metric's docstring examples."""

    def test_simple_boxed(self):
        assert extract_final_answer(r"The answer is \boxed{42}") == "42"

    def test_final_answer_phrase(self):
        assert extract_final_answer("The final answer is: 3.14") == "3.14"

    def test_bare_number_in_text(self):
        assert extract_final_answer("Just a number 100 in text") == "100"

    def test_none_returns_sentinel(self):
        assert extract_final_answer(None) == "NOT_FOUND"


class TestBoxedExtraction:
    """The boxed-form is the primary contract; everything else is fallback."""

    def test_last_boxed_wins_when_multiple(self):
        assert extract_final_answer(r"\boxed{first} then \boxed{second}") == "second"

    def test_nested_latex_inside_boxed(self):
        # The 2026-05-18 fix walks `\boxed{` starts and takes everything up to
        # the last `}` before the next `\boxed{`. Critical for LaTeX answers.
        assert (
            extract_final_answer(r"answer: \boxed{\frac{1}{2}}")
            == r"\frac{1}{2}"
        )

    def test_boxed_with_literal_closing_brace_in_body(self):
        assert extract_final_answer(r"\boxed{a}b}") == "a}b"

    def test_empty_boxed_returns_empty_string(self):
        # Host quirk: an empty `\boxed{}` short-circuits to "" — the fallback
        # phrase / numeric / last-line logic is NOT invoked. Pin this exactly,
        # since the host scorer's verify() treats "" as not equal to any real
        # answer (including the empty string after .strip()).
        assert extract_final_answer(r"\boxed{}") == ""

    def test_empty_boxed_does_not_trigger_fallback(self):
        assert extract_final_answer(r"value 99 then \boxed{}") == ""

    def test_whitespace_inside_boxed_stripped(self):
        assert extract_final_answer(r"\boxed{   42   }") == "42"

    def test_unterminated_boxed_takes_to_eot(self):
        # No closing brace → falls through to remaining-text segment, no `}`.
        result = extract_final_answer(r"prefix \boxed{42 trailing")
        assert result == "42 trailing"


class TestPhraseFallbacks:
    def test_final_answer_with_colon(self):
        assert extract_final_answer("Final answer: cat") == "cat"

    def test_final_answer_case_insensitive(self):
        assert extract_final_answer("FINAL ANSWER: dog") == "dog"

    def test_fullwidth_colon_supported(self):
        assert extract_final_answer("final answer：fish") == "fish"

    def test_last_phrase_match_wins(self):
        text = "Final answer: wrong\nFinal answer: right"
        assert extract_final_answer(text) == "right"


class TestNumericFallback:
    def test_picks_last_number(self):
        assert extract_final_answer("we tried 1, then 2, then 3") == "3"

    def test_negative_number(self):
        assert extract_final_answer("result is -7") == "-7"

    def test_decimal_number(self):
        assert extract_final_answer("π ≈ 3.14159") == "3.14159"


class TestEmptyAndDegenerate:
    def test_empty_string(self):
        assert extract_final_answer("") == "NOT_FOUND"

    def test_only_whitespace(self):
        assert extract_final_answer("   \n\t  ") == "NOT_FOUND"

    def test_last_nonempty_line_when_no_signals(self):
        assert extract_final_answer("foo\nbar\n   \n") == "bar"
