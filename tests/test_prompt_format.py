"""Tests pinning the host-eval user-message format.

The single load-bearing detail is the BACKTICKS around `\\boxed{}` and
`\\boxed{your answer}` in BOXED_INSTRUCTION. Training prompts must match
the host's inference prompt exactly — see docs/notebook_learnings/04_*.md
for the parity story.
"""

from __future__ import annotations

from nemotron_local_cv.host_metric import BOXED_INSTRUCTION, build_user_content


class TestBoxedInstruction:
    def test_exact_string(self):
        # The exact suffix verified against the 2026-05-18 host kernel.
        # If this changes, the change MUST come from re-pulling the host metric.
        expected = (
            "\nPlease put your final answer inside `\\boxed{}`. "
            "For example: `\\boxed{your answer}`"
        )
        assert BOXED_INSTRUCTION == expected

    def test_contains_backticks(self):
        # The 0.86 packaging quirk: prompts trained without backticks score
        # lower because the host suffix HAS backticks and the model learns to
        # ignore the surrounding tokens.
        assert "`\\boxed{}`" in BOXED_INSTRUCTION
        assert "`\\boxed{your answer}`" in BOXED_INSTRUCTION

    def test_starts_with_newline(self):
        # The host concatenates suffix directly onto the prompt with no
        # extra separator; the leading newline is the only separator.
        assert BOXED_INSTRUCTION.startswith("\n")

    def test_no_trailing_whitespace(self):
        assert BOXED_INSTRUCTION == BOXED_INSTRUCTION.rstrip()


class TestBuildUserContent:
    def test_appends_suffix_verbatim(self):
        assert build_user_content("PROMPT") == "PROMPT" + BOXED_INSTRUCTION

    def test_empty_prompt_still_gets_suffix(self):
        assert build_user_content("") == BOXED_INSTRUCTION

    def test_prompt_with_trailing_newline_preserved(self):
        # No collapsing or whitespace fixups — verbatim concat.
        assert build_user_content("foo\n") == "foo\n" + BOXED_INSTRUCTION

    def test_multibyte_unicode_preserved(self):
        prompt = "解一道题 🎲"
        assert build_user_content(prompt) == prompt + BOXED_INSTRUCTION
