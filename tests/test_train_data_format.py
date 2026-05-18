"""Tests for the SFT data formatter (`nemotron_train.data`).

The critical invariant: the user message we train on must equal the user
message the host scorer constructs at eval time — same prompt, same suffix,
same backticks. The single source of truth for the suffix is
`nemotron_local_cv.host_metric.BOXED_INSTRUCTION`; this test pins both
modules to that source.

We use a fake tokenizer + a stub `datasets` module so we don't need GPU
deps. The fake tokenizer records every `apply_chat_template` call so we
can assert byte-for-byte what messages get rendered for training.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from nemotron_local_cv.host_metric import BOXED_INSTRUCTION, build_user_content
from nemotron_train import data as train_data


@dataclass
class FakeTokenizer:
    calls: list = field(default_factory=list)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        # Deterministic, easy-to-grep render.
        return "<TPL>" + " || ".join(f"[{m['role']}] {m['content']}" for m in messages) + "</TPL>"


# --- _build_assistant_content -------------------------------------------------


class TestAssistantContent:
    def test_strips_stray_boxed_and_appends_verified(self):
        cot = r"Step 1. think think. The answer is \boxed{wrong}"
        out = train_data._build_assistant_content(cot, "right")
        assert r"\boxed{wrong}" not in out
        assert out.endswith(r"\boxed{right}")

    def test_keeps_body_when_no_stray_boxed(self):
        cot = "Plain reasoning, no box."
        out = train_data._build_assistant_content(cot, "42")
        assert out == "Plain reasoning, no box.\n\\boxed{42}"

    def test_empty_cot(self):
        out = train_data._build_assistant_content("", "42")
        assert out == "\n\\boxed{42}"

    def test_none_cot_treated_as_empty(self):
        out = train_data._build_assistant_content(None, "42")
        assert out == "\n\\boxed{42}"

    def test_multiple_stray_boxed_all_stripped(self):
        cot = r"\boxed{a} middle \boxed{b}"
        out = train_data._build_assistant_content(cot, "c")
        assert out.count(r"\boxed{") == 1
        assert out.endswith(r"\boxed{c}")


# --- _format_row --------------------------------------------------------------


class TestFormatRow:
    def test_user_content_matches_host_build_user_content(self):
        tok = FakeTokenizer()
        prompt = "PUZZLE"
        train_data._format_row(tok, prompt, "ANS", "COT")
        msgs = tok.calls[0]["messages"]
        # The user message MUST equal what the host scorer constructs.
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == build_user_content(prompt)
        assert msgs[0]["content"].endswith(BOXED_INSTRUCTION)

    def test_assistant_role_and_terminal_boxed(self):
        tok = FakeTokenizer()
        train_data._format_row(tok, "p", "ANS", "COT")
        assistant = tok.calls[0]["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["content"].endswith(r"\boxed{ANS}")

    def test_chat_template_kwargs(self):
        tok = FakeTokenizer()
        train_data._format_row(tok, "p", "a", "c")
        call = tok.calls[0]
        # Training-time render: never include the generation prompt; always
        # render bytes (not token ids); always include the thinking scaffold.
        assert call["tokenize"] is False
        assert call["add_generation_prompt"] is False
        assert call["enable_thinking"] is True


# --- load_sft_dataset ---------------------------------------------------------


class _FakeDataset:
    """Minimal stand-in for `datasets.Dataset` with the bits we read in tests."""

    def __init__(self, mapping):
        self._mapping = {k: list(v) for k, v in mapping.items()}

    @classmethod
    def from_dict(cls, mapping):
        return cls(mapping)

    def __len__(self):
        return len(next(iter(self._mapping.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return list(self._mapping[key])
        return {k: v[key] for k, v in self._mapping.items()}


@pytest.fixture
def fake_datasets_module(monkeypatch):
    """Install a stub `datasets` module so load_sft_dataset doesn't need the real dep."""
    mod = types.ModuleType("datasets")
    mod.Dataset = _FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return mod


@pytest.fixture
def corpus_csv(tmp_path: Path) -> Path:
    rows = [
        {"id": "a1", "prompt": "P1", "answer": "ANS1",
         "generated_cot": "thinking 1", "label": "bit_manipulation"},
        {"id": "a2", "prompt": "P2", "answer": "ANS2",
         "generated_cot": "thinking 2", "label": "UNKNOWN_NEEDS_RECLASSIFY"},
        {"id": "a3", "prompt": "P3", "answer": "ANS3",
         "generated_cot": r"thinking 3 with \boxed{stray}", "label": "numeral_conversion"},
    ]
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestLoadSftDataset:
    def test_drops_unknown_reclassify_rows_by_default(self, corpus_csv, fake_datasets_module):
        tok = FakeTokenizer()
        ds = train_data.load_sft_dataset(corpus_csv, tok)
        # 3 rows in, 1 dropped, 2 formatted.
        assert len(ds) == 2
        assert len(tok.calls) == 2

    def test_all_text_rows_have_correct_user_suffix(self, corpus_csv, fake_datasets_module):
        tok = FakeTokenizer()
        ds = train_data.load_sft_dataset(corpus_csv, tok)
        for text in ds["text"]:
            # Our fake tokenizer renders messages verbatim; the user message
            # we put on the tape must contain the host suffix.
            assert BOXED_INSTRUCTION in text

    def test_stray_boxed_stripped_in_assistant(self, corpus_csv, fake_datasets_module):
        tok = FakeTokenizer()
        train_data.load_sft_dataset(corpus_csv, tok)
        a3_call = next(c for c in tok.calls if c["messages"][0]["content"].startswith("P3"))
        assistant = a3_call["messages"][1]["content"]
        # exactly one \boxed{...} and it carries the verified answer.
        assert assistant.count(r"\boxed{") == 1
        assert assistant.endswith(r"\boxed{ANS3}")

    def test_missing_required_column_raises(self, tmp_path, fake_datasets_module):
        bad = tmp_path / "bad.csv"
        pd.DataFrame([{"id": "x", "prompt": "p", "answer": "a"}]).to_csv(bad, index=False)
        tok = FakeTokenizer()
        with pytest.raises(ValueError, match="generated_cot"):
            train_data.load_sft_dataset(bad, tok)

    def test_keep_all_rows_when_drop_labels_empty(self, corpus_csv, fake_datasets_module):
        tok = FakeTokenizer()
        ds = train_data.load_sft_dataset(corpus_csv, tok, drop_labels=())
        assert len(ds) == 3

    def test_sample_n_subsamples_deterministically(self, corpus_csv, fake_datasets_module):
        tok = FakeTokenizer()
        ds_a = train_data.load_sft_dataset(corpus_csv, tok, sample_n=1, seed=42)
        ds_b = train_data.load_sft_dataset(corpus_csv, FakeTokenizer(), sample_n=1, seed=42)
        assert len(ds_a) == 1
        assert ds_a["text"] == ds_b["text"]

    def test_max_chars_filter(self, corpus_csv, fake_datasets_module):
        tok = FakeTokenizer()
        # All CoTs in the fixture are < 100 chars; cap of 5 should drop everything
        # not UNKNOWN_NEEDS_RECLASSIFY.
        ds = train_data.load_sft_dataset(corpus_csv, tok, max_chars=5)
        assert len(ds) == 0
