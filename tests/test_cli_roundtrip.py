"""End-to-end test of `python -m nemotron_local_cv` with vLLM mocked out.

We monkeypatch `host_metric.generate_predictions` with a deterministic stub
that returns canned outputs. The test then drives the CLI's `main()` and
asserts:
  - overall accuracy matches what verify() says about our canned outputs
  - per-label breakdown is emitted and correct
  - debug_predictions_with_labels.csv is written and joinable on id

This catches plumbing regressions (val schema, temp-dir staging, debug CSV
roundtrip, --debug flag, per-label aggregation) without needing a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from nemotron_local_cv import __main__ as cli
from nemotron_local_cv import host_metric


CANNED_PREDICTIONS = {
    "v01": "10011000",       # binary, exact → correct
    "v02": "10011001",       # binary, off by one → wrong
    "v03": "24.6401",        # numeric within tol → correct
    "v04": "260",            # numeric outside tol → wrong
    "v05": "XLVII",          # string, case differs → correct (case-insensitive)
    "v06": "dog",            # string mismatch → wrong
}

VAL_ROWS = [
    {"id": "v01", "prompt": "p1", "answer": "10011000",  "label": "bit_manipulation"},
    {"id": "v02", "prompt": "p2", "answer": "10011000",  "label": "bit_manipulation"},
    {"id": "v03", "prompt": "p3", "answer": "24.64",     "label": "gravitational_constant"},
    {"id": "v04", "prompt": "p4", "answer": "250",       "label": "gravitational_constant"},
    {"id": "v05", "prompt": "p5", "answer": "xlvii",     "label": "numeral_conversion"},
    {"id": "v06", "prompt": "p6", "answer": "cat",       "label": "text_encryption"},
]

EXPECTED_CORRECT_IDS = {"v01", "v03", "v05"}
EXPECTED_OVERALL_ACCURACY = 3 / 6
EXPECTED_PER_LABEL = {
    "bit_manipulation": (0.5, 2),
    "gravitational_constant": (0.5, 2),
    "numeral_conversion": (1.0, 1),
    "text_encryption": (0.0, 1),
}


def _stub_generate_predictions(
    test_df,
    lora_path,
    row_id_col,
    max_lora_rank,
    max_tokens,
    top_p,
    temperature,
    max_num_seqs,
    gpu_memory_utilization,
    max_model_len,
    debug=False,
):
    """Drop-in for host_metric.generate_predictions — no GPU, no vLLM."""
    rows = []
    debug_rows = []
    for item in test_df.itertuples(index=False):
        row_id = getattr(item, row_id_col)
        pred = CANNED_PREDICTIONS[row_id]
        # Wrap in `\boxed{...}` so the real extract_final_answer would extract
        # the same value if this were a true LLM output. Belt-and-braces.
        raw = f"reasoning blah blah\n\\boxed{{{pred}}}"
        extracted = host_metric.extract_final_answer(raw)
        rows.append({row_id_col: row_id, "prediction": extracted})
        if debug:
            debug_rows.append(
                {
                    row_id_col: row_id,
                    "raw_output": raw,
                    "extracted_prediction": extracted,
                }
            )
    if debug and debug_rows:
        # The real function writes to CWD; mirror that contract exactly.
        pd.DataFrame(debug_rows).to_csv("debug_predictions.csv", index=False)
    return pd.DataFrame(rows)


@pytest.fixture
def val_csv(tmp_path: Path) -> Path:
    path = tmp_path / "val_mini.csv"
    pd.DataFrame(VAL_ROWS).to_csv(path, index=False)
    return path


def test_cli_roundtrip_with_debug(tmp_path, monkeypatch, capsys, val_csv):
    monkeypatch.setattr(host_metric, "generate_predictions", _stub_generate_predictions)
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "adapter_single"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / "adapter_model.safetensors").write_text("fake")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_local_cv",
            "--adapter", str(adapter_dir),
            "--val", str(val_csv),
            "--model", "fake-model-id",
            "--debug",
        ],
    )

    rc = cli.main()
    assert rc == 0

    out = capsys.readouterr().out
    assert f"overall accuracy: {EXPECTED_OVERALL_ACCURACY:.4f}" in out
    assert "per-label accuracy:" in out
    for label in EXPECTED_PER_LABEL:
        assert label in out

    archive = tmp_path / "debug_predictions_with_labels.csv"
    assert archive.exists(), "expected debug archive not written"

    archived = pd.read_csv(archive)
    assert len(archived) == len(VAL_ROWS)
    assert set(archived.columns) >= {
        "id",
        "prompt",
        "answer",
        "label",
        "extracted_prediction",
        "is_correct",
    }

    correct_ids = set(archived.loc[archived["is_correct"], "id"].astype(str))
    assert correct_ids == EXPECTED_CORRECT_IDS

    per_label = archived.groupby("label")["is_correct"].agg(["mean", "count"]).to_dict("index")
    for label, (mean, count) in EXPECTED_PER_LABEL.items():
        assert per_label[label]["mean"] == pytest.approx(mean)
        assert per_label[label]["count"] == count


def test_cli_roundtrip_without_debug(tmp_path, monkeypatch, capsys, val_csv):
    monkeypatch.setattr(host_metric, "generate_predictions", _stub_generate_predictions)
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "adapter_single"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / "adapter_model.safetensors").write_text("fake")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_local_cv",
            "--adapter", str(adapter_dir),
            "--val", str(val_csv),
            "--model", "fake-model-id",
        ],
    )

    rc = cli.main()
    assert rc == 0

    out = capsys.readouterr().out
    assert f"overall accuracy: {EXPECTED_OVERALL_ACCURACY:.4f}" in out
    # Without --debug we still print accuracy but skip per-label breakdown.
    assert "re-run with --debug" in out
    # And no archive file should appear in CWD.
    assert not (tmp_path / "debug_predictions_with_labels.csv").exists()


def test_cli_rejects_missing_columns(tmp_path, monkeypatch, val_csv):
    bad_csv = tmp_path / "no_label.csv"
    pd.DataFrame(
        [{"id": "x", "prompt": "p", "answer": "a"}]  # missing 'label'
    ).to_csv(bad_csv, index=False)

    monkeypatch.setattr(host_metric, "generate_predictions", _stub_generate_predictions)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_local_cv",
            "--adapter", "/fake/adapter/path",
            "--val", str(bad_csv),
            "--model", "fake-model-id",
        ],
    )

    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert "label" in str(ei.value)


def test_cli_batch_manifest_writes_ranked_summary(tmp_path, monkeypatch, capsys, val_csv):
    alpha = tmp_path / "adapter_alpha"
    alpha.mkdir()
    (alpha / "adapter_config.json").write_text("{}")
    (alpha / "adapter_model.safetensors").write_text("fake")
    beta = tmp_path / "adapter_beta"
    beta.mkdir()
    (beta / "adapter_config.json").write_text("{}")
    (beta / "adapter_model.safetensors").write_text("fake")

    def _stub_score(*args, **kwargs):
        submission = kwargs.get("submission", args[1] if len(args) > 1 else None)
        adapter = submission["prediction"].iloc[0]
        return {
            str(beta): 0.55,
            str(alpha): 0.83,
        }[adapter]

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"name": "beta", "adapter": str(beta)},
            {"name": "alpha", "adapter": str(alpha)},
        ]
    ).to_csv(manifest, index=False)

    summary = tmp_path / "summary.csv"
    monkeypatch.setattr(host_metric, "score", _stub_score)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_local_cv",
            "--adapter-manifest",
            str(manifest),
            "--summary-out",
            str(summary),
            "--val",
            str(val_csv),
            "--model",
            "fake-model-id",
        ],
    )

    rc = cli.main()
    assert rc == 0
    assert summary.exists()

    ranked = pd.read_csv(summary)
    assert ranked["name"].tolist() == ["alpha", "beta"]
    assert ranked["overall_accuracy"].tolist() == pytest.approx([0.83, 0.55])

    out = capsys.readouterr().out
    assert "ranked adapters" in out


def test_cli_batch_manifest_resolves_nested_adapter_path(
    tmp_path, monkeypatch, capsys, val_csv
):
    nested = tmp_path / "models" / "candidate_a" / "adapter" / "final"
    nested.mkdir(parents=True)
    (nested / "adapter_config.json").write_text("{}")
    (nested / "adapter_model.safetensors").write_text("fake")

    def _stub_score(*args, **kwargs):
        submission = kwargs.get("submission", args[1] if len(args) > 1 else None)
        adapter = submission["prediction"].iloc[0]
        assert adapter == str(nested)
        return 0.77

    manifest = tmp_path / "manifest_nested.csv"
    pd.DataFrame(
        [{"name": "candidate_a", "adapter": str(tmp_path / "models" / "candidate_a")}]
    ).to_csv(manifest, index=False)

    summary = tmp_path / "summary_nested.csv"
    monkeypatch.setattr(host_metric, "score", _stub_score)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_local_cv",
            "--adapter-manifest",
            str(manifest),
            "--summary-out",
            str(summary),
            "--val",
            str(val_csv),
            "--model",
            "fake-model-id",
        ],
    )

    rc = cli.main()
    assert rc == 0
    ranked = pd.read_csv(summary)
    assert ranked["overall_accuracy"].tolist() == pytest.approx([0.77])

    out = capsys.readouterr().out
    assert "resolved adapter path:" in out


def test_cli_batch_manifest_skips_missing_adapter_path(
    tmp_path, monkeypatch, capsys, val_csv
):
    called = {"score": 0}

    def _stub_score(*args, **kwargs):
        called["score"] += 1
        return 0.99

    manifest = tmp_path / "manifest_missing.csv"
    pd.DataFrame(
        [{"name": "broken_adapter", "adapter": str(tmp_path / "does_not_exist")}]
    ).to_csv(manifest, index=False)

    summary = tmp_path / "summary_missing.csv"
    monkeypatch.setattr(host_metric, "score", _stub_score)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_local_cv",
            "--adapter-manifest",
            str(manifest),
            "--summary-out",
            str(summary),
            "--val",
            str(val_csv),
            "--model",
            "fake-model-id",
        ],
    )

    rc = cli.main()
    assert rc == 0
    assert called["score"] == 0

    ranked = pd.read_csv(summary)
    assert ranked["status"].tolist() == ["missing"]
    assert "invalid adapter path" in ranked.loc[0, "error"]

    out = capsys.readouterr().out
    assert "skipped: invalid adapter path" in out
