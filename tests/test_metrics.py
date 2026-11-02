import csv
from dataclasses import replace

import pytest
import torch

from src.artifacts import (CheckpointRef, EvaluationResult, ModelRef, NLLStats, PredictorResult,
                           RetentionResult, RunSpec, SelectionDecision, SelectionEntry, SelectionResult)
from src.analysis.metrics import (_probe_tensor, assess_predictor, measure_retention,
                                  write_validation_trajectories)


def metric_case():
    model = ModelRef("U", "models/U", "revision", "tokenizer", "tok-rev", "bfloat16")
    checkpoint = CheckpointRef(model, "r", 2, 1)
    selection = SelectionResult("complete", SelectionDecision(4, 0.1, "fixed"),
                                [SelectionEntry("r", 1e-5, 0, checkpoint, 4, 2, 0)],
                                original_validation_nll=2, original_model_id="T")

    def record(model_id, variant, forget, retain):
        return EvaluationResult(f"{model_id}-{variant}", "attempt", model_id, variant, "settings",
                                {"forget_test": "forget", "retain_test": "retain"}, "complete",
                                {"forget_test": NLLStats(forget * 10, 10),
                                 "retain_test": NLLStats(retain * 10, 10)},
                                "bfloat16", "float32" if variant == "quantized" else "bfloat16",
                                {"torch": "fixed"}, tokenizer_id="tokenizer", tokenizer_revision="tok-rev",
                                quantization_id="quant-settings" if variant == "quantized" else "none")

    measurements = [record("T", "unquantized", 2, 2), record("T", "quantized", 3, 3),
                    record("U", "unquantized", 4, 2.2), record("U", "quantized", 7, 3.4)]
    return selection, measurements


def test_probe_requires_both_directions_and_never_mutates_original():
    tensor = torch.tensor([1.0, 2.0, 0.0], dtype=torch.bfloat16)
    original = tensor.clone()
    counts, total = _probe_tensor(tensor, [0.003], "cpu", chunk_size=1)
    assert counts == [1]
    assert total == 3
    assert torch.equal(tensor, original)


def test_predictor_calculation_identity_serializes_torch_version(monkeypatch, tmp_path):
    from src.analysis import metrics
    from src.eval import evaluator

    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    original = ModelRef("T", "models/T", "revision", "tokenizer", "tok-rev", "bfloat16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(evaluator, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(metrics, "_probe_tensor", lambda *args, **kwargs: ([1], 1))
    result = metrics.compute_predictor(original, [1e-5], {"parameter_dtype": "bfloat16"}, tmp_path)
    assert len(result[0].calculation_id) == 64
    assert result[0].value == 1


def test_retention_uses_quantized_baseline_and_does_not_clamp():
    selection, records = metric_case()
    result = measure_retention(selection, records)[0]
    assert result.status == "complete"
    assert result.unquantized_gap == 2
    assert result.quantized_gap == 4
    assert result.retention_ratio == 2
    assert result.retain_unquantized_delta == pytest.approx(0.2)
    assert result.retain_quantized_delta == pytest.approx(0.4)
    records[-1] = replace(records[-1], nll={**records[-1].nll, "forget_test": NLLStats(20, 10)})
    assert measure_retention(selection, records)[0].retention_ratio == -0.5


@pytest.mark.parametrize("gap", [0, -1, 1e-10])
def test_invalid_denominator_preserves_raw_evidence(gap):
    selection, records = metric_case()
    records[2] = replace(records[2], nll={**records[2].nll, "forget_test": NLLStats((2 + gap) * 10, 10)})
    result = measure_retention(selection, records)[0]
    assert result.status == "invalid"
    assert result.retention_ratio is None
    assert result.unquantized_gap == pytest.approx(gap, abs=1e-14)
    assert result.nll_t == 2 and result.nll_qu == 7
    assert result.quantized_gap == 4
    assert result.retain_unquantized_delta == pytest.approx(0.2)


@pytest.mark.parametrize("change", [
    {"dataset_ids": {"forget_test": "other", "retain_test": "retain"}},
    {"settings_id": "other"}, {"tokenizer_revision": "other"},
    {"compute_dtype": "bfloat16"}, {"quantization_id": "other"},
    {"software_versions": {"torch": "other"}},
    {"nll": {"forget_test": NLLStats(70, 10)}},
    {"nll": {"forget_test": NLLStats(70, 10), "retain_test": NLLStats(34, 20)}},
])
def test_quartet_requires_compatible_data_tokenizer_precision_and_versions(change):
    selection, records = metric_case()
    records[-1] = replace(records[-1], **change)
    result = measure_retention(selection, records)[0]
    assert result.status == "invalid"
    assert result.retention_ratio is None
    assert result.failure_reason


def test_missing_variant_is_explicit_and_retries_are_not_extra_samples():
    selection, records = metric_case()
    records.append(replace(records[-1], attempt_id="retry"))
    assert len(measure_retention(selection, records)) == 1
    assert measure_retention(selection, records)[0].status == "complete"
    result = measure_retention(selection, records[:3])[0]
    assert result.status == "invalid" and result.retention_ratio is None


def test_assessment_reports_one_seed_and_lr_level_summaries(tmp_path):
    selection, records = metric_case()
    result = measure_retention(selection, records)[0]
    predictor = PredictorResult(1e-5, 0.5, "T", "linears", "bfloat16", "float32", "calculation")
    report = assess_predictor([predictor], [result], selection, tmp_path)
    assert report.status == "complete"
    assert report.summary["comparable_learning_rate_count"] == 1
    assert report.summary["pearson_e_retention"] is None
    assert any("seed variability cannot be estimated" in line for line in report.summary["limitations"])
    with (tmp_path / "learning_rate_summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["valid_seed_count"] == "1" and row["retention_sd"] == ""
    assert (tmp_path / "predictor_assessment.png").exists()
    assert (tmp_path / "predictor_assessment.svg").exists()


def test_validation_trajectories_include_missing_checkpoint_evidence(tmp_path):
    selection, records = metric_case()
    checkpoint = selection.selections[0].checkpoint
    original = replace(checkpoint.model, model_id="T")
    run = RunSpec("r", 1e-5, 0, original, {}, {}, [2], "runs/r")
    report = write_validation_trajectories([run], [checkpoint], [], tmp_path)
    with (tmp_path / "validation_trajectories.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["forget_validation_nll"] == ""
    assert len(report["figure_paths"]) == 2
