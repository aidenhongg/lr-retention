"""Coordinator contracts with synthetic measurements and no model downloads."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from src.artifacts import (CheckpointRef, DatasetRef, EvaluationResult, ModelRef,
    NLLStats, PredictorResult, RunResult, SelectionDecision, SweepIndex,
    publish_record, read_record, stable_id)
from src.experiments import sweep


PROJECT = Path(__file__).resolve().parents[1]


def config(tmp_path):
    return sweep.load_config(PROJECT / "configs/experiment.yaml", tmp_path)


def test_load_config_resolves_paths_and_rejects_unpinned_weights(tmp_path):
    loaded = config(tmp_path)
    assert Path(loaded.data["forget_train"]).is_absolute()
    assert Path(loaded.model["path"]).is_absolute()
    assert loaded.execution["parallel_eval_on_separate_machine"] is False
    raw = yaml.safe_load((PROJECT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    raw["model"]["revision"] = "main"
    file = tmp_path / "bad.yaml"
    file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        sweep.load_config(file, tmp_path)


def setup_synthetic(monkeypatch, tmp_path):
    cfg = config(tmp_path)
    cfg = replace(cfg, sweep={"learning_rates": [1e-6, 1e-5], "seeds": [0]},
                  training=cfg.training | {"max_epochs": 1},
                  evaluation=cfg.evaluation | {"early_validation_steps": [1], "validation_every_epochs": 0.5})
    refs = {f"{domain}_{part}": DatasetRef(f"{domain}-{part}", f"tokens/{domain}-{part}.jsonl",
            part, f"tokens/{domain}-{part}-manifest.json", cfg.model["tokenizer"],
            cfg.model["tokenizer_revision"], "preprocessing", 8)
            for domain in ("forget", "retain") for part in ("train", "validation", "test")}
    def prepare(config, root):
        for reference in refs.values():
            manifest = root / reference.manifest
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({"sample_count": 4,
                "samples": [{"start_token": 0, "end_token": 3}] * 4}), encoding="utf-8")
        return refs
    monkeypatch.setattr(sweep, "prepare_data", prepare)
    monkeypatch.setattr(sweep, "load_data", lambda *a: [None] * 4)
    monkeypatch.setattr(sweep, "compute_predictor", lambda model, rates, *a: [
        PredictorResult(rate, 1 - index * 0.1, model.model_id, "linear", "bfloat16", "float32", "calculation")
        for index, rate in enumerate(rates)])
    checkpoints = {}
    def train(run, root, callback):
        refs = []
        for step in run.checkpoint_steps:
            model = replace(run.original_model, model_id=f"{run.run_id}-{step}",
                            location=f"checkpoints/{run.run_id}/{step}", source="artifact")
            bundle = root / model.location
            bundle.mkdir(parents=True, exist_ok=True)
            for name in ("config.json", "tokenizer_config.json", "checkpoint_complete.json", "model.safetensors"):
                (bundle / name).write_text("{}", encoding="utf-8")
            ref = CheckpointRef(model, run.run_id, step, step / max(run.checkpoint_steps))
            checkpoints[model.model_id] = step
            publish_record(ref, root)
            refs.append(ref)
            callback(ref)
        return RunResult(run.run_id, "complete", refs, max(run.checkpoint_steps))
    monkeypatch.setattr(sweep, "train_run", train)
    calls = []
    def evaluate(request, device, root):
        testing = any(key.endswith("_test") for key in request.datasets)
        if testing or request.variant == "quantized":
            assert list((root / "records/selection_result").glob("*.json")), "Selection must already be frozen"
        calls.append((request.variant, tuple(request.datasets), request.model.model_id))
        step = checkpoints.get(request.model.model_id, 0)
        value = 2 + step * (0.25 if request.variant == "quantized" else 0.5)
        return EvaluationResult(request.job_id, request.attempt_id, request.model.model_id,
            request.variant, request.settings_id,
            {key: ref.dataset_id for key, ref in request.datasets.items()}, "complete",
            {key: NLLStats(value * 4, 4) for key in request.datasets}, "bfloat16",
            "float32" if request.variant == "quantized" else "bfloat16", sweep._versions(),
            tokenizer_id=request.model.tokenizer_id, tokenizer_revision=request.model.tokenizer_revision,
            quantization_id=stable_id(request.quantization) if request.variant == "quantized" else "none")
    monkeypatch.setattr(sweep, "evaluate", evaluate)
    return cfg, calls


def test_sweep_then_freeze_then_test_and_reuse_compatible_baselines(monkeypatch, tmp_path):
    cfg, calls = setup_synthetic(monkeypatch, tmp_path)
    root = tmp_path / "artifacts"
    index_path = sweep.run_sweep(cfg, root)
    index = read_record(index_path, SweepIndex)
    assert index.status == "complete"
    assert len(index.runs) == 2
    assert len(index.expected_job_ids) == len(index.evaluation_result_paths) == 5
    assert all(variant == "unquantized" and all(key.endswith("_validation") for key in splits)
               for variant, splits, model in calls)
    decision = sweep.choose_decision(index, root)
    assert decision.target_nll == pytest.approx(2.5)
    assert decision.minimum_gain == pytest.approx(0.25)
    result = sweep.finalize_sweep(index_path, root, decision)
    assert result.status == "complete"
    assert len(result.measurement_paths) == 6  # T/Q(T), two U/Q(U) pairs.
    before = len(calls)
    sweep.finalize_sweep(index_path, root, decision)
    assert len(calls) == before  # Successful compatible attempts reused once.


def test_unmatched_selection_publishes_no_test_or_quantized_jobs(monkeypatch, tmp_path):
    cfg, calls = setup_synthetic(monkeypatch, tmp_path)
    root = tmp_path / "artifacts"
    index = sweep.run_sweep(cfg, root)
    before = len(calls)
    result = sweep.finalize_sweep(index, root, SelectionDecision(20.0, 0.1, "outside overlap", 0.25))
    assert result.status == "unavailable"
    assert len(calls) == before
    assert result.figure_paths and result.table_paths


def test_result_reuse_rejects_wrong_actual_precision(monkeypatch, tmp_path):
    cfg, calls = setup_synthetic(monkeypatch, tmp_path)
    req = sweep._request(sweep._original(cfg), {}, "unquantized", cfg)
    good = EvaluationResult(req.job_id, req.attempt_id, req.model.model_id, req.variant,
        req.settings_id, {}, "complete", {}, "bfloat16", "bfloat16", sweep._versions(),
        tokenizer_id=req.model.tokenizer_id, tokenizer_revision=req.model.tokenizer_revision)
    assert sweep._compatible(good, req)
    assert not sweep._compatible(replace(good, compute_dtype="float32"), req)
    assert not sweep._compatible(replace(good, tokenizer_revision="other"), req)


@pytest.mark.parametrize("section,key,value", [
    ("execution", "max_cost_usd", 16.0), ("execution", "shutdown_reserve_usd", float("nan")),
    ("execution", "max_gpu_cost_per_hour", 0), ("target_selection", "minimum_gain", 0),
    ("target_selection", "max_abs_error", -0.1)])
def test_invalid_budget_or_target_fails_before_gpu(tmp_path, section, key, value):
    raw = yaml.safe_load((PROJECT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    raw[section][key] = value
    file = tmp_path / "invalid.yaml"
    file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        sweep.load_config(file, tmp_path)


@pytest.mark.parametrize("status", ["failed", "complete"])
def test_missing_original_validation_publishes_failed_report_without_target_or_tests(monkeypatch, tmp_path, status):
    cfg, calls = setup_synthetic(monkeypatch, tmp_path)
    root = tmp_path / "artifacts"
    index = read_record(sweep.run_sweep(cfg, root), SweepIndex)
    original_id = sweep._original(cfg).model_id
    paths = [path for path in index.evaluation_result_paths
             if read_record(root / path).model_id != original_id]
    reason = "Traceback: original validation failed before completion"
    failed = replace(index, snapshot_id=f"missing-baseline-{status}", status=status,
                     evaluation_result_paths=paths, failure_reason=reason if status == "failed" else None)
    index_path = publish_record(failed, root)
    before = len(calls)
    monkeypatch.setattr("sys.argv", ["sweep", "--config", str(PROJECT / "configs/experiment.yaml"),
        "--root", str(tmp_path), "--artifacts", str(root), "--action", "finalize", "--index", str(index_path)])
    with pytest.raises(SystemExit) as stopped:
        sweep.main()
    assert stopped.value.code == 2
    assert len(calls) == before
    assert not list((root / "records/selection_result").glob("*.json"))
    reports = [read_record(path) for path in (root / "records/analysis_result").glob("*.json")]
    assert len(reports) == 1 and reports[0].status == "failed"
    assert reports[0].summary["target_selected"] is False
    assert "Traceback" in reports[0].failure_reason
    assert all((root / path).exists() for path in reports[0].table_paths + reports[0].figure_paths)
    assert any(path.endswith("validation_trajectories.csv") for path in reports[0].table_paths)
