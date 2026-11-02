"""Small artifact-backed coordinator; every selection precedes blind evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import math
from pathlib import Path
import re
import time
import traceback
import uuid

import yaml

from src.artifacts import (AnalysisResult, EvaluationRequest, EvaluationResult, ExperimentConfig,
    ModelRef, PredictorResult, RunResult, RunSpec, SelectionDecision, SweepIndex,
    publish_record, read_record, resolve_artifact_path, stable_id)
from src.analysis.metrics import assess_predictor, compute_predictor, measure_retention, write_validation_trajectories
from src.analysis.selection import select_checkpoints
from src.eval.evaluator import _versions, evaluate
from src.inputs import load_data, prepare_data
from src.training.unlearn import _validate_settings, checkpoint_schedule, train_run

LOG = logging.getLogger(__name__)


def load_config(config_file: Path, project_root: Path) -> ExperimentConfig:
    root = Path(project_root).resolve()
    with Path(config_file).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    sections = ("experiment", "execution", "model", "data", "sweep", "training",
                "checkpointing", "target_selection", "evaluation", "quantization", "predictor")
    if not isinstance(raw, dict) or set(raw) != set(sections):
        raise ValueError("Config must contain exactly the documented named sections")
    if any(not isinstance(raw[name], dict) for name in sections):
        raise ValueError("Every named config section must be a mapping")
    _validate_settings(raw["training"])
    execution = raw["execution"]
    budget = execution.get("max_cost_usd")
    reserve = execution.get("shutdown_reserve_usd")
    hourly = execution.get("max_gpu_cost_per_hour")
    if (type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 15
            or type(reserve) not in (int, float) or not math.isfinite(reserve) or not 0 < reserve < budget
            or type(hourly) not in (int, float) or not math.isfinite(hourly) or hourly <= 0):
        raise ValueError("Execution requires finite positive budget <= $15, shutdown reserve < budget, and hourly cap")
    if type(execution.get("parallel_eval_on_separate_machine")) is not bool:
        raise ValueError("parallel_eval_on_separate_machine must be boolean")
    timeout = execution.get("evaluation_timeout_seconds", 3600)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Evaluation timeout must be finite and positive")
    for key in ("revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(raw["model"].get(key, ""))):
            raise ValueError(f"model.{key} requires a pinned 40-character commit SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", str(raw["data"].get("revision", ""))):
        raise ValueError("data.revision requires a pinned commit SHA")
    for key in ("source", "tokenizer"):
        if not isinstance(raw["model"].get(key), str) or not raw["model"][key]:
            raise ValueError(f"model.{key} is required")
    rates, seeds = raw["sweep"]["learning_rates"], raw["sweep"]["seeds"]
    if (not rates or any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates)
            or len(set(rates)) != len(rates)):
        raise ValueError("Learning rates must be nonempty, unique, finite and positive")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty, unique, unsigned 32-bit integers")
    evaluation = raw["evaluation"]
    max_length = raw["data"].get("max_sequence_length")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("Data max_sequence_length must be an integer >= 2")
    for key in ("max_validation_samples", "max_test_samples"):
        if key in evaluation and (type(evaluation[key]) is not int or evaluation[key] <= 0):
            raise ValueError(f"{key} must be a positive integer")
    expected = {"metric": "causal_lm_nll", "reduction": "mean_over_non_padding_target_tokens",
                "log_base": "e", "inference_dtype": "bfloat16", "fixed_samples_across_runs": True}
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported evaluation metric or precision settings")
    if type(evaluation["per_device_batch_size"]) is not int or evaluation["per_device_batch_size"] <= 0:
        raise ValueError("Evaluation batch size must be positive")
    checkpoint_schedule(1, raw["training"]["per_device_batch_size"],
        raw["training"]["gradient_accumulation_steps"], raw["training"]["max_epochs"],
        evaluation["validation_every_epochs"], evaluation.get("early_validation_steps", []))
    quantization = raw["quantization"]
    required_quantization = {"backend": "bitsandbytes", "load_in_4bit": True,
        "bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": False,
        "bnb_4bit_compute_dtype": "float32", "non_quantized_parameter_dtype": "bfloat16",
        "calibration_required": False}
    if quantization != required_quantization:
        raise ValueError("Only the explicit bitsandbytes FP4 recipe is supported")
    if not raw["checkpointing"].get("save_final") or not raw["checkpointing"].get("save_on_validation_schedule"):
        raise ValueError("Scheduled and final checkpoints are required")
    selection = raw["target_selection"]
    gain, tolerance = selection.get("minimum_gain", 0.25), selection.get("max_abs_error", 0.1)
    if (type(gain) not in (int, float) or not math.isfinite(gain) or gain <= 0
            or type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Target selection requires finite positive minimum_gain and nonnegative max_abs_error")
    if (selection.get("timing") != "after_all_sweep_runs"
            or selection.get("model") != "unquantized"
            or not selection.get("freeze_before_quantized_evaluation")
            or not selection.get("freeze_before_test_evaluation")):
        raise ValueError("Selection must use validation and freeze before quantized/test evaluation")
    for section, key in (("experiment", "output_dir"), ("model", "path")):
        path = Path(raw[section][key])
        raw[section][key] = str(path.resolve() if path.is_absolute() else (root / path).resolve())
    for split in (f"{domain}_{part}" for domain in ("forget", "retain") for part in ("train", "validation", "test")):
        path = Path(raw["data"][split])
        raw["data"][split] = str(path.resolve() if path.is_absolute() else (root / path).resolve())
    return ExperimentConfig(project_root=str(root), **raw)


def _original(config):
    identity = {key: config.model[key] for key in ("source", "revision", "tokenizer", "tokenizer_revision")}
    identity["storage_dtype"] = config.training["parameter_dtype"]
    return ModelRef(stable_id(identity), config.model["source"], config.model["revision"],
                    config.model["tokenizer"], config.model["tokenizer_revision"],
                    config.training["parameter_dtype"], source="huggingface")


def _request(model, datasets, variant, config):
    settings = stable_id({"evaluation": config.evaluation, "quantization": config.quantization,
                          "original_model": _original(config)})
    job = stable_id({"model": model, "datasets": datasets, "variant": variant, "settings_id": settings})
    return EvaluationRequest(job, "attempt-001", model, datasets, variant,
                             config.evaluation, config.quantization, settings)


def _compatible(result, request):
    expected_compute = "float32" if request.variant == "quantized" else "bfloat16"
    return (result.status == "complete" and result.job_id == request.job_id
        and result.model_id == request.model.model_id and result.variant == request.variant
        and result.settings_id == request.settings_id
        and result.dataset_ids == {key: ref.dataset_id for key, ref in request.datasets.items()}
        and set(result.nll) == set(request.datasets)
        and result.inference_dtype == "bfloat16" and result.compute_dtype == expected_compute
        and result.tokenizer_id == request.model.tokenizer_id
        and result.tokenizer_revision == request.model.tokenizer_revision
        and result.quantization_id == (stable_id(request.quantization) if request.variant == "quantized" else "none")
        and result.software_versions == _versions())


def _attempt(request, root):
    requests = [read_record(path, EvaluationRequest) for path in
                sorted((root / "records/evaluation_request").glob(f"{request.job_id}--*.json"))]
    for previous in requests:
        result_path = root / "records/evaluation_result" / f"{previous.job_id}--{previous.attempt_id}.json"
        if result_path.exists() and _compatible(read_record(result_path, EvaluationResult), request):
            return previous, result_path
    pending = [previous for previous in requests if not
               (root / "records/evaluation_result" / f"{previous.job_id}--{previous.attempt_id}.json").exists()]
    if pending:
        return pending[-1], None
    return replace(request, attempt_id=f"attempt-{len(requests) + 1:03d}"), None


def _publish_request(request, root):
    if request.model.source in ("artifact", "local"):
        bundle = resolve_artifact_path(root, request.model.location)
        required = ("config.json", "tokenizer_config.json", "checkpoint_complete.json")
        if not all((bundle / name).is_file() for name in required):
            raise ValueError(f"Checkpoint bundle is incomplete: {bundle}")
        index = bundle / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            if not shards or any(not resolve_artifact_path(bundle, name).is_file() for name in shards):
                raise ValueError(f"Checkpoint shard missing: {bundle}")
        elif not (bundle / "model.safetensors").is_file():
            raise ValueError(f"Checkpoint model weights missing: {bundle}")
    request, result = _attempt(request, root)
    publish_record(request, root)
    return request, result


def _execute(request, config, root):
    request, existing = _publish_request(request, root)
    if existing is not None:
        return existing
    if config.execution.get("parallel_eval_on_separate_machine", False):
        timeout = float(config.execution.get("evaluation_timeout_seconds", 3600))
        deadline = time.monotonic() + timeout
        path = root / "records/evaluation_result" / f"{request.job_id}--{request.attempt_id}.json"
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Evaluation unfinished after {timeout}s: {request.job_id}")
            time.sleep(2)
        return path
    return publish_record(evaluate(request, config.execution.get("device", "cuda:0"), root), root)


def _snapshot(config, datasets, runs, expected, run_paths, eval_paths, predictor_paths,
              status, root, failure=None):
    record = SweepIndex(uuid.uuid4().hex, config, datasets, runs, sorted(set(expected)),
        sorted(set(run_paths)), sorted(set(eval_paths)), predictor_paths, status, _versions(), failure)
    return publish_record(record, root)


def _relative(path, root):
    return Path(path).resolve().relative_to(root).as_posix()


def run_sweep(config: ExperimentConfig, artifact_root: Path) -> Path:
    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    publish_record(config, root)
    datasets, runs, expected, run_paths, eval_paths, predictor_paths = {}, [], [], [], [], []
    try:
        datasets = prepare_data(config, root)
        for split, reference in datasets.items():
            manifest = json.loads(resolve_artifact_path(root, reference.manifest).read_text(encoding="utf-8"))
            lengths = [sample["end_token"] - sample["start_token"] for sample in manifest["samples"]]
            LOG.info("Prepared split=%s samples=%d input_tokens=%d target_tokens=%d", split,
                     manifest["sample_count"], sum(lengths), sum(max(0, length - 1) for length in lengths))
        original = _original(config)
        publish_record(original, root)
        train_data = {key: datasets[key] for key in ("forget_train", "retain_train")}
        steps = checkpoint_schedule(len(load_data(datasets["forget_train"], root)),
            config.training["per_device_batch_size"], config.training["gradient_accumulation_steps"],
            config.training["max_epochs"], config.evaluation["validation_every_epochs"],
            config.evaluation.get("early_validation_steps", []))
        LOG.info("Fixed checkpoint_schedule=%s total_optimizer_steps=%d epochs=%d expected_runs=%d",
                 steps, steps[-1], config.training["max_epochs"],
                 len(config.sweep["learning_rates"]) * len(config.sweep["seeds"]))
        for rate in config.sweep["learning_rates"]:
            for seed in config.sweep["seeds"]:
                training = {**config.training, "device": config.execution.get("device", "cuda:0")}
                identity = stable_id({"original": original, "data": train_data, "training": training,
                                      "lr": rate, "seed": seed, "checkpoint_steps": steps})
                run = RunSpec(identity, float(rate), seed, original, train_data, training,
                              steps, f"checkpoints/{identity}")
                publish_record(run, root)
                runs.append(run)
        if config.predictor.get("enabled", True):
            for prediction in compute_predictor(original, list(config.sweep["learning_rates"]), config.training, root):
                predictor_paths.append(_relative(publish_record(prediction, root), root))
        validation = {key: value for key, value in datasets.items() if key.endswith("_validation")}
        baseline = _request(original, validation, "unquantized", config)
        expected.append(baseline.job_id)
        path = _execute(baseline, config, root)
        eval_paths.append(_relative(path, root))
        if not _compatible(read_record(path, EvaluationResult), baseline):
            raise RuntimeError("Original validation evaluation failed or is incompatible")
        for run in runs:
            outcome_path = root / "records/run_result" / f"{run.run_id}.json"
            def ready(checkpoint):
                request = _request(checkpoint.model, validation, "unquantized", config)
                expected.append(request.job_id)
                _publish_request(request, root)
            if outcome_path.exists():
                outcome = read_record(outcome_path, RunResult)
            else:
                outcome = train_run(run, root, ready)
                outcome_path = publish_record(outcome, root)
            run_paths.append(_relative(outcome_path, root))
            # Reconcile every complete saved reference, including callback failures.
            requests = []
            for checkpoint in outcome.checkpoints:
                request = _request(checkpoint.model, validation, "unquantized", config)
                expected.append(request.job_id)
                _publish_request(request, root)
                requests.append(request)
            # train_run has released model and optimizer before sequential inference.
            for request in requests:
                path = _execute(request, config, root)
                eval_paths.append(_relative(path, root))
                if not _compatible(read_record(path, EvaluationResult), request):
                    raise RuntimeError(f"Checkpoint validation failed or is incompatible: {request.job_id}")
            _snapshot(config, datasets, runs, expected, run_paths, eval_paths, predictor_paths,
                      "running", root)
            if outcome.status != "complete" or {cp.step for cp in outcome.checkpoints} != set(run.checkpoint_steps):
                raise RuntimeError(f"Run incomplete: {run.run_id}: {outcome.failure_reason}")
        return _snapshot(config, datasets, runs, expected, run_paths, eval_paths, predictor_paths,
                         "complete", root)
    except Exception:
        failure = traceback.format_exc()
        LOG.exception("Sweep incomplete")
        return _snapshot(config, datasets, runs, expected, run_paths, eval_paths, predictor_paths,
                         "failed", root, failure)


def _evidence(index, root):
    outcomes = [read_record(resolve_artifact_path(root, path), RunResult) for path in index.run_result_paths]
    evaluations = [read_record(resolve_artifact_path(root, path), EvaluationResult) for path in index.evaluation_result_paths]
    checkpoints = [checkpoint for outcome in outcomes for checkpoint in outcome.checkpoints]
    return outcomes, evaluations, checkpoints


def _failed_report(index: SweepIndex, artifact_root: Path, reason=None) -> AnalysisResult:
    """Preserve incomplete validation evidence without inventing a target decision."""
    root = Path(artifact_root).resolve()
    failure = "\n".join(part for part in (index.failure_reason, reason) if part) or "Sweep did not complete"
    outcomes, validation, checkpoints = _evidence(index, root)
    report = root / "analysis" / f"failed-{index.snapshot_id}"
    report.mkdir(parents=True, exist_ok=True)
    curves = write_validation_trajectories(index.runs, checkpoints, validation, report)
    complete_jobs = {result.job_id for result in validation if result.status == "complete"}
    summary = {"status": "failed", "failure_reason": failure,
               "expected_run_count": len(index.runs), "recorded_run_count": len(outcomes),
               "missing_successful_validation_jobs": sorted(set(index.expected_job_ids) - complete_jobs),
               "target_selected": False}
    summary_path = report / "failure.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = AnalysisResult(status="failed", measurement_paths=index.evaluation_result_paths,
        table_paths=[_relative(path, root) for path in curves["table_paths"] + [summary_path]],
        figure_paths=[_relative(path, root) for path in curves["figure_paths"]],
        summary=summary, failure_reason=failure)
    publish_record(result, root)
    return result


def choose_decision(index: SweepIndex, artifact_root: Path) -> SelectionDecision:
    """Choose minimax nearest-checkpoint error using only unquantized validation."""
    root = Path(artifact_root).resolve()
    outcomes, evaluations, checkpoints = _evidence(index, root)
    gain = float(index.config.target_selection.get("minimum_gain", 0.25))
    tolerance = float(index.config.target_selection.get("max_abs_error", 0.10))
    if not math.isfinite(gain) or gain <= 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Automatic selection requires positive finite gain and nonnegative finite tolerance")
    original = _original(index.config)
    success = {result.model_id: result for result in evaluations if result.status == "complete"
               and result.variant == "unquantized" and "forget_validation" in result.nll}
    baseline = success.get(original.model_id)
    if baseline is None:
        raise ValueError("Cannot choose a target without original unquantized validation")
    base = baseline.nll["forget_validation"].mean
    curves = [[success[cp.model.model_id].nll["forget_validation"].mean for cp in checkpoints
               if cp.run_id == run.run_id and cp.model.model_id in success] for run in index.runs]
    target = base + gain
    if curves and all(curves):
        lower = max(target, max(min(curve) for curve in curves))
        upper = min(max(curve) for curve in curves)
        if lower <= upper:
            # Minimax solutions occur at observed values, interval boundaries, or
            # pairwise midpoints of observed values. Include those breakpoints.
            values = sorted({value for curve in curves for value in curve})
            candidates = {lower, upper}
            candidates.update(value for value in values if lower <= value <= upper)
            candidates.update((a + b) / 2 for i, a in enumerate(values) for b in values[i + 1:]
                              if lower <= (a + b) / 2 <= upper)
            feasible = []
            for value in candidates:
                candidate = SelectionDecision(value, tolerance, "Validation-only candidate", gain)
                matched = select_checkpoints(index.runs, checkpoints, evaluations, candidate,
                    run_results=outcomes, expected_validation_job_ids=index.expected_job_ids)
                if matched.status == "complete":
                    feasible.append(value)
            target = min(feasible or candidates, key=lambda value: (
                max(min(abs(point - value) for point in curve) for curve in curves), value))
    reason = ("Automatic target frozen from completed unquantized forget-validation curves only; "
              "minimize worst nearest saved-checkpoint mismatch across every expected run, "
              "then choose lower target on ties; no test, quantized, or predictor values used.")
    return SelectionDecision(target, tolerance, reason, minimum_gain=gain)


def finalize_sweep(sweep_index: Path, artifact_root: Path, decision: SelectionDecision) -> AnalysisResult:
    root = Path(artifact_root).resolve()
    index = read_record(sweep_index, SweepIndex)
    config = index.config
    outcomes, validation, checkpoints = _evidence(index, root)
    selection = select_checkpoints(index.runs, checkpoints, validation, decision,
        run_results=outcomes, expected_validation_job_ids=index.expected_job_ids)
    selection_path = publish_record(selection, root)
    # This publication is the freeze boundary. No quantized or test request exists above it.
    report = root / "analysis" / stable_id(selection)
    report.mkdir(parents=True, exist_ok=True)
    curves = write_validation_trajectories(index.runs, checkpoints, validation, report)
    predictions = [read_record(resolve_artifact_path(root, path), PredictorResult) for path in index.predictor_paths]
    if selection.status != "complete":
        result = assess_predictor(predictions, [], selection, report)
        result = replace(result, selection_path=_relative(selection_path, root),
                         table_paths=[_relative(path, root) for path in result.table_paths + curves["table_paths"]],
                         figure_paths=[_relative(path, root) for path in result.figure_paths + curves["figure_paths"]],
                         measurement_paths=index.evaluation_result_paths)
        publish_record(result, root)
        return result
    tests = {key: value for key, value in index.datasets.items() if key.endswith("_test")}
    models = {_original(config).model_id: _original(config)}
    models.update({entry.checkpoint.model.model_id: entry.checkpoint.model for entry in selection.selections})
    measurements, paths = [], []
    for model in models.values():
        for variant in ("unquantized", "quantized"):
            request = _request(model, tests, variant, config)
            path = _execute(request, config, root)
            result = read_record(path, EvaluationResult)
            if not _compatible(result, request):
                result = replace(result, status="failed", failure_reason=result.failure_reason or "Incompatible actual evaluation identity")
            measurements.append(result)
            paths.append(_relative(path, root))
    retention = measure_retention(selection, measurements)
    for record in retention:
        publish_record(record, root)
    result = assess_predictor(predictions, retention, selection, report)
    result = replace(result, selection_path=_relative(selection_path, root), measurement_paths=paths,
                     table_paths=[_relative(path, root) for path in result.table_paths + curves["table_paths"]],
                     figure_paths=[_relative(path, root) for path in result.figure_paths + curves["figure_paths"]])
    publish_record(result, root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--action", choices=("sweep", "finalize", "all"), default="all")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--target", type=float)
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--reason", default="Supplied runtime target from unquantized validation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(args.config, args.root)
    root = (args.artifacts or Path(config.experiment["output_dir"])).resolve()
    index_path = run_sweep(config, root) if args.action in ("sweep", "all") else args.index
    if index_path is None:
        parser.error("--index is required for finalize")
    LOG.info("Sweep index: %s", index_path)
    index = read_record(index_path, SweepIndex)
    if index.status != "complete":
        result = _failed_report(index, root)
        LOG.error("Analysis status=failed reason=%s", result.failure_reason)
        raise SystemExit(2)
    if args.action in ("finalize", "all"):
        if args.target is None:
            try:
                decision = choose_decision(index, root)
            except ValueError:
                result = _failed_report(index, root, traceback.format_exc())
                LOG.error("Analysis status=failed reason=%s", result.failure_reason)
                raise SystemExit(2)
        else:
            decision = SelectionDecision(args.target,
                args.max_error if args.max_error is not None else float(config.target_selection.get("max_abs_error", 0.1)),
                args.reason, minimum_gain=float(config.target_selection.get("minimum_gain", 0.25)))
        result = finalize_sweep(index_path, root, decision)
        LOG.info("Analysis status=%s summary=%s", result.status, result.summary)
        if result.status != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
