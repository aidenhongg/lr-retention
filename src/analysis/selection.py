"""Freeze checkpoint matches using completed, unquantized validation only."""

import math
from dataclasses import replace

from src.artifacts import SelectionEntry, SelectionResult


def _successful(results):
    jobs = {}
    for result in results:
        if result.status != "complete":
            continue
        previous = jobs.get(result.job_id)
        if previous is not None:
            fields = ("model_id", "variant", "settings_id", "dataset_ids", "nll",
                      "inference_dtype", "compute_dtype", "tokenizer_id", "tokenizer_revision",
                      "quantization_id", "software_versions")
            if any(getattr(previous, key) != getattr(result, key) for key in fields):
                raise ValueError(f"Conflicting successful attempts for {result.job_id}")
        jobs[result.job_id] = result
    return list(jobs.values())


def _identity(result, split):
    stats = result.nll[split]
    return (result.settings_id, result.dataset_ids[split], stats.token_count,
            result.inference_dtype, result.compute_dtype, result.tokenizer_id,
            result.tokenizer_revision, result.quantization_id,
            tuple(sorted(result.software_versions.items())))


def select_checkpoints(expected_runs, checkpoints, validation_results, decision, *,
                       run_results, expected_validation_job_ids):
    """Return every expected run, including explicit reasons when matching fails."""
    entries = [SelectionEntry(run.run_id, run.learning_rate, run.seed) for run in expected_runs]
    result = SelectionResult(status="unavailable", decision=decision, selections=entries)

    def unavailable(reason):
        return replace(result, selections=[replace(entry, failure_reason=entry.failure_reason or reason)
                                            for entry in entries], failure_reason=reason)

    if not expected_runs or len({r.run_id for r in expected_runs}) != len(expected_runs):
        return unavailable("Expected runs must be nonempty with unique run IDs")
    if (not math.isfinite(decision.target_nll) or not math.isfinite(decision.max_abs_error)
            or decision.max_abs_error < 0 or not math.isfinite(decision.minimum_gain)
            or decision.minimum_gain <= 0):
        return unavailable("Target, tolerance and positive minimum gain must be finite")
    original = expected_runs[0].original_model
    result = replace(result, original_model_id=original.model_id)
    if any(run.original_model != original for run in expected_runs):
        return unavailable("Expected runs do not share the same original model identity")
    outcomes = {outcome.run_id: outcome for outcome in run_results}
    unfinished = [r.run_id for r in expected_runs
                  if r.run_id not in outcomes or outcomes[r.run_id].status != "complete"]
    if unfinished:
        return unavailable("Missing or unsuccessful runs: " + ", ".join(unfinished))
    for run in expected_runs:
        saved = [cp for cp in checkpoints if cp.run_id == run.run_id]
        if (len({cp.step for cp in saved}) != len(saved)
                or {cp.step for cp in saved} != set(run.checkpoint_steps)
                or set(cp.model.model_id for cp in saved)
                != set(cp.model.model_id for cp in outcomes[run.run_id].checkpoints)
                or outcomes[run.run_id].last_step < max(run.checkpoint_steps, default=0)):
            return unavailable(f"Incomplete or conflicting saved checkpoint schedule for {run.run_id}")
    if any(r.variant != "unquantized" or any(not key.endswith("_validation") for key in r.nll)
           for r in validation_results):
        return unavailable("Selection accepts only unquantized validation measurements")
    try:
        successes = _successful(validation_results)
    except ValueError as error:
        return unavailable(str(error))
    expected_jobs = set(expected_validation_job_ids)
    if not expected_jobs or not expected_jobs.issubset({r.job_id for r in successes}):
        missing = expected_jobs - {r.job_id for r in successes}
        return unavailable("Missing successful expected validation jobs: " + ", ".join(sorted(missing)))
    successes = [r for r in successes if r.job_id in expected_jobs]
    if any(not r.nll or set(r.nll) != set(r.dataset_ids) or not r.settings_id
           or not r.inference_dtype or not r.compute_dtype or not r.software_versions
           or any(not dataset_id for dataset_id in r.dataset_ids.values()) for r in successes):
        return unavailable("Validation results lack complete data/settings/precision/software identities")
    baseline = [r for r in successes if r.model_id == original.model_id and "forget_validation" in r.nll]
    if len(baseline) != 1:
        return unavailable("Need exactly one original forget-validation measurement")
    baseline = baseline[0]
    if (baseline.tokenizer_id != original.tokenizer_id
            or baseline.tokenizer_revision != original.tokenizer_revision
            or baseline.quantization_id != "none"):
        return unavailable("Original validation tokenizer does not match original model")
    base = baseline.nll["forget_validation"].mean
    result = replace(result, original_validation_nll=base)
    if not math.isfinite(base) or decision.target_nll - base < decision.minimum_gain:
        return unavailable("Target must exceed original NLL by the chosen minimum gain")
    curves = {}
    for index, entry in enumerate(entries):
        points = []
        saved = [cp for cp in checkpoints if cp.run_id == entry.run_id]
        if not saved:
            entry = replace(entry, failure_reason="No saved checkpoints")
        for checkpoint in saved:
            measurements = [r for r in successes if r.model_id == checkpoint.model.model_id
                            and "forget_validation" in r.nll]
            if len(measurements) != 1:
                entry = replace(entry, failure_reason=f"Missing or ambiguous checkpoint validation at step {checkpoint.step}")
                break
            measurement = measurements[0]
            if _identity(measurement, "forget_validation") != _identity(baseline, "forget_validation"):
                entry = replace(entry, failure_reason=f"Incompatible checkpoint validation at step {checkpoint.step}")
                break
            value = measurement.nll["forget_validation"].mean
            if not math.isfinite(value):
                entry = replace(entry, failure_reason=f"Nonfinite checkpoint validation at step {checkpoint.step}")
                break
            points.append((checkpoint, value))
        curves[entry.run_id] = points
        entries[index] = entry
    if any(entry.failure_reason for entry in entries):
        return unavailable("Every saved checkpoint needs compatible successful validation")
    lower = max(min(value for _, value in curve) for curve in curves.values())
    upper = min(max(value for _, value in curve) for curve in curves.values())
    result = replace(result, common_range=[lower, upper])
    if lower > upper or not lower <= decision.target_nll <= upper:
        return unavailable("Target is outside the common observed checkpoint NLL range")
    for index, entry in enumerate(entries):
        checkpoint, value = min(curves[entry.run_id],
                                key=lambda point: (abs(point[1] - decision.target_nll), point[0].step))
        entry = replace(entry, checkpoint=checkpoint, achieved_nll=value,
                        gain_over_original=value-base, signed_error=value-decision.target_nll)
        if abs(entry.signed_error) > decision.max_abs_error:
            entry = replace(entry, failure_reason="Nearest saved checkpoint exceeds matching tolerance")
        elif entry.gain_over_original < decision.minimum_gain:
            entry = replace(entry, failure_reason="Selected checkpoint does not achieve minimum forgetting gain")
        entries[index] = entry
    if any(entry.failure_reason for entry in entries):
        return replace(result, selections=entries,
                       failure_reason="No matched comparison: one or more expected runs are unavailable")
    return replace(result, status="complete", selections=entries)
