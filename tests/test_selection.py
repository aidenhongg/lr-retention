from dataclasses import replace

import pytest

from src.artifacts import (CheckpointRef, EvaluationResult, ModelRef, NLLStats, RunResult,
                           RunSpec, SelectionDecision)
from src.analysis.selection import select_checkpoints


def selection_case():
    original = ModelRef("T", "models/T", "revision", "tokenizer", "tok-rev", "bfloat16")
    runs = [RunSpec(f"r{seed}", 1e-5, seed, original, {}, {}, [1, 2], f"runs/r{seed}")
            for seed in (0, 1)]
    checkpoints = [CheckpointRef(replace(original, model_id=f"{run.run_id}-{step}"), run.run_id, step, float(step))
                   for run in runs for step in (1, 2)]

    def result(model_id, value):
        return EvaluationResult(model_id, "attempt", model_id, "unquantized", "settings",
                                {"forget_validation": "forget-val"}, "complete",
                                {"forget_validation": NLLStats(value * 10, 10)},
                                "bfloat16", "bfloat16", {"torch": "fixed"},
                                tokenizer_id="tokenizer", tokenizer_revision="tok-rev")

    measurements = [result("T", 1)] + [result(cp.model.model_id, 2 if cp.step == 1 else 4)
                                      for cp in checkpoints]
    outcomes = [RunResult(run.run_id, "complete", [cp for cp in checkpoints if cp.run_id == run.run_id], 2)
                for run in runs]
    return runs, checkpoints, measurements, outcomes


def select(case, decision=None, expected_jobs=None):
    runs, checkpoints, measurements, outcomes = case
    return select_checkpoints(runs, checkpoints, measurements, decision or SelectionDecision(3, 1, "test"),
                              run_results=outcomes, expected_validation_job_ids=expected_jobs
                              if expected_jobs is not None else [r.job_id for r in measurements])


def test_tie_uses_earliest_saved_step_and_signed_error():
    result = select(selection_case())
    assert result.status == "complete"
    assert result.common_range == [2, 4]
    assert len(result.selections) == 2
    assert all(entry.checkpoint.step == 1 and entry.signed_error == -1
               and entry.gain_over_original == 1 for entry in result.selections)


def test_discrete_checkpoints_can_fail_despite_common_range():
    result = select(selection_case(), SelectionDecision(3, 0.5, "strict"))
    assert result.status == "unavailable"
    assert len(result.selections) == 2
    assert all(entry.failure_reason for entry in result.selections)


@pytest.mark.parametrize("target", [float("nan"), float("inf"), 1.0, 0.5, 5.0])
def test_target_must_be_finite_above_original_and_in_common_support(target):
    assert select(selection_case(), SelectionDecision(target, 10, "bad target")).status == "unavailable"


def test_no_expected_seed_can_be_omitted_after_failed_run():
    case = list(selection_case())
    case[3][1] = replace(case[3][1], status="failed", failure_reason="OOM")
    result = select(case)
    assert result.status == "unavailable"
    assert {entry.seed for entry in result.selections} == {0, 1}
    assert all(entry.failure_reason for entry in result.selections)


def test_successful_runs_do_not_replace_expected_job_completion():
    case = selection_case()
    jobs = [r.job_id for r in case[2]] + ["not-finished"]
    assert select(case, expected_jobs=jobs).status == "unavailable"


def test_completed_job_list_cannot_hide_missing_saved_checkpoint():
    case = list(selection_case())
    case[1] = case[1][:-1]
    assert select(case).status == "unavailable"


@pytest.mark.parametrize("change", [
    {"dataset_ids": {"forget_validation": "other-data"}},
    {"tokenizer_revision": "other-tokenizer"},
    {"compute_dtype": "float32"},
    {"software_versions": {"torch": "other"}},
    {"nll": {"forget_validation": NLLStats(40, 20)}},
])
def test_validation_identity_must_match_original(change):
    case = list(selection_case())
    case[2][1] = replace(case[2][1], **change)
    assert select(case).status == "unavailable"


def test_quantized_or_blind_measurements_are_rejected_for_selection():
    case = list(selection_case())
    case[2].append(replace(case[2][0], job_id="blind", nll={"forget_test": NLLStats(10, 10)}))
    assert select(case).status == "unavailable"


def test_successful_retry_counted_once_but_conflicting_success_rejected():
    case = list(selection_case())
    case[2].append(replace(case[2][0], attempt_id="retry"))
    assert select(case).status == "complete"
    case[2][-1] = replace(case[2][-1], nll={"forget_validation": NLLStats(15, 10)})
    assert select(case).status == "unavailable"
