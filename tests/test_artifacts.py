import dataclasses
import json

import pytest

from src.artifacts import (
    DatasetRef, EvaluationRequest, EvaluationResult, ModelRef, NLLStats,
    publish_record, read_record, resolve_artifact_path, stable_id,
)


def model():
    return ModelRef("original", "models/original", "a" * 40, "tokenizer", "b" * 40, "bfloat16")


def dataset():
    return DatasetRef("data-id", "data/samples.json", "forget_validation", "data/manifest.json",
                      "tokenizer", "b" * 40, "preprocessing", 8)


def test_nested_record_round_trip_and_immutable_publication(tmp_path):
    request = EvaluationRequest("job", "attempt-1", model(), {"forget_validation": dataset()},
                                "unquantized", {"batch_size": 2}, {"backend": "bitsandbytes"}, "settings")
    path = publish_record(request, tmp_path)
    assert path.relative_to(tmp_path).as_posix() == "records/evaluation_request/job--attempt-1.json"
    assert read_record(path, EvaluationRequest) == request
    assert isinstance(read_record(path).model, ModelRef)
    assert publish_record(request, tmp_path) == path
    with pytest.raises(FileExistsError, match="Conflicting immutable"):
        publish_record(dataclasses.replace(request, settings_id="different"), tmp_path)
    assert read_record(path) == request
    assert not list(tmp_path.rglob("*.tmp"))


def test_retries_have_separate_immutable_attempt_records(tmp_path):
    failed = EvaluationResult("job", "attempt-1", "original", "unquantized", "settings",
                              {"forget_validation": "data-id"}, "failed", {}, "bfloat16", "bfloat16", {},
                              failure_reason="worker interrupted")
    succeeded = dataclasses.replace(failed, attempt_id="attempt-2", status="complete",
                                    nll={"forget_validation": NLLStats(9.0, 3)}, failure_reason=None)
    first = publish_record(failed, tmp_path)
    second = publish_record(succeeded, tmp_path)
    assert first != second
    assert read_record(first).failure_reason == "worker interrupted"
    assert read_record(second).nll["forget_validation"].mean == 3.0


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"kind": "unknown"},
    {"nll_sum": "9"}, {"token_count": True}, {"token_count": 0}, {"unexpected": 1},
])
def test_read_rejects_schema_and_field_type_changes(tmp_path, change):
    payload = dataclasses.asdict(NLLStats(9.0, 3))
    payload.update(change)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_record(path)


def test_read_rejects_wrong_expected_type_and_nonfinite_json(tmp_path):
    path = publish_record(NLLStats(9.0, 3), tmp_path)
    with pytest.raises(ValueError, match="Expected ModelRef"):
        read_record(path, ModelRef)
    path = tmp_path / "nan.json"
    path.write_text('{"kind":"nll_stats","schema_version":1,"nll_sum":NaN,"token_count":3}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_record(path)


@pytest.mark.parametrize("location", ["../outside", "/absolute", "C:/absolute", "data\\windows", "data/../../outside"])
def test_artifact_paths_cannot_escape_the_local_root(tmp_path, location):
    with pytest.raises(ValueError):
        resolve_artifact_path(tmp_path, location)


def test_content_identity_is_order_independent_and_rejects_non_json():
    assert stable_id({"a": 1, "b": [2, 3]}) == stable_id({"b": [2, 3], "a": 1})
    assert stable_id({"a": 1}) != stable_id({"a": 2})
    with pytest.raises(ValueError):
        stable_id({"bad": float("inf")})
    with pytest.raises(TypeError):
        stable_id(object())


def test_publish_validates_nested_record_types(tmp_path):
    request = EvaluationRequest("job", "attempt", model(), {"forget_validation": "not-a-DatasetRef"},
                                "unquantized", {}, {}, "settings")
    with pytest.raises(ValueError):
        publish_record(request, tmp_path)
    assert not list(tmp_path.rglob("*.json"))
