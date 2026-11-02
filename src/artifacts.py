"""Small JSON contracts shared by training, evaluation, and analysis."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class Record:
    schema_version: int = field(default=1, init=False)
    kind: str = field(default="record", init=False)


@dataclass(frozen=True)
class ExperimentConfig(Record):
    project_root: str
    experiment: dict[str, Any]
    execution: dict[str, Any]
    model: dict[str, Any]
    data: dict[str, Any]
    sweep: dict[str, Any]
    training: dict[str, Any]
    checkpointing: dict[str, Any]
    target_selection: dict[str, Any]
    evaluation: dict[str, Any]
    quantization: dict[str, Any]
    predictor: dict[str, Any]
    kind: str = field(default="experiment_config", init=False)


@dataclass(frozen=True)
class ModelRef(Record):
    model_id: str
    location: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    storage_dtype: str
    source: str = "artifact"
    kind: str = field(default="model_ref", init=False)


@dataclass(frozen=True)
class DatasetRef(Record):
    dataset_id: str
    location: str
    split: str
    manifest: str
    tokenizer_id: str
    tokenizer_revision: str
    preprocessing_id: str
    max_sequence_length: int
    kind: str = field(default="dataset_ref", init=False)


@dataclass(frozen=True)
class RunSpec(Record):
    run_id: str
    learning_rate: float
    seed: int
    original_model: ModelRef
    train_datasets: dict[str, DatasetRef]
    training: dict[str, Any]
    checkpoint_steps: list[int]
    output_dir: str
    kind: str = field(default="run_spec", init=False)


@dataclass(frozen=True)
class CheckpointRef(Record):
    model: ModelRef
    run_id: str
    step: int
    epoch: float
    kind: str = field(default="checkpoint_ref", init=False)


@dataclass(frozen=True)
class RunResult(Record):
    run_id: str
    status: str
    checkpoints: list[CheckpointRef]
    last_step: int
    failure_reason: str | None = None
    kind: str = field(default="run_result", init=False)


@dataclass(frozen=True)
class EvaluationRequest(Record):
    job_id: str
    attempt_id: str
    model: ModelRef
    datasets: dict[str, DatasetRef]
    variant: str
    evaluation: dict[str, Any]
    quantization: dict[str, Any]
    settings_id: str
    kind: str = field(default="evaluation_request", init=False)


@dataclass(frozen=True)
class NLLStats(Record):
    nll_sum: float
    token_count: int
    kind: str = field(default="nll_stats", init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.nll_sum) or self.nll_sum < 0:
            raise ValueError("NLL sum must be finite and nonnegative")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("NLL requires a positive target-token count")

    @property
    def mean(self) -> float:
        return self.nll_sum / self.token_count


@dataclass(frozen=True)
class EvaluationResult(Record):
    job_id: str
    attempt_id: str
    model_id: str
    variant: str
    settings_id: str
    dataset_ids: dict[str, str]
    status: str
    nll: dict[str, NLLStats]
    inference_dtype: str
    compute_dtype: str
    software_versions: dict[str, str]
    failure_reason: str | None = None
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    quantization_id: str = "none"
    kind: str = field(default="evaluation_result", init=False)


@dataclass(frozen=True)
class PredictorResult(Record):
    learning_rate: float
    value: float
    model_id: str
    tensor_scope: str
    storage_dtype: str
    working_dtype: str
    calculation_id: str
    kind: str = field(default="predictor_result", init=False)


@dataclass(frozen=True)
class SweepIndex(Record):
    snapshot_id: str
    config: ExperimentConfig
    datasets: dict[str, DatasetRef]
    runs: list[RunSpec]
    expected_job_ids: list[str]
    run_result_paths: list[str]
    evaluation_result_paths: list[str]
    predictor_paths: list[str]
    status: str
    software_versions: dict[str, str]
    failure_reason: str | None = None
    kind: str = field(default="sweep_index", init=False)


@dataclass(frozen=True)
class SelectionDecision(Record):
    target_nll: float
    max_abs_error: float
    reason: str
    minimum_gain: float = 1e-6
    kind: str = field(default="selection_decision", init=False)


@dataclass(frozen=True)
class SelectionEntry(Record):
    run_id: str
    learning_rate: float
    seed: int
    checkpoint: CheckpointRef | None = None
    achieved_nll: float | None = None
    gain_over_original: float | None = None
    signed_error: float | None = None
    failure_reason: str | None = None
    kind: str = field(default="selection_entry", init=False)


@dataclass(frozen=True)
class SelectionResult(Record):
    status: str
    decision: SelectionDecision
    selections: list[SelectionEntry]
    original_validation_nll: float | None = None
    common_range: list[float] | None = None
    failure_reason: str | None = None
    original_model_id: str | None = None
    kind: str = field(default="selection_result", init=False)


@dataclass(frozen=True)
class RetentionResult(Record):
    run_id: str
    learning_rate: float
    seed: int
    dataset_id: str
    settings_id: str
    nll_t: float | None
    nll_qt: float | None
    nll_u: float | None
    nll_qu: float | None
    unquantized_gap: float | None
    quantized_gap: float | None
    retention_ratio: float | None
    retain_unquantized_delta: float | None = None
    retain_quantized_delta: float | None = None
    status: str = "complete"
    failure_reason: str | None = None
    kind: str = field(default="retention_result", init=False)


@dataclass(frozen=True)
class AnalysisResult(Record):
    status: str
    selection_path: str | None = None
    measurement_paths: list[str] = field(default_factory=list)
    table_paths: list[str] = field(default_factory=list)
    figure_paths: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    kind: str = field(default="analysis_result", init=False)


_RECORD_TYPES = {cls.__dataclass_fields__["kind"].default: cls for cls in (
    ExperimentConfig, ModelRef, DatasetRef, RunSpec, CheckpointRef, RunResult,
    EvaluationRequest, NLLStats, EvaluationResult, PredictorResult, SweepIndex,
    SelectionDecision, SelectionEntry, SelectionResult, RetentionResult, AnalysisResult,
)}


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("JSON object keys must be strings")
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Not a JSON artifact value: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(value: Any) -> str:
    """Canonical content identity independent of dictionary ordering."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def resolve_artifact_path(root: Path, location: str) -> Path:
    """Resolve only portable relative references beneath this machine's root."""
    path = Path(location)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", location) or "\\" in location:
        raise ValueError(f"Expected artifact-relative POSIX path: {location}")
    resolved = (Path(root).resolve() / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Artifact path escapes root: {location}")
    return resolved


def _decode(value: Any, expected: Any) -> Any:
    if expected is Any:
        _canonical(value)
        return value
    origin, args = get_origin(expected), get_args(expected)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _decode(value, option)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"Value does not match {expected}")
    if expected is type(None):
        if value is not None:
            raise ValueError("Expected null")
        return None
    if origin is list:
        if not isinstance(value, list):
            raise ValueError("Expected list")
        return [_decode(v, args[0]) for v in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError("Expected object")
        return {_decode(k, args[0]): _decode(v, args[1]) for k, v in value.items()}
    if isinstance(expected, type) and issubclass(expected, Record):
        if not isinstance(value, dict):
            raise ValueError(f"Expected {expected.__name__} object")
        fields = {f.name: f for f in dataclasses.fields(expected)}
        if set(value) != set(fields):
            raise ValueError(f"Invalid fields for {expected.__name__}: {set(value) ^ set(fields)}")
        if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
            raise ValueError("Unsupported artifact schema version")
        if value["kind"] != fields["kind"].default:
            raise ValueError(f"Wrong artifact kind for {expected.__name__}")
        hints = get_type_hints(expected)
        return expected(**{name: _decode(value[name], hints[name]) for name, f in fields.items() if f.init})
    if expected is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if expected in (str, int, bool) and type(value) is expected:
        return value
    raise ValueError(f"Expected {expected}, got {type(value).__name__}")


def read_record(path: Path, expected_type: type[Record] | None = None) -> Record:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict) or value.get("kind") not in _RECORD_TYPES:
        raise ValueError("Unknown artifact record kind")
    actual_type = _RECORD_TYPES[value["kind"]]
    if expected_type is not None and actual_type is not expected_type:
        raise ValueError(f"Expected {expected_type.__name__}, got {actual_type.__name__}")
    return _decode(value, actual_type)


def publish_record(record: Record, artifact_root: Path) -> Path:
    """Atomically publish once; byte-identical retries are harmless."""
    if type(record) not in _RECORD_TYPES.values():
        raise TypeError("publish_record requires a known record dataclass")
    payload = _canonical(record)
    _decode(json.loads(payload), type(record))
    if isinstance(record, (EvaluationRequest, EvaluationResult)):
        identity = f"{record.job_id}--{record.attempt_id}"
    elif isinstance(record, SweepIndex):
        identity = record.snapshot_id
    elif isinstance(record, (RunSpec, RunResult)):
        identity = record.run_id
    elif isinstance(record, ModelRef):
        identity = record.model_id
    elif isinstance(record, DatasetRef):
        identity = record.dataset_id
    else:
        identity = stable_id(record)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity) or identity in (".", ".."):
        raise ValueError(f"Invalid artifact record identity: {identity}")
    path = resolve_artifact_path(artifact_root, f"records/{record.kind}/{identity}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"Conflicting immutable artifact: {path}") from None
        return path
    finally:
        temporary.unlink(missing_ok=True)
