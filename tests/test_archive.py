import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.artifacts import (CheckpointRef, ModelRef, RunResult, RunSpec, SelectionDecision,
                           SelectionEntry, SelectionResult, publish_record)


_spec = importlib.util.spec_from_file_location("pilot_archive_tool", Path(__file__).parents[1] / "runpod" / "archive.py")
archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archive)


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "pilot"
    root.mkdir()
    original = ModelRef("original", "public/model", "a" * 40, "tokenizer", "b" * 40, "bfloat16", "hub")
    publish_record(RunSpec("run", 1e-5, 0, original, {}, {}, [1, 2, 3], "checkpoints/run"), root)
    refs = []
    for step in (1, 2, 3):
        location = f"checkpoints/run/checkpoint-{step}"
        directory = root / location
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "tokenizer_config.json").write_text("{}")
        (directory / "checkpoint_complete.json").write_text(json.dumps({"run_id": "run", "step": step}))
        (directory / "model.safetensors").write_bytes(f"weights-{step}".encode())
        model = ModelRef(f"model-{step}", location, f"revision-{step}", "tokenizer", "b" * 40, "bfloat16")
        reference = CheckpointRef(model, "run", step, float(step))
        publish_record(reference, root)
        refs.append(reference)
    publish_record(RunResult("run", "complete", refs, 3), root)
    publish_record(SelectionResult("complete", SelectionDecision(2.0, 0.1, "validation only"),
                                   [SelectionEntry("run", 1e-5, 0, refs[1], 2.0, 1.0, 0.0)],
                                   original_validation_nll=1.0, original_model_id="original"), root)
    (root / "analysis").mkdir()
    (root / "analysis" / "summary.csv").write_text("lr,retention\n1e-5,0.8\n")
    (root / "data").mkdir()
    (root / "data" / "samples.json").write_text('{"samples":[[1,2,3]]}')
    (root / "resolved-config.json").write_text('{"immutable":"settings"}')
    (root / ".env").write_text("must-never-be-copied")
    (root / "credentials.json").write_text("must-never-be-copied")
    (root / "temporary.tmp").write_text("unfinished")
    return root, refs


def test_plan_keeps_selected_and_final_once_and_all_evidence(artifacts):
    root, refs = artifacts
    plan = archive.plan_archive(root)
    locations = {item["location"]: item["roles"] for item in plan["checkpoint_directories"]}
    assert locations == {refs[1].model.location: ["selected"], refs[2].model.location: ["final"]}
    paths = [item["path"] for item in plan["files"]]
    assert f"{refs[0].model.location}/model.safetensors" not in paths
    assert f"{refs[1].model.location}/model.safetensors" in paths
    assert f"{refs[2].model.location}/model.safetensors" in paths
    assert {"analysis/summary.csv", "data/samples.json", "resolved-config.json"} <= set(paths)
    assert not {".env", "credentials.json", "temporary.tmp"} & set(paths)
    assert plan["total_bytes"] == sum((root / path).stat().st_size for path in paths)
    assert plan["checkpoint_bytes"] + plan["evidence_bytes"] == plan["total_bytes"]
    assert plan["estimated_minimum_monthly_usd"] == pytest.approx(0.7)


def test_checkpoint_selected_and_final_is_not_duplicated(artifacts):
    root, refs = artifacts
    for path in (root / "records" / "selection_result").glob("*.json"):
        path.unlink()
    publish_record(SelectionResult("complete", SelectionDecision(2.0, 0.1, "validation only"),
                                   [SelectionEntry("run", 1e-5, 0, refs[2], 2.0, 1.0, 0.0)]), root)
    plan = archive.plan_archive(root)
    assert len(plan["checkpoint_directories"]) == 1
    assert plan["checkpoint_directories"][0]["roles"] == ["selected", "final"]
    assert len(plan["files"]) == len({item["path"] for item in plan["files"]})


def test_unavailable_selection_candidates_remain_evidence_without_archiving_weights(artifacts):
    root, refs = artifacts
    for path in (root / "records" / "selection_result").glob("*.json"):
        path.unlink()
    unavailable = SelectionResult("unavailable", SelectionDecision(2.0, 0.1, "validation only"),
                                   [SelectionEntry("run", 1e-5, 0, refs[1], 2.5, 1.5, 0.5,
                                                   failure_reason="Matching error exceeds tolerance")],
                                   failure_reason="No matched comparison")
    record_path = publish_record(unavailable, root)
    plan = archive.plan_archive(root)
    assert [(entry["location"], entry["roles"]) for entry in plan["checkpoint_directories"]] == [
        (refs[2].model.location, ["final"])]
    paths = {entry["path"] for entry in plan["files"]}
    assert f"{refs[1].model.location}/model.safetensors" not in paths
    assert f"{refs[2].model.location}/model.safetensors" in paths
    assert record_path.relative_to(root).as_posix() in paths
    assert plan["selection_records"][0]["status"] == "unavailable"


def test_plan_rejects_missing_final_shards_without_deleting_anything(artifacts):
    root, refs = artifacts
    final = root / refs[2].model.location
    (final / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tensor": "missing-shard.safetensors"}}))
    with pytest.raises(archive.ArchiveError, match="shard.*missing"):
        archive.plan_archive(root)
    assert (final / "model.safetensors").exists()


def test_missing_expected_outcome_produces_explicit_partial_receipt(artifacts, tmp_path):
    root, _ = artifacts
    original = ModelRef("original", "public/model", "a" * 40, "tokenizer", "b" * 40, "bfloat16", "hub")
    publish_record(RunSpec("missing-run", 1e-4, 0, original, {}, {}, [1], "checkpoints/missing-run"), root)
    plan = archive.plan_archive(root)
    assert plan["experiment_status"] == "incomplete"
    assert plan["missing_run_ids"] == ["missing-run"]
    assert any(entry["roles"] == ["final"] for entry in plan["checkpoint_directories"])
    result = archive.archive_to_volume(root, tmp_path / "partial-volume")
    receipt = json.loads(Path(result["destination_manifest_path"]).read_text())
    assert receipt["status"] == "verified"
    assert receipt["experiment_status"] == "incomplete" and receipt["missing_run_ids"] == ["missing-run"]


def test_published_checkpoint_without_outcome_is_kept_without_claiming_final(artifacts):
    root, refs = artifacts
    for path in (root / "records" / "run_result").glob("*.json"):
        path.unlink()
    plan = archive.plan_archive(root)
    assert plan["experiment_status"] == "incomplete" and plan["missing_run_ids"] == ["run"]
    recovered = next(entry for entry in plan["checkpoint_directories"] if entry["location"] == refs[2].model.location)
    assert recovered["roles"] == ["last_saved_without_outcome"]
    assert not any("final" in entry["roles"] for entry in plan["checkpoint_directories"])


def test_evidence_only_archive_is_verified_but_experiment_incomplete(tmp_path):
    root = tmp_path / "evidence-only"
    root.mkdir()
    original = ModelRef("original", "public/model", "a" * 40, "tokenizer", "b" * 40, "bfloat16", "hub")
    publish_record(RunSpec("not-started", 1e-5, 0, original, {}, {}, [1], "checkpoints/not-started"), root)
    (root / "failure.json").write_text('{"reason":"budget exhausted before training"}')
    result = archive.archive_to_volume(root, tmp_path / "evidence-volume")
    assert result["status"] == "verified" and result["experiment_status"] == "incomplete"
    assert result["archive_scope"] == "evidence_only" and result["checkpoint_directories"] == []
    assert result["missing_run_ids"] == ["not-started"]
    assert (tmp_path / "evidence-volume" / "failure.json").exists()


def test_copy_rehashes_destination_and_publishes_manifest_last(artifacts, tmp_path):
    root, refs = artifacts
    destination = tmp_path / "network-volume" / "archive"
    original = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = archive.archive_to_volume(root, destination)
    assert result["status"] == "verified" and result["verification"] == "sha256_source_and_destination"
    manifest_path = Path(result["destination_manifest_path"])
    assert json.loads(manifest_path.read_text())["archive_id"] == result["archive_id"]
    assert Path(result["local_manifest_path"]).read_bytes() == manifest_path.read_bytes()
    for item in result["files"]:
        copied = destination / item["path"]
        assert copied.stat().st_size == item["bytes"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == item["sha256"]
    assert all((root / path).read_bytes() == value for path, value in original.items())
    assert not (destination / refs[0].model.location).exists()
    assert (root / refs[0].model.location / "model.safetensors").exists()
    assert archive.archive_to_volume(root, destination)["archive_id"] == result["archive_id"]


def test_stage_finals_skips_active_and_selected_checkpoints_without_publishing_manifest(artifacts, tmp_path):
    root, refs = artifacts
    active_location = "checkpoints/active/checkpoint-1"
    active_directory = root / active_location
    active_directory.mkdir(parents=True)
    (active_directory / "config.json").write_text("{}")
    (active_directory / "tokenizer_config.json").write_text("{}")
    (active_directory / "checkpoint_complete.json").write_text('{"run_id":"active","step":1}')
    (active_directory / "model.safetensors").write_bytes(b"active-weights")
    active_model = ModelRef("active-model", active_location, "revision", "tokenizer", "b" * 40, "bfloat16")
    publish_record(CheckpointRef(active_model, "active", 1, 1.0), root)
    destination = tmp_path / "staged-volume"
    first = archive.stage_final_checkpoints(root, destination)
    assert first["status"] == "staged"
    assert [(item["run_id"], item["step"]) for item in first["checkpoints"]] == [("run", 3)]
    assert (destination / refs[2].model.location / "model.safetensors").read_bytes() == b"weights-3"
    assert not (destination / refs[1].model.location).exists()
    assert not (destination / active_location).exists()
    assert not (destination / "archive-manifest.json").exists()
    assert not (root / "archive").exists()
    assert archive.stage_final_checkpoints(root, destination) == first


def test_staging_selected_requires_flag_and_complete_frozen_selection(artifacts, tmp_path):
    root, refs = artifacts
    destination = tmp_path / "selected-stage-volume"
    archive.stage_final_checkpoints(root, destination)
    assert not (destination / refs[1].model.location).exists()
    for path in (root / "records" / "selection_result").glob("*.json"):
        path.unlink()
    decision = SelectionDecision(2.0, 0.1, "validation only")
    entries = [SelectionEntry("run", 1e-5, 0, refs[1], 2.0, 1.0, 0.0),
               SelectionEntry("run", 1e-5, 0, refs[2], 2.0, 1.0, 0.0)]
    publish_record(SelectionResult("unavailable", decision, entries), root)
    archive.stage_final_checkpoints(root, destination, include_selected=True)
    assert not (destination / refs[1].model.location).exists()
    publish_record(SelectionResult("complete", decision, entries), root)
    result = archive.stage_final_checkpoints(root, destination, include_selected=True)
    assert (destination / refs[1].model.location / "model.safetensors").read_bytes() == b"weights-2"
    checkpoints = {entry["location"]: entry for entry in result["checkpoints"]}
    assert len(checkpoints) == 2
    assert checkpoints[refs[1].model.location]["roles"] == ["selected"]
    assert checkpoints[refs[2].model.location]["roles"] == ["final", "selected"]
    assert not (destination / "archive-manifest.json").exists()
    assert not (root / "archive").exists()


def test_restaging_defers_same_size_corruption_but_final_archive_rejects_it(artifacts, tmp_path, monkeypatch, capsys):
    root, refs = artifacts
    destination = tmp_path / "deferred-verification-volume"
    archive.stage_final_checkpoints(root, destination)
    target = destination / refs[2].model.location / "model.safetensors"
    target.write_bytes(b"corrupt-3")
    assert target.stat().st_size == (root / refs[2].model.location / "model.safetensors").stat().st_size
    def forbidden_copy(*args):
        pytest.fail("Restaging must defer byte verification for existing same-size files")
    with monkeypatch.context() as staging_patch:
        staging_patch.setattr(archive, "_copy_verified", forbidden_copy)
        assert archive.stage_final_checkpoints(root, destination)["status"] == "staged"
    assert "staged_existing_file_verification_deferred" in capsys.readouterr().out
    with pytest.raises(archive.ArchiveError, match="SHA256 verification failed"):
        archive.archive_to_volume(root, destination)
    assert not (destination / "archive-manifest.json").exists()
    assert not (root / "archive").exists()


def test_conflicting_destination_never_marks_archive_verified(artifacts, tmp_path):
    root, refs = artifacts
    destination = tmp_path / "volume"
    conflicting = destination / refs[1].model.location / "model.safetensors"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"wrong-weights")
    with pytest.raises(archive.ArchiveError, match="verification failed"):
        archive.archive_to_volume(root, destination)
    assert not (destination / "archive-manifest.json").exists()
    assert conflicting.read_bytes() == b"wrong-weights"
    assert all((root / reference.model.location / "model.safetensors").exists() for reference in refs)


def test_insufficient_capacity_is_detected_before_copy(artifacts, tmp_path, monkeypatch):
    root, _ = artifacts
    destination = tmp_path / "volume"
    monkeypatch.setattr(archive.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(archive.ArchiveError, match="Insufficient free"):
        archive.archive_to_volume(root, destination)
    assert not list(destination.rglob("*"))


def test_copy_failure_does_not_publish_manifest_or_delete_source(artifacts, tmp_path, monkeypatch):
    root, refs = artifacts
    destination = tmp_path / "volume"
    def fail(*args):
        raise OSError("storage failure")
    monkeypatch.setattr(archive, "_copy_verified", fail)
    with pytest.raises(archive.ArchiveError, match="original files retained"):
        archive.archive_to_volume(root, destination)
    assert not (destination / "archive-manifest.json").exists()
    assert not list((root / "archive").glob("*.json"))
    assert all((root / reference.model.location / "model.safetensors").exists() for reference in refs)


def test_source_and_destination_cannot_overlap(artifacts):
    root, _ = artifacts
    with pytest.raises(archive.ArchiveError, match="separate directory trees"):
        archive.archive_to_volume(root, root / "recursive-copy")


def test_symbolic_links_are_rejected(artifacts, tmp_path):
    root, _ = artifacts
    outside = tmp_path / "outside.txt"
    outside.write_text("external content")
    link = root / "linked-evidence.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links unavailable on this test platform")
    with pytest.raises(archive.ArchiveError):
        archive.plan_archive(root)
