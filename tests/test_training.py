"""Full-weight precision and objective tests for cloud execution only."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.training import unlearn
from src.eval.evaluator import _collate
from src.artifacts import (CheckpointRef, DatasetRef, ModelRef, RunSpec, read_record,
                           stable_id)


def settings():
    return {"method": "ga_gdr", "optimizer": "adamw_torch", "parameter_dtype": "bfloat16",
            "optimizer_state_dtype": "bfloat16", "fp32_master_weights": False,
            "stochastic_rounding": False, "lr_scheduler": "constant", "warmup_steps": 0,
            "per_device_batch_size": 2, "gradient_accumulation_steps": 2, "max_epochs": 2,
            "retain_loss_weight": 1.0, "max_grad_norm": 1.0, "weight_decay": 0.0,
            "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8}


def test_schedule_uses_absolute_epoch_boundaries_and_remainder():
    # Five microbatches, accumulation=2: three optimizer updates each epoch.
    assert unlearn.checkpoint_schedule(10, 2, 2, 2, 0.5) == [2, 3, 5, 6]
    assert unlearn.checkpoint_schedule(10, 2, 2, 2, 0.5, [1, 2, 50]) == [1, 2, 3, 5, 6]
    assert unlearn.checkpoint_schedule(1, 2, 5, 1, 0.1) == [1]


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_schedule_rejects_invalid_interval(interval):
    with pytest.raises(ValueError):
        unlearn.checkpoint_schedule(10, 2, 2, 2, interval)


def test_unsupported_settings_fail_before_loading():
    bad = settings() | {"fp32_master_weights": True}
    with pytest.raises(ValueError, match="Unsupported"):
        unlearn._validate_settings(bad)


def test_adamw_bf16_moments_no_master_and_direct_rounding():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = unlearn._make_optimizer(model, settings(), 1e-6)
    original_pointer = model.weight.data_ptr()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    unlearn._verify_optimizer(model, optimizer)
    assert model.weight.data_ptr() == original_pointer
    assert torch.equal(model.weight, torch.ones_like(model.weight))  # Small update rounds away.
    state = optimizer.state[model.weight]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.bfloat16
    assert state["step"].numel() == 1
    assert optimizer.param_groups[0]["foreach"] is False
    assert optimizer.param_groups[0]["fused"] is False


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.2, 0.1, -0.1],
            [0.3, -0.2, 0.1], [-0.3, 0.1, 0.2]], dtype=torch.float32))

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(logits=self.weight[input_ids])


def batch(ids):
    return _collate([{"input_ids": ids, "labels": ids, "attention_mask": [1] * len(ids)}])


def test_ga_gdr_gradient_is_separate_branch_token_means_with_partial_accumulation():
    model = TinyLM()
    pairs = [(batch([0, 1, 2, 1]), batch([2, 0])),
             (batch([1, 0]), batch([0, 2, 0]))]
    unlearn._backward_window(model, pairs, 0.7, torch.device("cpu"))
    actual = model.weight.grad.clone()
    reference = TinyLM()
    forget = F.cross_entropy(reference.weight[[0, 1, 2, 1]],
                              torch.tensor([1, 2, 1, 0]), reduction="mean")
    retain = F.cross_entropy(reference.weight[[2, 0, 2]],
                              torch.tensor([0, 2, 0]), reduction="mean")
    (-forget + 0.7 * retain).backward()
    assert torch.allclose(actual, reference.weight.grad, atol=1e-7)
    # A single remainder microbatch divides by its own tokens, never configured accumulation.
    remainder = TinyLM()
    losses = unlearn._backward_window(remainder, pairs[-1:], 0.7, torch.device("cpu"))
    assert losses[0] == pytest.approx(F.cross_entropy(remainder.weight[[1]], torch.tensor([0])).item())


def spec():
    original = ModelRef("original", "models/original", "revision", "tokenizer", "tokrev", "bfloat16")
    dataset = DatasetRef("data", "data/tokens", "train", "data/manifest", "tokenizer",
                         "tokrev", "preprocess", 8)
    return RunSpec("run", 1e-3, 0, original,
                   {"forget_train": dataset, "retain_train": dataset},
                   settings() | {"gradient_accumulation_steps": 1, "max_epochs": 1},
                   [1, 2], "runs/run")


def test_checkpoint_bundle_complete_before_reference_is_returned(tmp_path):
    class FakeModel:
        def save_pretrained(self, path, **kwargs):
            assert not (tmp_path / "runs/run/checkpoint-1").exists()
            (path / "model.safetensors").write_bytes(b"test weights")
            (path / "config.json").write_text("{}", encoding="utf-8")
    class FakeTokenizer:
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    ref = unlearn._save_checkpoint(FakeModel(), FakeTokenizer(), spec(), 1, 0.5, tmp_path)
    directory = tmp_path / ref.model.location
    assert (directory / "checkpoint_complete.json").exists()
    assert (directory / "model.safetensors").exists()
    assert (directory / "tokenizer_config.json").exists()
    assert ref.model.source == "artifact" and ref.model.storage_dtype == "bfloat16"
    assert not list(directory.parent.glob("*.tmp"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RunPod CUDA training lifecycle test")
def test_train_preserves_refs_after_callback_error_and_saves_final_once(monkeypatch, tmp_path):
    class TrainLM(TinyLM):
        def gradient_checkpointing_enable(self, **kwargs):
            pass
    data = [{"input_ids": [0, 1, 2], "labels": [0, 1, 2], "attention_mask": [1, 1, 1]}] * 3
    model = TrainLM().bfloat16().cuda()
    monkeypatch.setattr(unlearn, "load_model", lambda *a, **kw: model)
    monkeypatch.setattr(unlearn, "load_data", lambda *a: data)
    monkeypatch.setattr(unlearn, "_load_tokenizer", lambda *a: object())
    saved, callbacks = [], []
    def save(model, tokenizer, run, step, epoch, root):
        saved.append(step)
        original = run.original_model
        ref = ModelRef(f"checkpoint-{step}", f"runs/run/checkpoint-{step}", "revision",
                       original.tokenizer_id, original.tokenizer_revision, "bfloat16")
        return CheckpointRef(ref, run.run_id, step, epoch)
    monkeypatch.setattr(unlearn, "_save_checkpoint", save)
    def callback(ref):
        # Reference publication precedes callbacks, even when publication of jobs fails.
        persisted = list((tmp_path / "records/checkpoint_ref").glob("*.json"))
        assert any(read_record(path) == ref for path in persisted)
        callbacks.append(ref.step)
        if ref.step == 1:
            raise RuntimeError("synthetic queue publication failure")
    result = unlearn.train_run(spec(), tmp_path, callback)
    assert result.status == "complete"
    assert result.last_step == 2
    assert [ref.step for ref in result.checkpoints] == saved == callbacks == [1, 2]
    assert "Traceback" in result.failure_reason and "queue publication" in result.failure_reason
