"""Synthetic numerical tests; run on RunPod without fetching model weights."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.artifacts import (DatasetRef, EvaluationRequest, EvaluationResult, ModelRef,
                           NLLStats, publish_record, read_record, stable_id)
from src.eval import evaluator


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor([
            [2.0, 0.0, -1.0], [0.0, 1.5, 0.0], [-1.0, 0.0, 1.0]]))
        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(logits=self.dropout(self.table[input_ids]))


def sample(ids, labels=None, attention=None):
    return {"input_ids": ids, "labels": labels if labels is not None else ids,
            "attention_mask": attention if attention is not None else [1] * len(ids)}


def test_token_weighting_masks_shift_and_mode_restore():
    model = TinyLM().train()
    model.dropout.eval()  # Restore heterogeneous submodule modes too.
    data = [sample([0, 1, 2, 2]), sample([2, 0]),
            sample([0, 2, 1], labels=[0, -100, 1], attention=[1, 1, 0])]
    # Third sample has no valid targets, but shares a batch with real tokens.
    expected = F.cross_entropy(model.table[[0, 1, 2, 2]], torch.tensor([1, 2, 2, 0]),
                               reduction="sum").item()
    stats = evaluator.measure_nll(model, data, 3)
    assert stats.token_count == 4
    assert stats.nll_sum == pytest.approx(expected, abs=1e-6)
    assert model.training and not model.dropout.training
    # Unequal batch/token lengths must yield identical token-weighted aggregate.
    assert evaluator.measure_nll(model, data, 1).nll_sum == pytest.approx(expected, abs=1e-6)


def test_masked_eos_is_positional_and_not_token_id():
    batch = evaluator._collate([sample([2, 2, 2]), sample([2, 2])])
    stats = evaluator.measure_nll(TinyLM(), [sample([2, 2, 2]), sample([2, 2])], 2)
    assert stats.token_count == 3
    assert batch["labels"][1, 2].item() == -100
    assert batch["labels"][0, 2].item() == 2


def test_failure_restores_training_mode():
    model = TinyLM().train()
    with pytest.raises(ValueError, match="nonempty"):
        evaluator.measure_nll(model, [sample([0, 1], labels=[-100, -100])], 1)
    assert model.training


def test_precision_rejects_float32():
    with pytest.raises(ValueError, match="BF16"):
        evaluator._verify_precision(TinyLM(), "unquantized")
    assert evaluator._verify_precision(TinyLM().bfloat16(), "unquantized")["compute_dtype"] == "bfloat16"


def test_software_versions_are_plain_json_strings():
    versions = evaluator._versions()
    assert all(type(value) is str for value in versions.values())
    # PyTorch exposes a TorchVersion str subclass; strict immutable records must
    # receive plain strings without loosening the artifact serializer.
    assert stable_id(versions)


def request():
    model = ModelRef("model", "models/original", "revision", "tokenizer", "tokrev", "bfloat16")
    data = DatasetRef("dataset", "data/tokens.jsonl", "test", "data/manifest.json",
                      "tokenizer", "tokrev", "preprocessing", 8)
    return EvaluationRequest("job", "attempt", model, {"forget_test": data},
                             "unquantized", {"per_device_batch_size": 2}, {}, "settings")


def test_evaluate_failure_is_explicit_with_traceback(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("test loader failure")
    monkeypatch.setattr(evaluator, "load_model", broken)
    result = evaluator.evaluate(request(), "cpu", tmp_path)
    assert result.status == "failed"
    assert "Traceback" in result.failure_reason and "test loader failure" in result.failure_reason
    assert not result.nll


def test_evaluate_all_splits_and_actual_precision(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator, "load_model", lambda *a, **kw: TinyLM().bfloat16())
    monkeypatch.setattr(evaluator, "load_data", lambda *a: [sample([0, 1, 2])])
    result = evaluator.evaluate(request(), "cpu", tmp_path)
    assert result.status == "complete"
    assert result.nll["forget_test"].token_count == 2
    assert result.inference_dtype == result.compute_dtype == "bfloat16"
    assert result.quantization_id == "none"
    assert evaluator._quantization_id({"backend": "bitsandbytes"}) == stable_id({"backend": "bitsandbytes"})


def test_worker_ignores_temporary_files_and_keeps_failed_attempt(monkeypatch, tmp_path):
    req = request()
    ready = publish_record(req, tmp_path)
    ready.with_suffix(".tmp").write_text("incomplete", encoding="utf-8")
    calls = []
    def fake_evaluate(req, device, root):
        calls.append(req.attempt_id)
        return EvaluationResult(req.job_id, req.attempt_id, req.model.model_id,
            req.variant, req.settings_id, {"forget_test": "dataset"}, "failed", {},
            "unknown", "unknown", {}, "Traceback: synthetic failure")
    monkeypatch.setattr(evaluator, "evaluate", fake_evaluate)
    evaluator.run_worker(tmp_path, "cpu", once=True)
    evaluator.run_worker(tmp_path, "cpu", once=True)
    assert calls == ["attempt"]
    result = read_record(tmp_path / "records/evaluation_result/job--attempt.json")
    assert result.status == "failed"
    assert not list((tmp_path / "worker_claims").glob("*.json"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RunPod CUDA FP4 backend smoke test")
def test_real_fp4_backend_preserves_float32_compute_with_bf16_modules(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from bitsandbytes.nn import Linear4bit
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32)
    bundle = tmp_path / "tiny"
    LlamaForCausalLM(config).bfloat16().save_pretrained(bundle, safe_serialization=True)
    ref = ModelRef("tiny", "tiny", "revision", "tokenizer", "tokrev", "bfloat16")
    quantization = {"backend": "bitsandbytes", "load_in_4bit": True,
        "bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": False,
        "bnb_4bit_compute_dtype": "float32", "non_quantized_parameter_dtype": "bfloat16",
        "calibration_required": False}
    model = evaluator.load_model(ref, {"inference_dtype": "bfloat16"}, quantization,
                                 "cuda:0", tmp_path, variant="quantized")
    try:
        assert evaluator.measure_nll(model, [sample([1, 2, 3, 4])], 1).token_count == 3
        precision = evaluator._verify_precision(model, "quantized")
        assert precision["compute_dtype"] == "float32"
        assert all(layer.compute_dtype == torch.float32 for layer in model.modules()
                   if isinstance(layer, Linear4bit))
    finally:
        del model
        torch.cuda.empty_cache()
