"""One model lifecycle for local evaluation and artifact-backed workers."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.artifacts import (
    EvaluationRequest, EvaluationResult, ModelRef, NLLStats,
    publish_record, read_record, resolve_artifact_path, stable_id,
)
from src.inputs import load_data

LOG = logging.getLogger(__name__)


def _versions():
    versions = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda)}
    for name in ("transformers", "accelerate", "bitsandbytes", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _quantization_id(settings):
    return stable_id(settings)


def _model_location(ref, root):
    if ref.source in ("huggingface", "hub"):
        if not ref.revision or len(ref.revision) != 40:
            raise ValueError("Hub weights require an immutable 40-character commit revision")
        return ref.location, {"revision": ref.revision}
    if ref.source not in ("artifact", "local"):
        raise ValueError(f"Unsupported model source: {ref.source}")
    location = resolve_artifact_path(root, ref.location)
    if not location.is_dir():
        raise FileNotFoundError(f"Model bundle missing: {location}")
    return str(location), {"local_files_only": True}


def _verify_precision(model, variant):
    """Read effective settings instead of trusting requested loader arguments."""
    floating = {str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    if floating != {"torch.bfloat16"}:
        raise ValueError(f"Expected BF16 floating parameters, observed {floating}")
    if variant == "quantized":
        from bitsandbytes.nn import Linear4bit
        layers = [m for m in model.modules() if isinstance(m, Linear4bit)]
        if not layers:
            raise ValueError("Quantized model contains no bitsandbytes Linear4bit modules")
        for layer in layers:
            state = layer.weight.quant_state
            if (layer.compute_dtype != torch.float32 or not layer.compute_type_is_set
                    or state is None or state.quant_type != "fp4" or state.nested):
                raise ValueError("Effective FP4/compute/double-quant settings differ from request")
        return {"inference_dtype": "bfloat16", "compute_dtype": "float32",
                "quantized_linear_count": len(layers)}
    return {"inference_dtype": "bfloat16", "compute_dtype": "bfloat16"}


def load_model(model_ref: ModelRef, evaluation: dict, quantization: dict,
               device: str, artifact_root: Path, *, variant="unquantized"):
    """Load pinned weights; BF16 nonquantized modules and explicit FP4 arithmetic."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if evaluation.get("inference_dtype", "bfloat16") != "bfloat16":
        raise ValueError("Only bfloat16 inference is supported")
    if variant not in ("unquantized", "quantized"):
        raise ValueError(f"Unsupported evaluation variant: {variant}")
    location, kwargs = _model_location(model_ref, artifact_root)
    kwargs.update(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                  trust_remote_code=False, attn_implementation="sdpa")
    if variant == "quantized":
        required = {"backend": "bitsandbytes", "load_in_4bit": True,
                    "bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": False,
                    "bnb_4bit_compute_dtype": "float32",
                    "non_quantized_parameter_dtype": "bfloat16",
                    "calibration_required": False}
        if any(quantization.get(k) != v for k, v in required.items()):
            raise ValueError("Only the explicit bitsandbytes FP4 recipe is supported")
        if not str(device).startswith("cuda"):
            raise ValueError("FP4 evaluation requires a CUDA device")
        kwargs.update(device_map={"": device}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=torch.float32))
    model = AutoModelForCausalLM.from_pretrained(location, **kwargs)
    if variant == "unquantized":
        model.to(device)
    else:
        from bitsandbytes.nn import Linear4bit
        for layer in model.modules():
            if isinstance(layer, Linear4bit):
                # Lock the configured type against lazy input-dtype inference.
                layer.compute_dtype = torch.float32
                layer.compute_type_is_set = True
    model.eval()
    model.config.use_cache = False
    model._experiment_precision = _verify_precision(model, variant)
    LOG.info("Loaded %s %s: %s", model_ref.model_id, variant, model._experiment_precision)
    return model


def _collate(samples):
    """Pad by position, preserving actual EOS tokens even when EOS is the pad ID."""
    length = max(len(sample["input_ids"]) for sample in samples)
    batch = {name: torch.full((len(samples), length), value, dtype=torch.long)
             for name, value in (("input_ids", 0), ("attention_mask", 0), ("labels", -100))}
    for i, sample in enumerate(samples):
        for name in batch:
            values = torch.as_tensor(sample[name], dtype=torch.long)
            batch[name][i, :len(values)] = values
    return batch


def _shifted_nll(logits, batch):
    targets = batch["labels"][:, 1:].clone()
    attention = batch["attention_mask"].bool()
    valid = attention[:, 1:] & attention[:, :-1] & targets.ne(-100)
    count = int(valid.sum().item())
    if not count:
        return logits.new_zeros((), dtype=torch.float64), 0
    targets.masked_fill_(~valid, -100)
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                             targets.reshape(-1), ignore_index=-100, reduction="none")
    return losses.double().sum(), count


def measure_nll(model, dataset, batch_size: int) -> NLLStats:
    """Sum next-token NLL in nats over masked targets; restore every module's mode."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    total, count = 0.0, 0
    try:
        model.eval()
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=_collate):
                batch = {name: value.to(device) for name, value in batch.items()}
                # Autocast stays off: FP4 kernels must retain float32 compute.
                with torch.autocast(device.type, enabled=False):
                    outputs = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"], use_cache=False)
                    nll, tokens = _shifted_nll(outputs.logits, batch)
                total += float(nll.item())
                count += tokens
        if not count or not math.isfinite(total):
            raise ValueError("NLL requires a finite sum and nonempty target tokens")
        return NLLStats(nll_sum=total, token_count=count)
    finally:
        for module, training in modes:
            module.training = training


def evaluate(request: EvaluationRequest, device: str, artifact_root: Path) -> EvaluationResult:
    model = None
    nll, precision = {}, {"inference_dtype": "unknown", "compute_dtype": "unknown"}
    status, failure = "complete", None
    try:
        if not request.datasets:
            raise ValueError("Evaluation request has no datasets")
        for ref in request.datasets.values():
            if (ref.tokenizer_id, ref.tokenizer_revision) != (
                    request.model.tokenizer_id, request.model.tokenizer_revision):
                raise ValueError("Evaluation dataset tokenizer differs from model tokenizer")
        model = load_model(request.model, request.evaluation, request.quantization,
                           device, artifact_root, variant=request.variant)
        precision = _verify_precision(model, request.variant)
        for name, ref in request.datasets.items():
            nll[name] = measure_nll(model, load_data(ref, artifact_root),
                                    int(request.evaluation["per_device_batch_size"]))
            precision = _verify_precision(model, request.variant)
    except Exception:
        status, failure = "failed", traceback.format_exc()
        LOG.exception("Evaluation failed: %s/%s", request.job_id, request.attempt_id)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return EvaluationResult(
        job_id=request.job_id, attempt_id=request.attempt_id, model_id=request.model.model_id,
        variant=request.variant, settings_id=request.settings_id,
        dataset_ids={name: ref.dataset_id for name, ref in request.datasets.items()},
        status=status, nll=nll, failure_reason=failure, software_versions=_versions(),
        tokenizer_id=request.model.tokenizer_id, tokenizer_revision=request.model.tokenizer_revision,
        quantization_id=_quantization_id(request.quantization) if request.variant == "quantized" else "none",
        inference_dtype=precision["inference_dtype"], compute_dtype=precision["compute_dtype"])


def run_worker(artifact_root: Path, device: str, *, once=False, poll_seconds=2.0):
    """Process immutable ready requests; failed attempts require a new attempt ID."""
    root = Path(artifact_root)
    claims = root / "worker_claims"
    claims.mkdir(parents=True, exist_ok=True)
    while True:
        for path in sorted((root / "records" / "evaluation_request").glob("*.json")):
            request = read_record(path, EvaluationRequest)
            result_path = root / "records" / "evaluation_result" / path.name
            if result_path.exists():
                continue
            claim = claims / path.name
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump({"pid": os.getpid(), "started_at": time.time()}, handle)
                publish_record(evaluate(request, device, root), root)
            finally:
                claim.unlink(missing_ok=True)
        if once:
            return
        time.sleep(poll_seconds)
