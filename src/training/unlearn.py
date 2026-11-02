"""GA_GDR with explicit storage precision and complete checkpoint publication."""
from __future__ import annotations

from contextlib import nullcontext
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import traceback
import uuid

import torch

from src.artifacts import (CheckpointRef, ModelRef, RunResult, RunSpec,
                           publish_record, resolve_artifact_path, stable_id)
from src.eval.evaluator import _collate, _shifted_nll, _versions, load_model
from src.inputs import load_data

LOG = logging.getLogger(__name__)


def checkpoint_schedule(dataset_size, batch_size, gradient_accum, max_epochs,
                        validation_every_epochs, early_validation_steps=()):
    """Round absolute epoch boundaries, flush accumulation remainders, include final."""
    if any(type(v) is not int or v <= 0 for v in
           (dataset_size, batch_size, gradient_accum, max_epochs)):
        raise ValueError("Dataset, batch, accumulation, and integer epochs must be positive")
    if not math.isfinite(validation_every_epochs) or validation_every_epochs <= 0:
        raise ValueError("Validation epoch interval must be finite and positive")
    updates = math.ceil(math.ceil(dataset_size / batch_size) / gradient_accum)
    total = updates * max_epochs
    steps = {total}
    boundary = 1
    while boundary * validation_every_epochs < max_epochs:
        steps.add(max(1, math.ceil(boundary * validation_every_epochs * updates)))
        boundary += 1
    for step in early_validation_steps:
        if type(step) is not int or step <= 0:
            raise ValueError("Early validation steps must be positive integers")
        if step <= total:
            steps.add(step)
    return sorted(steps)


def _validate_settings(settings):
    required = {"method": "ga_gdr", "optimizer": "adamw_torch",
                "parameter_dtype": "bfloat16", "optimizer_state_dtype": "bfloat16",
                "fp32_master_weights": False, "stochastic_rounding": False,
                "lr_scheduler": "constant", "warmup_steps": 0}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported unlearning recipe or precision settings")
    for key in ("per_device_batch_size", "gradient_accumulation_steps", "max_epochs"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("retain_loss_weight", "max_grad_norm", "weight_decay"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("adam_beta1", "adam_beta2"):
        if not math.isfinite(settings[key]) or not 0 <= settings[key] < 1:
            raise ValueError(f"{key} must lie in [0, 1)")
    if not math.isfinite(settings["adam_epsilon"]) or settings["adam_epsilon"] <= 0:
        raise ValueError("adam_epsilon must be finite and positive")
    if type(settings.get("shuffle", True)) is not bool:
        raise ValueError("shuffle must be boolean")


def _make_optimizer(model, settings, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if any(p.dtype != torch.bfloat16 or not p.requires_grad for p in model.parameters()):
        raise ValueError("Full-weight training requires every parameter trainable and BF16")
    # Single-tensor AdamW has BF16 moment buffers and updates the BF16 parameters
    # directly. The FP32 scalar step counter is not a master weight/moment tensor.
    return torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             betas=(settings["adam_beta1"], settings["adam_beta2"]),
                             eps=settings["adam_epsilon"], weight_decay=settings["weight_decay"],
                             foreach=False, fused=False)


def _verify_optimizer(model, optimizer):
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("Parameter storage precision changed during training")
    for state in optimizer.state.values():
        for name, value in state.items():
            if name != "step" and (not torch.is_tensor(value) or value.dtype != torch.bfloat16):
                raise ValueError(f"Unexpected optimizer state precision: {name}")


def _target_count(batch):
    attention = batch["attention_mask"].bool()
    return int((attention[:, 1:] & attention[:, :-1] & batch["labels"][:, 1:].ne(-100)).sum())


def _backward_window(model, pairs, retain_weight, device):
    """Exact per-branch token means across the actual accumulation window."""
    counts = [sum(_target_count(pair[branch]) for pair in pairs) for branch in (0, 1)]
    if not all(counts):
        raise ValueError("Unlearning branch contains no valid next-token targets")
    losses = [0.0, 0.0]
    for pair in pairs:
        for branch, coefficient in ((0, -1.0), (1, retain_weight)):
            if _target_count(pair[branch]) == 0:
                continue
            batch = {name: value.to(device) for name, value in pair[branch].items()}
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"], use_cache=False)
                nll, _ = _shifted_nll(outputs.logits, batch)
                loss = nll.float() / counts[branch]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite GA_GDR branch loss")
            (coefficient * loss).backward()
            losses[branch] += float(loss.detach())
            del outputs, nll, loss
    return losses


def _save_checkpoint(model, tokenizer, spec, step, epoch, root):
    directory = resolve_artifact_path(root, f"{spec.output_dir}/checkpoint-{step}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(temporary)
    metadata = {"run_id": spec.run_id, "step": step, "epoch": epoch,
                "learning_rate": spec.learning_rate, "seed": spec.seed,
                "parameter_dtype": "bfloat16", "optimizer_moment_dtype": "bfloat16",
                "optimizer_step_counter_dtype": "float32", "fp32_master_weights": False,
                "autocast_dtype": "bfloat16", "gradient_checkpointing": True,
                "effective_batch_size": spec.training["per_device_batch_size"] *
                    spec.training["gradient_accumulation_steps"],
                "software_versions": _versions(), "original_model_id": spec.original_model.model_id}
    with (temporary / "checkpoint_complete.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if directory.exists():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    temporary.rename(directory)
    identity = stable_id({"original": spec.original_model, "run": spec, "step": step})
    ref = CheckpointRef(model=ModelRef(
        model_id=identity, location=directory.relative_to(Path(root).resolve()).as_posix(),
        revision=identity, tokenizer_id=spec.original_model.tokenizer_id,
        tokenizer_revision=spec.original_model.tokenizer_revision,
        storage_dtype="bfloat16", source="artifact"), run_id=spec.run_id, step=step, epoch=epoch)
    return ref


def _load_tokenizer(ref, root):
    from transformers import AutoTokenizer
    if ref.source in ("artifact", "local"):
        location = resolve_artifact_path(root, ref.location)
        if (location / "tokenizer_config.json").exists():
            return AutoTokenizer.from_pretrained(location, local_files_only=True,
                                                 use_fast=False, legacy=True, trust_remote_code=False)
    if not ref.tokenizer_revision or len(ref.tokenizer_revision) != 40:
        raise ValueError("Tokenizer requires an immutable 40-character commit revision")
    return AutoTokenizer.from_pretrained(ref.tokenizer_id, revision=ref.tokenizer_revision,
                                         use_fast=False, legacy=True, trust_remote_code=False)


def train_run(spec: RunSpec, artifact_root: Path, on_checkpoint) -> RunResult:
    """Start from T and fresh RNG/optimizer state; preserve refs on publication failures."""
    model = optimizer = tokenizer = None
    checkpoints, callback_errors, step = [], [], 0
    status, failure = "complete", None
    try:
        _validate_settings(spec.training)
        random.seed(spec.seed)
        torch.manual_seed(spec.seed)
        torch.cuda.manual_seed_all(spec.seed)
        import numpy as np
        np.random.seed(spec.seed)
        device = torch.device(spec.training.get("device", "cuda:0"))
        if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("Training requires a BF16-capable CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = False
        forget = load_data(spec.train_datasets["forget_train"], artifact_root)
        retain = load_data(spec.train_datasets["retain_train"], artifact_root)
        if not len(forget) or not len(retain):
            raise ValueError("Both unlearning datasets must be nonempty")
        batch_size = spec.training["per_device_batch_size"]
        accumulation = spec.training["gradient_accumulation_steps"]
        batches_per_epoch = math.ceil(len(forget) / batch_size)
        updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
        total_steps = updates_per_epoch * spec.training["max_epochs"]
        schedule = set(spec.checkpoint_steps) | {total_steps}
        if any(type(value) is not int or not 0 < value <= total_steps for value in schedule):
            raise ValueError("Checkpoint steps lie outside the fixed training budget")
        model = load_model(spec.original_model, {"inference_dtype": "bfloat16"}, {},
                           str(device), artifact_root)
        model.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        tokenizer = _load_tokenizer(spec.original_model, artifact_root)
        optimizer = _make_optimizer(model, spec.training, spec.learning_rate)
        order_rng = torch.Generator().manual_seed(spec.seed)
        for epoch_index in range(spec.training["max_epochs"]):
            order = (torch.randperm(len(forget), generator=order_rng).tolist()
                     if spec.training.get("shuffle", True) else list(range(len(forget))))
            for first_batch in range(0, batches_per_epoch, accumulation):
                pairs = []
                last_batch = min(first_batch + accumulation, batches_per_epoch)
                for batch_index in range(first_batch, last_batch):
                    ids = order[batch_index * batch_size:(batch_index + 1) * batch_size]
                    # Upstream ForgetRetainDataset pairs by original item index;
                    # shuffling the forget order carries that pairing with it.
                    retain_ids = [index % len(retain) for index in ids]
                    pairs.append((_collate([forget[i] for i in ids]),
                                  _collate([retain[i] for i in retain_ids])))
                optimizer.zero_grad(set_to_none=True)
                loss_f, loss_r = _backward_window(model, pairs,
                    spec.training["retain_loss_weight"], device)
                if spec.training["max_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), spec.training["max_grad_norm"],
                                                   error_if_nonfinite=True, foreach=False)
                elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                step += 1
                _verify_optimizer(model, optimizer)
                if step == 1 or step % 25 == 0 or step == total_steps:
                    LOG.info("run=%s step=%d/%d forget_nll=%.6f retain_nll=%.6f gpu_gb=%.2f",
                             spec.run_id, step, total_steps, loss_f, loss_r,
                             torch.cuda.max_memory_allocated(device) / 1e9)
                if step in schedule:
                    epoch = epoch_index + last_batch / batches_per_epoch
                    ref = _save_checkpoint(model, tokenizer, spec, step, epoch, artifact_root)
                    checkpoints.append(ref)
                    publish_record(ref, artifact_root)
                    try:
                        on_checkpoint(ref)
                    except Exception:
                        callback_errors.append(traceback.format_exc())
                        LOG.exception("Checkpoint job publication failed; saved ref is retained")
        if callback_errors:
            failure = "Checkpoint callback publication failures (reconcile saved refs):\n" + "\n".join(callback_errors)
    except Exception:
        status, failure = "failed", traceback.format_exc()
        LOG.exception("Training failed: %s", spec.run_id)
    finally:
        del optimizer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return RunResult(run_id=spec.run_id, status=status, checkpoints=checkpoints,
                     last_step=step, failure_reason=failure)
