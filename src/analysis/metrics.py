"""Rounding probes, matched retention measurements, and descriptive reports."""

import csv
import gc
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from src.artifacts import AnalysisResult, PredictorResult, RetentionResult, stable_id
from src.analysis.selection import _successful


def _probe_tensor(tensor, learning_rates, device, chunk_size=1_048_576):
    import torch

    counts = [0] * len(learning_rates)
    flattened = tensor.detach().reshape(-1)
    with torch.no_grad():
        for start in range(0, flattened.numel(), chunk_size):
            stored = flattened[start:start + chunk_size].to(device=device, dtype=torch.bfloat16)
            if not torch.isfinite(stored).all().item():
                raise ValueError("Predictor weights must be finite")
            wider = stored.to(torch.float32)
            for index, rate in enumerate(learning_rates):
                positive = (wider + rate).to(torch.bfloat16)
                negative = (wider - rate).to(torch.bfloat16)
                counts[index] += ((positive == stored) & (negative == stored)).sum().item()
    return counts, flattened.numel()


def compute_predictor(original_model, learning_rates, training, artifact_root):
    """Probe all original nn.Linear weights, including lm_head, on GPU chunks."""
    import torch
    from src.eval.evaluator import load_model

    if training.get("parameter_dtype") != "bfloat16" or original_model.storage_dtype != "bfloat16":
        raise ValueError("Predictor currently supports bfloat16 parameter storage only")
    if not learning_rates or any(not math.isfinite(rate) or rate <= 0 for rate in learning_rates):
        raise ValueError("Learning rates must be finite and positive")
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError("Predictor learning rates must be unique")
    if not torch.cuda.is_available():
        raise RuntimeError("Predictor requires the RunPod CUDA device")
    model = load_model(original_model, {"inference_dtype": "bfloat16"}, {}, "cpu", Path(artifact_root))
    counts, total, names, seen = [0] * len(learning_rates), 0, [], set()
    try:
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear) or id(module.weight) in seen:
                continue
            if module.weight.dtype != torch.bfloat16:
                raise ValueError(f"Original linear weight {name} is not stored in bfloat16")
            seen.add(id(module.weight))
            found, size = _probe_tensor(module.weight, learning_rates, "cuda:0")
            counts = [old + new for old, new in zip(counts, found)]
            total += size
            names.append(name)
        if not total:
            raise ValueError("Original model contains no nn.Linear weights")
        scope = f"all torch.nn.Linear.weight including lm_head; unique parameters; tensors={len(names)}; elements={total}"
        calculation = stable_id({"model": original_model, "linear_names": names, "elements": total,
                                 "working_dtype": "float32", "storage_dtype": "bfloat16",
                                 "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
                                 "probe": "both (float32(T)+lr).bf16==T and (float32(T)-lr).bf16==T"})
        return [PredictorResult(rate, count / total, original_model.model_id, scope,
                                "bfloat16", "float32", calculation)
                for rate, count in zip(learning_rates, counts)]
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _compatible_quartet(records, checkpoint):
    if any(record is None for record in records):
        return "Missing or ambiguous complete T, Q(T), U, or Q(U) measurement"
    t, qt, u, qu = records
    if len({r.settings_id for r in records}) != 1:
        return "Evaluation settings identities differ"
    expected_tokenizer = (checkpoint.model.tokenizer_id, checkpoint.model.tokenizer_revision)
    if any((r.tokenizer_id, r.tokenizer_revision) != expected_tokenizer for r in records):
        return "Tokenizer identities differ from the selected checkpoint"
    if not expected_tokenizer[0] or not expected_tokenizer[1]:
        return "Missing tokenizer identity"
    if any(r.software_versions != t.software_versions for r in records):
        return "Software versions differ across measurements"
    for split in ("forget_test", "retain_test"):
        if any(split not in r.nll or split not in r.dataset_ids for r in records):
            return f"Every variant must measure {split}"
        if len({(r.dataset_ids[split], r.nll[split].token_count) for r in records}) != 1:
            return f"Dataset or target-token count differs for {split}"
    for left, right in ((t, u), (qt, qu)):
        if (left.inference_dtype, left.compute_dtype, left.quantization_id) != (
                right.inference_dtype, right.compute_dtype, right.quantization_id):
            return "Actual precision or quantization identities differ within a variant"
    if t.quantization_id != "none" or qt.quantization_id in ("", "none"):
        return "Invalid unquantized or quantized settings identity"
    if any(not r.inference_dtype or not r.compute_dtype for r in records):
        return "Missing actual precision identity"
    return None


def measure_retention(selection, measurements, *, minimum_denominator=1e-8):
    """Keep raw effects and invalidate tiny denominators; never clamp ratios."""
    if not math.isfinite(minimum_denominator) or minimum_denominator <= 0:
        raise ValueError("Minimum denominator must be finite and positive")
    try:
        successful = _successful(measurements)
        conflict = None
    except ValueError as error:
        successful, conflict = [], str(error)
    output = []
    for entry in selection.selections:
        records = []
        if entry.checkpoint is not None:
            for model_id, variant in ((selection.original_model_id, "unquantized"),
                                      (selection.original_model_id, "quantized"),
                                      (entry.checkpoint.model.model_id, "unquantized"),
                                      (entry.checkpoint.model.model_id, "quantized")):
                found = [r for r in successful if r.model_id == model_id and r.variant == variant
                         and "forget_test" in r.nll]
                records.append(found[0] if len(found) == 1 else None)
        else:
            records = [None] * 4
        raw = [r.nll["forget_test"].mean if r and "forget_test" in r.nll else None for r in records]
        t, qt, u, qu = raw
        unquantized_gap = u - t if t is not None and u is not None else None
        quantized_gap = qu - qt if qt is not None and qu is not None else None
        reason = conflict or (selection.failure_reason or "Selection is unavailable"
                              if selection.status != "complete" else None)
        if reason is None:
            reason = _compatible_quartet(records, entry.checkpoint)
        retain_u = retain_q = ratio = None
        if reason is None:
            retain_u = records[2].nll["retain_test"].mean - records[0].nll["retain_test"].mean
            retain_q = records[3].nll["retain_test"].mean - records[1].nll["retain_test"].mean
            if (not all(math.isfinite(value) for value in raw)
                    or not math.isfinite(unquantized_gap) or not math.isfinite(quantized_gap)):
                reason = "Nonfinite forget NLL or effect gap"
            elif unquantized_gap <= minimum_denominator:
                reason = f"Unquantized forget gap must exceed {minimum_denominator:g}; raw gap preserved"
            else:
                ratio = quantized_gap / unquantized_gap
                if not math.isfinite(ratio):
                    ratio, reason = None, "Nonfinite retention ratio"
        reference = next((r for r in reversed(records) if r is not None), None)
        output.append(RetentionResult(
            entry.run_id, entry.learning_rate, entry.seed,
            reference.dataset_ids.get("forget_test", "") if reference else "",
            reference.settings_id if reference else "", t, qt, u, qu,
            unquantized_gap, quantized_gap, ratio, retain_u, retain_q,
            "invalid" if reason else "complete", reason))
    return output


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_figure(figure, output_dir, name):
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / f"{name}.{extension}"
        figure.savefig(path, bbox_inches="tight", dpi=160)
        paths.append(str(path))
    return paths


def write_validation_trajectories(expected_runs, checkpoints, validation_results, output_dir):
    """Write all available validation evidence, also when matching is unavailable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _successful([r for r in validation_results if r.variant == "unquantized"])
    by_model = {r.model_id: r for r in results if "forget_validation" in r.nll}
    rows = []
    for run in expected_runs:
        original = by_model.get(run.original_model.model_id)
        if original:
            rows.append(dict(run_id=run.run_id, learning_rate=run.learning_rate, seed=run.seed,
                             step=0, epoch=0, forget_validation_nll=original.nll["forget_validation"].mean,
                             retain_validation_nll=original.nll["retain_validation"].mean
                             if "retain_validation" in original.nll else None, model_id=original.model_id))
        for checkpoint in sorted((cp for cp in checkpoints if cp.run_id == run.run_id), key=lambda cp: cp.step):
            measurement = by_model.get(checkpoint.model.model_id)
            rows.append(dict(run_id=run.run_id, learning_rate=run.learning_rate, seed=run.seed,
                             step=checkpoint.step, epoch=checkpoint.epoch,
                             forget_validation_nll=measurement.nll["forget_validation"].mean if measurement else None,
                             retain_validation_nll=measurement.nll["retain_validation"].mean
                             if measurement and "retain_validation" in measurement.nll else None,
                             model_id=checkpoint.model.model_id))
    table = output_dir / "validation_trajectories.csv"
    fields = ["run_id", "learning_rate", "seed", "step", "epoch", "forget_validation_nll",
              "retain_validation_nll", "model_id"]
    _write_csv(table, rows, fields)
    plt = _plotting()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in expected_runs:
        curve = [row for row in rows if row["run_id"] == run.run_id]
        for axis, column, title in zip(axes, ("forget_validation_nll", "retain_validation_nll"),
                                       ("Forget validation", "Retain validation")):
            available = [row for row in curve if row[column] is not None]
            axis.plot([row["epoch"] for row in available], [row[column] for row in available],
                      marker="o", label=f"LR {run.learning_rate:g}, seed {run.seed}")
            axis.set(xlabel="Completed epochs", ylabel="Token NLL (nats)", title=title)
    if rows:
        axes[0].legend(fontsize=7)
    figure.tight_layout()
    figures = _save_figure(figure, output_dir, "validation_trajectories")
    plt.close(figure)
    return {"table_paths": [str(table)], "figure_paths": figures}


def _correlation(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def _ranks(values):
    return [sum(other < value for other in values) + (sum(other == value for other in values) + 1) / 2
            for value in values]


def assess_predictor(predictors, retention, selection, output_dir):
    """Compare LR-level seed means descriptively, keeping every run visible."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor_by_lr = {}
    for predictor in predictors:
        if (predictor.model_id != selection.original_model_id or not math.isfinite(predictor.value)
                or not 0 <= predictor.value <= 1 or not math.isfinite(predictor.learning_rate)
                or predictor.learning_rate <= 0):
            raise ValueError("Predictor identity or value is incompatible with selection")
        if predictor.learning_rate in predictor_by_lr:
            raise ValueError("Duplicate predictor learning rate")
        predictor_by_lr[predictor.learning_rate] = predictor
    identities = {(p.model_id, p.tensor_scope, p.storage_dtype, p.working_dtype, p.calculation_id) for p in predictors}
    if len(identities) > 1:
        raise ValueError("Predictor calculation identities differ")
    by_run = {row.run_id: row for row in retention}
    if len(by_run) != len(retention):
        raise ValueError("Duplicate retention run records")
    if set(by_run) - {entry.run_id for entry in selection.selections}:
        raise ValueError("Retention includes runs outside the frozen selection")
    baseline_identities = {(r.dataset_id, r.settings_id, r.nll_t, r.nll_qt)
                           for r in retention if r.status == "complete"}
    if len(baseline_identities) > 1:
        raise ValueError("Retention rows do not share matching data/settings/original baselines")
    rows = []
    for entry in selection.selections:
        record = by_run.get(entry.run_id)
        row = asdict(record) if record else {"status": "missing", "failure_reason": "Missing retention record"}
        predictor = predictor_by_lr.get(entry.learning_rate)
        row.update(run_id=entry.run_id, learning_rate=entry.learning_rate, seed=entry.seed,
                   predictor_e=predictor.value if predictor else None,
                   validation_nll=entry.achieved_nll, validation_gain=entry.gain_over_original,
                   matching_error=entry.signed_error, selected_step=entry.checkpoint.step if entry.checkpoint else None)
        if record and (record.learning_rate, record.seed) != (entry.learning_rate, entry.seed):
            raise ValueError("Retention LR/seed identity differs from selection")
        rows.append(row)
    table = output_dir / "per_run.csv"
    fields = ["run_id", "learning_rate", "seed", "predictor_e", "status", "selected_step",
              "validation_nll", "validation_gain", "matching_error", "dataset_id", "settings_id",
              "nll_t", "nll_qt", "nll_u", "nll_qu", "unquantized_gap", "quantized_gap",
              "retention_ratio", "retain_unquantized_delta", "retain_quantized_delta", "failure_reason"]
    _write_csv(table, rows, fields)
    aggregates = []
    for rate in sorted({entry.learning_rate for entry in selection.selections}):
        group = [row for row in rows if row["learning_rate"] == rate]
        valid = [row["retention_ratio"] for row in group if row.get("status") == "complete"
                 and row.get("retention_ratio") is not None and math.isfinite(row["retention_ratio"])]
        aggregates.append(dict(learning_rate=rate, predictor_e=predictor_by_lr[rate].value
                               if rate in predictor_by_lr else None, expected_seed_count=len(group),
                               valid_seed_count=len(valid), retention_mean=statistics.mean(valid) if valid else None,
                               retention_sd=statistics.stdev(valid) if len(valid) > 1 else None))
    aggregate_table = output_dir / "learning_rate_summary.csv"
    _write_csv(aggregate_table, aggregates, ["learning_rate", "predictor_e", "expected_seed_count",
                                           "valid_seed_count", "retention_mean", "retention_sd"])
    comparable = [row for row in aggregates if selection.status == "complete" and row["retention_mean"] is not None
                  and row["predictor_e"] is not None and row["valid_seed_count"] == row["expected_seed_count"]]
    es = [row["predictor_e"] for row in comparable]
    lrs = [math.log10(row["learning_rate"]) for row in comparable]
    effects = [row["retention_mean"] for row in comparable]
    single_seed = any(row["expected_seed_count"] < 2 for row in aggregates)
    limitations = ["Exploratory comparison within one fixed model/data/precision/quantizer setup.",
                   "Correlations use one seed mean per learning rate; seeds are not independent LR settings.",
                   "E is determined by LR and original weights; correlation alone does not establish added value beyond LR.",
                   "Validation matching does not guarantee matched test forgetting; inspect raw test effects and retain costs."]
    if single_seed:
        limitations.append("At least one LR has one seed: seed variability cannot be estimated there.")
    complete = (selection.status == "complete" and len(comparable) == len(aggregates)
                and bool(aggregates) and all(row.get("status") == "complete" for row in rows))
    summary = dict(comparison="matched" if complete else "unavailable_or_incomplete",
                   learning_rate_count=len(aggregates), comparable_learning_rate_count=len(comparable),
                   seed_counts={str(row["learning_rate"]): row["expected_seed_count"] for row in aggregates},
                   pearson_e_retention=_correlation(es, effects),
                   pearson_log_lr_retention=_correlation(lrs, effects),
                   spearman_e_retention=_correlation(_ranks(es), _ranks(effects)),
                   spearman_log_lr_retention=_correlation(_ranks(lrs), _ranks(effects)),
                   limitations=limitations)
    plt = _plotting()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for row in rows:
        if row.get("retention_ratio") is not None and row.get("status") == "complete":
            if row["predictor_e"] is not None:
                axes[0].scatter(row["predictor_e"], row["retention_ratio"], alpha=0.55)
            axes[1].scatter(row["learning_rate"], row["retention_ratio"], alpha=0.55)
        if row["matching_error"] is not None:
            axes[2].scatter(row["learning_rate"], row["matching_error"], alpha=0.7)
    for row in comparable:
        for axis, key in ((axes[0], "predictor_e"), (axes[1], "learning_rate")):
            axis.errorbar(row[key], row["retention_mean"], yerr=row["retention_sd"],
                          fmt="D", color="black", capsize=3)
    axes[0].set(xlabel="E(LR)", ylabel="Forget-effect retention", title="Seeds and LR means")
    axes[1].set(xlabel="Learning rate", ylabel="Forget-effect retention", title="Same outcomes versus LR", xscale="log")
    axes[2].set(xlabel="Learning rate", ylabel="Achieved minus target NLL", title="Validation matching error", xscale="log")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].axhline(selection.decision.max_abs_error, color="gray", linestyle=":")
    axes[2].axhline(-selection.decision.max_abs_error, color="gray", linestyle=":")
    figure.tight_layout()
    figures = _save_figure(figure, output_dir, "predictor_assessment")
    plt.close(figure)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    headline = (f"Matched comparison includes {len(comparable)} learning rates."
                if complete else "A complete matched comparison is unavailable; inspect per-run reasons.")
    text = headline + "\n\n"
    e_correlation = summary["pearson_e_retention"]
    lr_correlation = summary["pearson_log_lr_retention"]
    show = lambda value: "undefined" if value is None else f"{value:.3f}"
    text += (f"Descriptive Pearson correlations of LR-level mean retention: E(LR) {show(e_correlation)}; "
             f"log10(LR) {show(lr_correlation)}.\n\n")
    if complete and e_correlation is not None:
        text += ("Lower E tended to coincide with higher retention across these learning rates."
                 if e_correlation < 0 else
                 "The observed E trend did not follow the predicted direction of lower E with higher retention.") + "\n\n"
    valid_rows = [row for row in rows if row.get("status") == "complete"]
    if valid_rows:
        text += (f"Observed retention ranged from {min(row['retention_ratio'] for row in valid_rows):.4g} "
                 f"to {max(row['retention_ratio'] for row in valid_rows):.4g}; ratios are not clamped. "
                 "Per-run tables preserve test forgetting gaps, validation matching errors, and retain-NLL changes.\n\n")
    text += "\n".join(limitations) + "\n"
    narrative = output_dir / "summary.md"
    narrative.write_text(text, encoding="utf-8")
    return AnalysisResult("complete" if complete else "unavailable", table_paths=[str(table), str(aggregate_table),
                           str(summary_path), str(narrative)], figure_paths=figures, summary=summary,
                          failure_reason=None if complete else selection.failure_reason or "Missing or invalid matched measurements")
