"""Add a report from finalized refinement records. Run on RunPod after freeze."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


DENOMINATOR_CUTOFF = 1e-8


def read(path, kind):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != kind or value.get("schema_version") != 1:
        raise ValueError(f"Expected schema-1 {kind}: {path}")
    return value


def inside(root, location):
    path = (root / location).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Reference escapes artifact root: {location}")
    return path


def unique(records, description):
    if len(records) != 1:
        raise ValueError(f"Need exactly one {description}; found {len(records)}")
    return records[0]


def mean(record, split):
    stats = record.get("nll", {}).get(split) if record else None
    if not stats or stats["token_count"] <= 0 or not math.isfinite(stats["nll_sum"]):
        return None
    return stats["nll_sum"] / stats["token_count"]


def difference(left, right):
    return left - right if left is not None and right is not None else None


def display(value):
    if value is None:
        return "unmeasured"
    return f"{value:.8g}" if isinstance(value, (int, float)) else str(value).replace("|", "/").replace("\n", " ")


def table(lines, headers, rows):
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
    lines.extend("| " + " | ".join(display(value) for value in row) + " |" for row in rows)
    lines.append("")


def successful(evaluations):
    jobs = {}
    ignored = {"attempt_id", "failure_reason"}
    for record in evaluations:
        if record["status"] != "complete":
            continue
        previous = jobs.get(record["job_id"])
        if previous and {k: v for k, v in previous.items() if k not in ignored} != {
                k: v for k, v in record.items() if k not in ignored}:
            raise ValueError(f"Conflicting successful attempts: {record['job_id']}")
        jobs[record["job_id"]] = record
    return list(jobs.values())


def measurement(records, model_id, variant, split):
    candidates = [record for record in records if record["model_id"] == model_id
                  and record["variant"] == variant and split in record["nll"]]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous measurement: {model_id}/{variant}/{split}")
    return candidates[0] if candidates else None


def compatible(records, model):
    if any(record is None for record in records):
        return "Missing complete test variant"
    t, qt, u, qu = records
    if len({record["settings_id"] for record in records}) != 1:
        return "Settings identities differ"
    for record in records:
        if (record["tokenizer_id"], record["tokenizer_revision"]) != (model["tokenizer_id"], model["tokenizer_revision"]):
            return "Tokenizer identities differ"
        if record["software_versions"] != t["software_versions"]:
            return "Software versions differ"
    for split in ("forget_test", "retain_test"):
        if any(split not in record["nll"] or split not in record["dataset_ids"] for record in records):
            return f"Missing {split}"
        identities = {(record["dataset_ids"][split], record["nll"][split]["token_count"]) for record in records}
        if len(identities) != 1:
            return f"Dataset/token counts differ for {split}"
    for left, right in ((t, u), (qt, qu)):
        fields = ("inference_dtype", "compute_dtype", "quantization_id")
        if any(left[field] != right[field] for field in fields):
            return "Actual precision/quantization identities differ within variant"
    if t["quantization_id"] != "none" or qt["quantization_id"] in ("", "none"):
        return "Missing or invalid quantization identity"
    return None


def plot(rows, output):
    if not any(row["unquantized_forget_gap"] is not None for row in rows):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    series = [
        (("signed_matching_error", "Achieved minus target", "o"),),
        (("unquantized_forget_gap", "Test U − T", "o"), ("quantized_forget_gap", "Test Q(U) − Q(T)", "s")),
        (("test_retain_delta_u", "Blind test U − T", "o"), ("test_retain_delta_qu", "Blind test Q(U) − Q(T)", "s"),
         ("validation_retain_delta", "Validation U − T", "^")),
    ]
    for seed in sorted({row["seed"] for row in rows}):
        group = sorted((row for row in rows if row["seed"] == seed), key=lambda row: row["learning_rate"])
        for axis, fields in zip(axes, series):
            for field, label, marker in fields:
                axis.plot([row["learning_rate"] for row in group],
                          [row[field] if row[field] is not None else float("nan") for row in group],
                          marker=marker, label=f"{label}, seed {seed}")
    for axis, title, ylabel in zip(axes, ("Actual validation mismatch", "Raw forget-test effects", "Retain utility costs"),
                                    ("Signed NLL error (nats)", "Forget NLL change (nats)", "Retain NLL change (nats)")):
        axis.set(xscale="log", xlabel="Learning rate", ylabel=ylabel, title=title)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    paths = []
    for extension in ("png", "svg"):
        path = output / f"raw_effects_and_utility.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path.name)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Final RunPod artifact root")
    parser.add_argument("--analysis", help="Optional artifact-relative final AnalysisResult JSON")
    parser.add_argument("--sweep", help="Artifact-relative complete SweepIndex JSON; required if multiple snapshots match")
    parser.add_argument("--output", help="New artifact-relative report directory; must not already exist")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.analysis:
        analysis_path = inside(root, args.analysis)
    else:
        candidates = [path for path in (root / "records/analysis_result").glob("*.json")
                      if read(path, "analysis_result").get("selection_path")]
        analysis_path = unique(candidates, "final AnalysisResult (or supply --analysis)")
    analysis = read(analysis_path, "analysis_result")
    if analysis["status"] not in ("complete", "unavailable") or not analysis.get("selection_path"):
        parser.error("No finalized analysis with a frozen selection; do not run during training")
    selection_path = inside(root, analysis["selection_path"])
    selection = read(selection_path, "selection_result")
    run_ids = {entry["run_id"] for entry in selection["selections"]}
    if args.sweep:
        index_path = inside(root, args.sweep)
        index = read(index_path, "sweep_index")
    else:
        snapshots = [(path, read(path, "sweep_index")) for path in (root / "records/sweep_index").glob("*.json")]
        index_path, index = unique([(path, value) for path, value in snapshots if value["status"] == "complete"
                                   and {run["run_id"] for run in value["runs"]} == run_ids],
                                  "complete matching sweep snapshot; supply --sweep records/sweep_index/<id>.json if ambiguous")
    if index["status"] != "complete" or {run["run_id"] for run in index["runs"]} != run_ids:
        parser.error("Selected --sweep must be complete and contain exactly the frozen selection's runs")
    config = index["config"]
    evaluations_paths = sorted(set(index["evaluation_result_paths"] + analysis["measurement_paths"]))
    all_evaluations = [read(inside(root, path), "evaluation_result") for path in evaluations_paths]
    evaluations = successful(all_evaluations)
    analysis_evaluations = successful([read(inside(root, path), "evaluation_result")
                                      for path in analysis["measurement_paths"]])
    outcomes = {record["run_id"]: record for record in
                (read(inside(root, path), "run_result") for path in index["run_result_paths"])}
    predictors = [read(inside(root, path), "predictor_result") for path in index["predictor_paths"]]
    retention = ([read(path, "retention_result") for path in (root / "records/retention_result").glob("*.json")]
                 if selection["status"] == "complete" else [])
    original_id = selection["original_model_id"]
    base_val = measurement(evaluations, original_id, "unquantized", "forget_validation")
    rows, checks, provenance = [], [], []
    for entry in sorted(selection["selections"], key=lambda entry: (entry["learning_rate"], entry["seed"])):
        run_id = entry["run_id"]
        run = unique([run for run in index["runs"] if run["run_id"] == run_id], f"RunSpec {run_id}")
        outcome = outcomes.get(run_id)
        prediction = unique([p for p in predictors if p["learning_rate"] == entry["learning_rate"]
                             and p["model_id"] == original_id], f"predictor for {run_id}")
        saved = sorted(outcome["checkpoints"], key=lambda cp: cp["step"]) if outcome else []
        curve = [mean(measurement(evaluations, cp["model"]["model_id"], "unquantized", "forget_validation"),
                      "forget_validation") for cp in saved]
        observed = [value for value in curve if value is not None]
        first = measurement(evaluations, saved[0]["model"]["model_id"], "unquantized", "forget_validation") if saved else None
        final = measurement(evaluations, saved[-1]["model"]["model_id"], "unquantized", "forget_validation") if saved else None
        checkpoint = entry["checkpoint"]
        selected_id = checkpoint["model"]["model_id"] if checkpoint else None
        selected_val = measurement(evaluations, selected_id, "unquantized", "forget_validation") if checkpoint else None
        quartet = [measurement(analysis_evaluations, model, variant, "forget_test") for model, variant in
                   ((original_id, "unquantized"), (original_id, "quantized"),
                    (selected_id, "unquantized"), (selected_id, "quantized"))] if selection["status"] == "complete" else [None] * 4
        raw = [mean(record, "forget_test") for record in quartet]
        matching = []
        if all(record is not None for record in quartet):
            reference = quartet[-1]
            matching = [record for record in retention
                        if (record["run_id"], record["learning_rate"], record["seed"])
                        == (run_id, entry["learning_rate"], entry["seed"])
                        and record["dataset_id"] == reference["dataset_ids"].get("forget_test")
                        and record["settings_id"] == reference["settings_id"]
                        and [record[field] for field in ("nll_t", "nll_qt", "nll_u", "nll_qu")] == raw]
        retained = unique(matching, f"retention record scoped to this frozen quartet for {run_id}") if matching else None
        missing_retention = ("Selection unavailable; retention intentionally unmeasured"
                             if selection["status"] != "complete" else
                             "No retention record matches this analysis's complete selected test quartet"
                             if all(record is not None for record in quartet) else
                             "No complete selected test quartet in this analysis; prior retention is not reused")
        gap, qgap = difference(raw[2], raw[0]), difference(raw[3], raw[1])
        denominator_valid = gap is not None and math.isfinite(gap) and gap > DENOMINATOR_CUTOFF
        denominator_status = "unmeasured" if gap is None else "valid" if denominator_valid else "invalid"
        identity_failure = compatible(quartet, checkpoint["model"]) if checkpoint else "No selected checkpoint"
        test_retain = [mean(record, "retain_test") for record in quartet]
        selected_retain = mean(selected_val, "retain_validation")
        row = dict(run_id=run_id, learning_rate=entry["learning_rate"], seed=entry["seed"],
                   run_status=outcome["status"] if outcome else "missing", last_step=outcome["last_step"] if outcome else None,
                   checkpoint_count=len(saved), predictor_e=prediction["value"],
                   checkpoint_forget_min=min(observed) if observed else None, checkpoint_forget_max=max(observed) if observed else None,
                   first_step=saved[0]["step"] if saved else None, first_forget_validation=mean(first, "forget_validation"),
                   final_forget_validation=mean(final, "forget_validation"), final_retain_validation=mean(final, "retain_validation"),
                   selection_status=selection["status"], selected_step=checkpoint["step"] if checkpoint else None,
                   selected_epoch=checkpoint["epoch"] if checkpoint else None, achieved_validation_nll=entry["achieved_nll"],
                   gain_over_original=entry["gain_over_original"], signed_matching_error=entry["signed_error"],
                   abs_matching_error=abs(entry["signed_error"]) if entry["signed_error"] is not None else None,
                   selection_failure=entry["failure_reason"],
                   validation_retain_t=mean(base_val, "retain_validation"), validation_retain_u=selected_retain,
                   validation_retain_delta=difference(selected_retain, mean(base_val, "retain_validation")),
                   test_forget_t=raw[0], test_forget_qt=raw[1], test_forget_u=raw[2], test_forget_qu=raw[3],
                   unquantized_forget_gap=gap, quantized_forget_gap=qgap,
                   retention_ratio=retained["retention_ratio"] if retained else None,
                   denominator_validity=denominator_status, denominator_cutoff=DENOMINATOR_CUTOFF,
                   retention_status=retained["status"] if retained else "unmeasured",
                   retention_failure=retained["failure_reason"] if retained else missing_retention,
                   test_identity_check=identity_failure or "compatible",
                   test_retain_t=test_retain[0], test_retain_qt=test_retain[1], test_retain_u=test_retain[2], test_retain_qu=test_retain[3],
                   test_retain_delta_u=difference(test_retain[2], test_retain[0]) if not identity_failure else None,
                   test_retain_delta_qu=difference(test_retain[3], test_retain[1]) if not identity_failure else None)
        rows.append(row)
        comparisons = [("selected achieved NLL", entry["achieved_nll"], mean(selected_val, "forget_validation")),
                       ("matching error", entry["signed_error"], difference(entry["achieved_nll"], selection["decision"]["target_nll"])),
                       ("validation gain", entry["gain_over_original"], difference(entry["achieved_nll"], selection["original_validation_nll"]))]
        if retained:
            comparisons += [("unquantized gap", retained["unquantized_gap"], gap), ("quantized gap", retained["quantized_gap"], qgap)]
            comparisons += [(f"raw forget {label}", value, mean(record, "forget_test"))
                            for label, value, record in zip(("T", "QT", "U", "QU"), raw, quartet)]
            if retained["status"] == "complete" and denominator_valid and qgap is not None:
                comparisons.append(("unclamped retention", retained["retention_ratio"], qgap / gap))
                comparisons += [("retain test change U", retained["retain_unquantized_delta"], row["test_retain_delta_u"]),
                                ("retain test change QU", retained["retain_quantized_delta"], row["test_retain_delta_qu"])]
            if retained["status"] == "complete" and (not denominator_valid or identity_failure):
                checks.append(f"{run_id}: recorded complete retention conflicts with denominator or identity checks")
        for label, stored, derived in comparisons:
            if stored is not None and derived is not None and not math.isclose(stored, derived, rel_tol=1e-9, abs_tol=1e-10):
                checks.append(f"{run_id}: {label} recorded {stored!r} versus derived {derived!r}")
        provenance.append(dict(run_id=run_id, original_model=run["original_model"], checkpoint=checkpoint,
                               predictor=prediction, test_evaluations=quartet))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = inside(root, args.output or f"postprocessing/refinement-report-{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "overview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = [] if args.no_plots else plot(rows, output)
    decision = selection["decision"]
    valid_count = sum(row["retention_status"] == "complete" and row["retention_ratio"] is not None for row in rows)
    state = "Available under the saved tolerance" if selection["status"] == "complete" else "Unavailable"
    lines = [f"# Refinement result: {state.lower()}", "",
             f"Frozen selection status: **{selection['status']}**. Final analysis status: **{analysis['status']}**. "
             f"Recorded valid retention results: {valid_count} of {len(rows)} runs.", "",
             f"The accepted maximum absolute validation mismatch was **{decision['max_abs_error']:g} nats**. "
             "A tolerance of 10 nats is broad: acceptance under this rule does not substantiate close matching. "
             "Actual errors, gains, and selected steps are shown below. Differences in achieved forgetting can confound the predictor comparison.", "",
             f"The frozen target was {display(decision['target_nll'])} nats; the minimum gain over T was {display(decision['minimum_gain'])} nats. "
             f"Original forget-validation NLL was {display(selection['original_validation_nll'])} nats. "
             f"Recorded common-range bounds: {selection['common_range']}. "
             "If the lower bound exceeds the upper bound, the intersection is empty; those bounds are not a feasible interval.", ""]
    if selection.get("failure_reason") or analysis.get("failure_reason"):
        lines.extend(["Reason: " + str(selection.get("failure_reason") or analysis.get("failure_reason")), ""])
    table(lines, ["LR", "Seed", "E(LR)", "Step", "Achieved validation NLL", "Gain over T", "Signed error", "Run status"],
          [[r[key] for key in ("learning_rate", "seed", "predictor_e", "selected_step", "achieved_validation_nll", "gain_over_original", "signed_matching_error", "run_status")] for r in rows])
    lines += ["Errors are achieved minus target NLL; all NLLs and NLL changes use nats. CSV values retain full numerical precision. "
              "If global selection is unavailable, any populated checkpoint fields are recorded candidates, not a completed matched selection.", "",
              "## Raw blind-test forgetting and retention", "",
              "Retention is (NLL(Q(U)) − NLL(Q(T))) / (NLL(U) − NLL(T)) on the same forget-test data. "
              f"The existing analysis requires a finite denominator greater than {DENOMINATOR_CUTOFF:g}; nonpositive, tiny, or nonfinite effects are invalid. "
              "Ratios remain unclamped, including negative values and values above one. Unmeasured values are not zero.", ""]
    table(lines, ["LR", "Seed", "T", "Q(T)", "U", "Q(U)", "U−T", "Q(U)−Q(T)", "Retention", "Denominator", "Record status"],
          [[r[key] for key in ("learning_rate", "seed", "test_forget_t", "test_forget_qt", "test_forget_u", "test_forget_qu", "unquantized_forget_gap", "quantized_forget_gap", "retention_ratio", "denominator_validity", "retention_status")] for r in rows])
    for row in rows:
        if row["retention_status"] != "complete" or row["test_identity_check"] != "compatible":
            lines.append(f"- LR {row['learning_rate']:g}, seed {row['seed']}: {row['retention_failure']}; identity check: {row['test_identity_check']}.")
    lines += ["", "## Utility: validation and blind tests are separate", "",
              "Retain-validation NLL is a utility proxy used here to describe the selected checkpoint. "
              "Blind retain-test NLL is a separate final measurement. Positive changes indicate worse prediction; each quantized change uses Q(T) as its baseline. "
              "Neither split should be described as proving comprehensive retained capability.", ""]
    table(lines, ["LR", "Seed", "Validation T", "Validation U", "Validation U−T", "Test T", "Test Q(T)", "Test U", "Test Q(U)", "Test U−T", "Test Q(U)−Q(T)"],
          [[r[key] for key in ("learning_rate", "seed", "validation_retain_t", "validation_retain_u", "validation_retain_delta", "test_retain_t", "test_retain_qt", "test_retain_u", "test_retain_qu", "test_retain_delta_u", "test_retain_delta_qu")] for r in rows])
    lines += ["## Training trajectories", "", "Ranges below cover saved checkpoints, excluding original step zero. Final checkpoints do not replace the frozen selections.", ""]
    table(lines, ["LR", "Seed", "Saved forget minimum", "Saved forget maximum", "First step", "First forget NLL", "Final forget NLL", "Final retain NLL", "Final step", "Saved checkpoints"],
          [[r[key] for key in ("learning_rate", "seed", "checkpoint_forget_min", "checkpoint_forget_max", "first_step", "first_forget_validation", "final_forget_validation", "final_retain_validation", "last_step", "checkpoint_count")] for r in rows])
    summary = analysis["summary"]
    lines += ["## Exploratory predictor comparison", "",
              f"The saved analysis reports {summary.get('comparable_learning_rate_count', 0)} comparable learning-rate settings. "
              "The experiment uses one seed per LR, so seed variability cannot be estimated. Correlations below reproduce the saved LR-level seed-mean analysis; "
              "seeds are not counted as additional LR settings. Undefined or unavailable correlations are labeled explicitly.", ""]
    table(lines, ["Predictor", "Pearson with retention", "Spearman with retention"],
          [["E(LR)", *[summary.get(key) if summary.get(key) is not None else "undefined/unavailable"
                       for key in ("pearson_e_retention", "spearman_e_retention")]],
           ["log10(LR)", *[summary.get(key) if summary.get(key) is not None else "undefined/unavailable"
                           for key in ("pearson_log_lr_retention", "spearman_log_lr_retention")]]])
    lines += ["E(LR) is determined by LR and the original weights. These correlations do not show added predictive value beyond LR, establish causality, "
              "or overcome the broad validation mismatch allowance. Inspect raw test effect sizes, actual matching errors, and utility losses together. "
              "If selection is unavailable, the planned matched quantization comparison remains untested; missing retention entries describe unmeasured outcomes.", "",
              "## Provenance and record checks", "",
              f"- Selection: `{selection_path.relative_to(root).as_posix()}`.",
              f"- Final analysis: `{analysis_path.relative_to(root).as_posix()}`; sweep snapshot: `{index_path.relative_to(root).as_posix()}`.",
              f"- Model `{config['model']['source']}` at `{config['model']['revision']}`; tokenizer `{config['model']['tokenizer']}` at `{config['model']['tokenizer_revision']}`.",
              f"- Data `{config['data'].get('source', 'see dataset manifests')}` at `{config['data'].get('revision', 'see dataset manifests')}`; split memberships and preprocessing remain in the dataset manifests.",
              f"- Configured storage `{config['training']['parameter_dtype']}`, optimizer moments `{config['training']['optimizer_state_dtype']}`, "
              f"FP32 master weights `{config['training']['fp32_master_weights']}`; quantizer `{config['quantization']}`. Actual variant precision, software versions, data IDs and token counts are reproduced in `provenance.json`.",
              "- E(LR) measures original linear weights unchanged in both perturbation directions after FP32 arithmetic and BF16 rounding; it is not an exact AdamW update guarantee.",
              "- Arithmetic consistency issues: " + (str(len(checks)) + "; see provenance.json." if checks else "none found among available recorded values."),
              "- `overview.csv` preserves every expected run, raw values, failure reasons, denominator validity, and separate validation/test utility costs."]
    logs = set((root / "runtime").glob("*test*.log"))
    setup_log = root / "runtime/setup.log"
    if setup_log.exists():
        logs.add(setup_log)
    for log in sorted(logs):
        matches = re.findall(r"\d+ passed[^\n]*", log.read_text(encoding="utf-8", errors="replace"))
        if matches:
            lines.append(f"- Existing test evidence `{log.relative_to(root).as_posix()}`: {matches[-1]}. No tests were run by this report helper.")
    if figures:
        lines += ["", "![Actual mismatch, raw forgetting gaps, and separate utility changes](raw_effects_and_utility.png)", "",
                  "Figure values are existing measurements; lines connect the same seed across LRs and do not imply additional samples."]
    lines += ["", "This additive report changes no selection, scientific source, config, immutable record, or generated summary. "
              "It performs record aggregation only, with no model loading, training, quantization, evaluation, or cost/teardown claims."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance_document = dict(created_utc=stamp, selection=analysis["selection_path"],
                               analysis=analysis_path.relative_to(root).as_posix(), sweep=index_path.relative_to(root).as_posix(),
                               evaluation_record_paths=evaluations_paths, checks=checks, runs=provenance,
                               denominator_cutoff=DENOMINATOR_CUTOFF, figures=figures)
    (output / "provenance.json").write_text(json.dumps(provenance_document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
