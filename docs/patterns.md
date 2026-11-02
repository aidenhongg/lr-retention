
| Pattern | Rule |
|---|---|
| Independent experiment choices | Support supplied and curve-chosen NLL targets. Both require measurements. Keep target choice, training budget, evaluation interval, and execution mode separate. Targets select checkpoints; every run uses its configured budget. Freeze selection before inspecting quantized or blind-test results, without using predictor agreement. |
| Fixed comparison identities | Pin weights, tokenizer, data membership, preprocessing, precision, quantization/calibration settings, and relevant code versions. Reuse only compatible measurements. Targets and T/Q(T) baselines stay within the same model, tokenizer, and data setup; token NLL is not interchangeable across tokenizers. |
| Artifact references | Exchange small typed records referencing files, not loaded models or tensors. Use artifact-relative paths. Verify complete checkpoint bundles before publishing jobs. Publish ready/result records atomically on the receiving filesystem and ignore temporary files. |
| Same work in both execution modes | Local calls and remote workers use the same requests and results. Training owns checkpoints, coordination owns jobs/retries, and evaluation owns results. Attach delayed measurements to the evaluated checkpoint, not the current model. |
| Small adapters | Use named functions for different model, data, or backend behavior. Start with the causal-LM contract; other tasks need explicit adapters. Validate capabilities before expensive work and reject unsupported combinations. Avoid a plugin framework or a class/module for every option. |
| Normalize datasets once | Map source fields and document boundaries into common samples, then create shared input IDs, attention masks, and labels. Freeze splits, tokenizer/templates, chunking, and masks. Keep validation/test documents out of unlearning training and blind tests out of calibration. |
| Backend-specific preparation | Hide backend preparation behind a common evaluation contract. Distinguish quantizing a new checkpoint from loading an existing quantized one. Record backend/version/settings and calibration identity. Some methods quantize during loading; others require calibration and a saved variant. [Bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [GPTQ](https://huggingface.co/docs/transformers/quantization/gptq). |
| Public entry points | Expose main operations and keep helpers private. Pass named settings and relevant records, not the whole sweep state. Reject ambiguous or unrelated options. Separate numerical calculations from file handling and worker loops; add abstractions only when needed. |
| Explicit precision roles | Separate parameter storage, optimizer/master weights, autocast, inference dtype, and quantized compute dtype. Bits/format are separate from floating precision. Apply and verify effective settings; mixed precision alone does not define stored weights. [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html). |
| Schedules with units | Use explicit epoch or optimizer-step units and actual trainer progress, including accumulation and partial batches. Lengthless streams need a step budget. Round absolute boundaries, deduplicate steps, and include the final checkpoint. With three updates/epoch, half-epoch checkpoints are 2, 3, 5, 6. Translate backend units: fractional Transformers `eval_steps` refers to total training steps. [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). |
| Token-weighted measurements | Aggregate NLL sums and valid target-token counts before division; do not average batch means. Apply the next-token shift once and use attention/label masks, preserving real EOS tokens when EOS also pads. Reject empty/nonfinite measurements. Report achieved NLL, signed matching error, and unavailable matches. |
| Numerical probes | Calculate predictor probes in a wider working dtype, then round once to storage precision. Compare against the same storage representation without modifying the original weights. |
| Explicit outcomes | Report completed, failed, missing, unsupported, and unmatched work explicitly. Keep logical job IDs stable and attempt records separate. Count one compatible, complete success per job. Require all expected runs/jobs and requested splits; empty queues or partial results are not completion. Report invalid retention denominators and raw gaps without inventing or clamping ratios. |

**Repository**

- Fixed comparison identities
- Artifact references
- Public entry points
- Explicit outcomes

Patterns apply to children. Component names follow [design.md](design.md).

## **experiments**

- Independent experiment choices
- Same work in both execution modes

### • sweep

- Public entry points
- Schedules with units

- `load_config` — Independent experiment choices; Small adapters; Explicit precision roles; Schedules with units
- `run_sweep` — Fixed comparison identities; Normalize datasets once; Same work in both execution modes; Explicit outcomes
- `finalize_sweep` — Independent experiment choices; Fixed comparison identities; Explicit outcomes

## **training**

- Independent experiment choices
- Explicit precision roles

### • unlearn

- Small adapters
- Schedules with units

- `train_run` — Explicit precision roles; Schedules with units; Artifact references; Explicit outcomes

## **eval**

- Small adapters
- Backend-specific preparation
- Same work in both execution modes

### • evaluator

- Public entry points
- Fixed comparison identities

- `load_model` — Small adapters; Backend-specific preparation; Explicit precision roles
- `measure_nll` — Token-weighted measurements
- `evaluate` — Fixed comparison identities; Token-weighted measurements; Explicit outcomes
- `run_worker` — Artifact references; Same work in both execution modes; Explicit outcomes

## **analysis**

- Independent experiment choices
- Fixed comparison identities

### • selection

- Public entry points
- Token-weighted measurements

- `select_checkpoints` — Independent experiment choices; Token-weighted measurements; Explicit outcomes

### • metrics

- Public entry points
- Fixed comparison identities

- `compute_predictor` — Numerical probes; Explicit precision roles
- `measure_retention` — Token-weighted measurements; Fixed comparison identities; Explicit outcomes
- `assess_predictor` — Fixed comparison identities; Explicit outcomes

## Shared modules

- Artifact references
- Fixed comparison identities

### • inputs

- Normalize datasets once
- Small adapters
- Public entry points

- `prepare_data` — Normalize datasets once; Fixed comparison identities
- `load_data` — Normalize datasets once; Explicit outcomes

### • artifacts

- Artifact references
- Public entry points
- Explicit outcomes

- `publish_record` — Artifact references; Explicit outcomes
- `read_record` — Public entry points; Explicit outcomes
