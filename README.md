# Learning rate and unlearning retention

Does **E(lr)**, a rounding probe computed from the original weights, predict how much forgetting survives FP4 quantization? This repository contains a completed MUSE-News experiment, its implementation, frozen data, raw measurements and reports.

## Background

Unlearning aims to remove selected information while preserving a model's other capabilities. [Zhang et al.](https://arxiv.org/abs/2410.16454) showed that quantization can restore information a model appeared to have forgotten. Small updates can also disappear earlier, when training writes them into BF16 weights without an FP32 master copy, as studied by [Singh](https://arxiv.org/abs/2610.31859).

E(lr) is the fraction of original linear weights unchanged after **both** adding and subtracting the learning rate in FP32 and rounding back to BF16. We test whether lower E(lr) corresponds to more forgetting surviving quantization. It is a probe of rounding sensitivity, not a measurement of actual AdamW updates.

## Experiment

We use the [MUSE-News](https://arxiv.org/abs/2407.06460) Llama-2-7B target and GA_GDR: gradient ascent on forget data plus gradient descent on retain data. Model, tokenizer and data revisions are pinned in [the configuration](configs/experiment.yaml).

| Setting | Completed run |
| --- | --- |
| Learning rates | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |
| Training | Seed 0; 3 epochs; 1,248 updates per LR |
| Precision | BF16 weights and optimizer moments; no FP32 master copy |
| Checkpoints | Every 0.25 epochs plus early steps; 21 per LR, 105 total |
| Selection | Nearest forget-validation NLL to a shared target, frozen before quantized or test evaluation |
| Quantization | bitsandbytes FP4; 12 final evaluations across the original and five selected models |

The main outcome is the fraction of the forget-test NLL increase that survives quantization:

```text
retention = [NLL(Q(U)) - NLL(Q(T))] / [NLL(U) - NLL(T)]
```

Here T is the original model, U is a selected unlearned model, and Q applies FP4 quantization. Higher retention means more forgetting survives. Ratios are unclamped, so values above one are possible. Retain-set NLL measures the accompanying utility cost separately.

## Latest results

At LR **1e-5**, retention was **0.005956**; at the other four LRs it ranged from **0.885885 to 1.031947**. E(lr) correlated negatively with retention (Pearson **-0.714**, Spearman **-0.527**). Log learning rate had a larger absolute association (Pearson **+0.870**, Spearman **+0.600**). These descriptive correlations do not establish that E adds predictive value beyond LR.

![Forgetting retention versus E and learning rate, with validation matching errors](graphs/refinement_predictor_assessment.png)

The same five outcomes plotted against E(lr) and learning rate. There is one seed per LR and only three distinct E values.

![Validation mismatch, forgetting before and after FP4, and retain-performance damage](graphs/refinement_effects_and_utility.png)

Actual matching errors reached **+/-0.732 nats** under a configured 10-nat tolerance. Every selected unquantized model increased retain-test NLL by **9.716-11.197 nats**. The association therefore describes models with substantial utility damage and does not demonstrate useful selective unlearning.

![Full and early forget/retain validation trajectories with selected checkpoints](graphs/refinement_validation_panels.png)

Stars mark the frozen selected checkpoints; the dashed line marks the shared forget-validation target. Higher NLL means worse prediction on the corresponding data.

Read the [full report](results/muse_news_bf16_lr_refinement_20261102T105346Z/postprocessing/final-report/report.md), [numerical overview](results/muse_news_bf16_lr_refinement_20261102T105346Z/postprocessing/final-report/overview.csv), or [interpretation and audit](results/muse_news_bf16_lr_refinement_20261102T105346Z/postprocessing/final-report/interpretation.md). The cloud setup suite passed **128 tests**.

## Usage

### 1. Inspect the completed experiment

Clone on your local machine with an account that has access to this private repository:

```bash
gh repo clone aidenhongg/lr-retention
cd lr-retention
```

The saved reports and graphs can be inspected without a GPU. The [timestamped results directory](results/muse_news_bf16_lr_refinement_20261102T105346Z/) includes all six frozen tokenized dataset splits and document-membership manifests, raw records, runtime logs, a source snapshot and the environment freeze.

### 2. Prepare RunPod for a new run

Run model computation and tests on a Linux GPU pod with BF16 support, Python 3.11 and CUDA 12.8. The completed sweep used **2 TB of temporary disk**; its ten selected/final checkpoint bundles require about **135 GB**, so allow sufficient free space on a persistent network volume mounted at `/workspace`.

Copy the checkout to `/opt/mock-experiment` on the pod. Give the new experiment a fresh name and output directory in `configs/experiment.yaml`. The pipeline downloads the pinned model and materializes the source documents before tokenization.

For CLI provisioning, inspect `python runpod/pod.py create --help` on your local machine, set `RUNPOD_KEY` in its environment, and pass `--disk-gb 2000`. Supply the persistent volume's ID and allocated size with `--network-volume` and `--network-volume-gb`. The helper saves ownership and budget settings to the path supplied with `--state`. Keep its budget guard running in another local terminal, using that same state path:

```bash
python runpod/pod.py guard --state runpod/new-run-state.json --poll 30
```

The helper defaults to a $15 ceiling and a $1 shutdown reserve. The guard enforces these limits while it is running.

### 3. Install dependencies and validate the pod

Run on the pod:

```bash
cd /opt/mock-experiment
bash runpod/setup.sh
source /opt/pilot-env/bin/activate
```

Setup installs pinned dependencies, checks CUDA/BF16 support and runs the test suite.

### 4. Train, select checkpoints and evaluate

Continue in the same pod shell. Use a fresh artifact directory for every experiment:

```bash
RUN_ID="muse_news_lr_$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACTS="/artifacts/$RUN_ID"
mkdir -p "$ARTIFACTS/runtime"
export MPLBACKEND=Agg
set -o pipefail

python -u -m src.experiments.sweep \
  --action all \
  --config configs/experiment.yaml \
  --root /opt/mock-experiment \
  --artifacts "$ARTIFACTS" \
  2>&1 | tee "$ARTIFACTS/runtime/experiment.log"
```

The pipeline trains each run, evaluates validation checkpoints, freezes selection, then performs test/FP4 evaluation and writes summaries and figures under `$ARTIFACTS/analysis/`. An unavailable matched comparison exits with code 2 and saves the available diagnostics. The log and immutable records preserve failures.

### 5. Archive weights, collect results and shut down

In the same pod shell, plan the archive and copy it to a fresh directory on the mounted network volume:

```bash
python runpod/archive.py plan --root "$ARTIFACTS"
python runpod/archive.py copy \
  --root "$ARTIFACTS" \
  --destination "/workspace/$RUN_ID"
```

Confirm the receipt reports `status: verified` and check its experiment status and missing/incomplete runs. Copy the small artifacts back into `results/$RUN_ID/`, excluding `checkpoints/`; preserve the source snapshot and environment freeze alongside them. After verifying the archive and local collection, terminate the pod. If it was created through the helper, run locally with the same state path:

```bash
python runpod/pod.py terminate --state runpod/new-run-state.json
python runpod/pod.py status --state runpod/new-run-state.json
```

## Stored weights and provenance

The completed run's GPU is terminated. Five selected and five final BF16 checkpoints remain on RunPod volume `vh9mvmjd8a` in EUR-IS-1, under `/workspace/muse_news_bf16_lr_refinement_20261102T105346Z`. Their [archive receipt](results/muse_news_bf16_lr_refinement_20261102T105346Z/archive/5197ff1b2566c586480e3fb3a62649cb13b7adab50ad9742ac82be84227bd8ac.json) verifies source and destination hashes. Persistent storage remains allocated and billed after GPU teardown.

Git contains the completed refinement's evidence and tokenized inputs. Weights and credentials are excluded. The raw upstream corpus is identified by its pinned revision; top-level `data/` paths are generated runtime inputs. Earlier pilot results remain in a separate archive in the original research workspace. See [background](docs/background.md), [hypotheses](docs/hypotheses.md) and [design](docs/design.md) for further detail.
