# Learning-rate retention under FP4 quantization

This experiment tests whether the original-weight rounding probe E(lr) predicts how much unlearning survives FP4 quantization. The completed refinement uses the pinned MUSE-News Llama-2-7B target, learning rates [1e-5, 2e-5, 3e-5, 4e-5, 5e-5], one seed and three epochs per run. All five training runs, 105 validation checkpoints and 12 final evaluations completed.

Start with the [completed run and archive summary](results/muse_news_bf16_lr_refinement_20261102T105346Z/README.md), then the [interpretation](results/muse_news_bf16_lr_refinement_20261102T105346Z/postprocessing/final-report/interpretation.md), [full report](results/muse_news_bf16_lr_refinement_20261102T105346Z/postprocessing/final-report/report.md) and [numerical overview](results/muse_news_bf16_lr_refinement_20261102T105346Z/postprocessing/final-report/overview.csv). Figures are in [graphs](graphs/).

Lower E(lr) was associated with greater forgetting retention (Pearson -0.714), while log learning rate had a larger absolute correlation (+0.870). These descriptive correlations do not establish predictive value beyond learning rate. E had only three distinct values, matching errors reached +/-0.732 nats, and every selected unquantized model suffered severe retain-performance damage. This is preliminary evidence about quantization survival, not a demonstration of useful selective unlearning.

## Data and evidence

The timestamped [results directory](results/muse_news_bf16_lr_refinement_20261102T105346Z/) contains all six frozen tokenized dataset splits and their document-membership manifests under `data/`, immutable training and evaluation records under `records/`, reports and figures, the source snapshot, environment freeze, runtime logs and verified archive receipts. The raw upstream corpus and original model are identified by pinned revisions in [experiment.yaml](configs/experiment.yaml); the RunPod pipeline materializes the source documents before tokenization. The top-level `data/` paths are runtime inputs rather than the saved dataset snapshots. The earlier pilot is preserved separately in the original research workspace and is outside this repository.

Model weights are stored separately on RunPod. Five selected and five final BF16 checkpoints are preserved on volume `vh9mvmjd8a` in EUR-IS-1 at `/workspace/muse_news_bf16_lr_refinement_20261102T105346Z`. The [archive receipt](results/muse_news_bf16_lr_refinement_20261102T105346Z/archive/5197ff1b2566c586480e3fb3a62649cb13b7adab50ad9742ac82be84227bd8ac.json) records their locations and SHA-256 verification. Weights, credentials, caches and loose pod bookkeeping are excluded from Git. The experiment GPU has been terminated; persistent storage remains allocated.

## Running the pipeline

The recipe is in `configs/experiment.yaml`. Source documents are partitioned deterministically before tokenization. Validation selects checkpoints; test and quantized measurements begin only after the selection is published. An unavailable match produces curves and an explicit report instead of dropping a learning rate.

All model computation and tests run on RunPod. On the GPU pod, install and validate with `bash runpod/setup.sh`, then execute `bash runpod/run.sh`. The scripts use `/opt/mock-experiment` for source, `/opt/pilot-env` for dependencies, and `/artifacts` for experiment artifacts. The mounted network volume at `/workspace` holds the verified final archive.

Run `python runpod/pod.py guard --state runpod/pilot-main-state.json --poll 30` on the orchestration machine while the pod exists. The guard includes GPU and storage costs, reserves shutdown credit, and checks accrued account spending. The experiment ceiling is $15 including setup; a replacement pod receives only the remaining allowance. Other account workloads can reduce available credit independently.

Records and failed outcomes are immutable JSON files. Training and evaluation errors include tracebacks. Results contain validation curves, frozen selection, raw NLL measurements, predictor records, CSV tables, PNG/SVG figures, and a summary. Small results and logs are copied locally.

Use a fresh artifact root when changing scientific implementation or dependency versions. Keep the source snapshot and exact environment freeze with the results.

After analysis, use `python runpod/archive.py copy --root /artifacts --destination /workspace/NEW_RUN_NAME` on the pod, replacing the destination with a fresh run name. Verify its manifest before terminating the GPU. The archive retains selected and final weights and the experiment evidence; unselected weights remain disposable. Network-volume storage continues to be billed after GPU teardown.
