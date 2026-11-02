# MUSE-News learning-rate refinement

All five learning rates completed three epochs with seed 0, producing 105 checkpoints. Frozen selection and all 12 final unquantized/FP4 evaluations completed. Only the configuration grid, checkpoint frequency, run name and matching tolerance differ from the original implementation.

FP4 nearly erased the forgetting effect at LR 1e-5: retention was 0.005956. The other four ratios were 0.885885, 1.031947, 0.996667 and 0.962694 in ascending LR order. Ratios remain unclamped. Lower E correlated with higher retention descriptively (Pearson -0.713988), while log learning rate had a stronger absolute correlation (+0.870424). This single-seed comparison does not establish added predictive value beyond learning rate.

Every selected unquantized model suffered severe retain-test degradation: NLL increased by 9.716261–11.196978 nats. The results therefore do not demonstrate useful selective unlearning. Matching passed the configured 10-nat tolerance, with actual signed errors between -0.731258 and +0.731258 nats; describe this as approximate matching under that rule.

Read the [interpretation and independent audit](postprocessing/final-report/interpretation.md), [full report](postprocessing/final-report/report.md), and [numerical overview](postprocessing/final-report/overview.csv). Figures show [raw effects and utility costs](../../graphs/refinement_effects_and_utility.png), [predictor comparisons](../../graphs/refinement_predictor_assessment.png), and [validation trajectories](../../graphs/refinement_validation_panels.png). PNG and SVG versions are preserved. The figures were visually inspected.

All numerical work and tests ran on RunPod. Setup passed 128 tests; the independent record audit and report arithmetic checks found no inconsistencies. Source, pinned data membership, environment freeze, logs, frozen records and provenance are included here.

The refinement GPU is terminated and its absence was confirmed through the API. The conservative estimate for this follow-up is **$8.64**, below its $15 run cap, excluding the prior pilot, unrelated workloads and future storage. This is a duration-and-rate estimate rather than an invoice. See [costs](cloud-operations/costs.json) and [teardown verification](cloud-operations/teardown-verification.json).

All five selected and five final BF16 checkpoints remain on volume `vh9mvmjd8a` in EUR-IS-1 at `/workspace/muse_news_bf16_lr_refinement_20261102T105346Z`. Its 150 GB costs **$10.50/month**. The previous 80 GB volume `27k6qtrwsc` remains preserved at **$5.60/month**; combined ongoing storage is **$16.10/month** at the current [standard storage rate](https://www.runpod.io/pricing).

The [immutable archive receipt](archive/5197ff1b2566c586480e3fb3a62649cb13b7adab50ad9742ac82be84227bd8ac.json) verifies source and destination SHA-256 for 514 files totaling 134,803,668,375 bytes, with no missing or incomplete runs. The [supplement receipt](postprocessing/final-report/supplement-manifest.json) separately verifies seven additive report files without replacing that manifest. Post-teardown administrative receipts and this navigation page are local additions. No weights were downloaded locally.
