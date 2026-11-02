# Background and experiment summary

## The problem

Machine unlearning tries to reduce a model's ability to reproduce selected training data while preserving its performance on other data. Quantization serves a different purpose: it stores model weights with fewer bits so the model takes less memory to deploy. Our experiment asks how these two processes interact.

In [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454), Zhang et al. found that quantizing an unlearned model could restore information that it appeared to have forgotten. Their explanation was that unlearning makes small weight changes, which later quantization can round away. Their experiments draw on [MUSE](https://arxiv.org/abs/2407.06460), a benchmark that evaluates forgetting alongside privacy and performance on retained data.

## Why training precision matters

Rounding can also affect the updates before a model is quantized. When weights are stored and updated directly in bf16, a small change can round back to the original value. This creates two places where changes may be lost: during training and during later quantization.

This numerical problem has a long history. [Gupta et al. (2015)](https://arxiv.org/abs/1502.02551) showed how stochastic rounding could support training with limited precision. [Mixed Precision Training](https://arxiv.org/abs/1710.03740), published at ICLR 2018, described keeping an fp32 copy of the weights to accumulate updates while using lower precision for computation. [Zamirai et al. (2020)](https://arxiv.org/abs/2010.06192) studied small updates being cancelled in bf16 training and examined stochastic rounding and Kahan summation as remedies.

More recently, [Singh (2026)](https://arxiv.org/abs/2610.31859) examined lost bf16 updates when fine-tuning vision-language-action models. The paper describes E(u), a weight-based estimate of how many weights cannot change under a perturbation of size u. We use the learning rate as that probe size to compute E(lr). Singh provides the numerical motivation for our predictor, while Zhang provides the unlearning and quantization setup. Whether this probe predicts how much forgetting survives quantization is the question we still need to test.

## Earlier project work

The [project explainer](../../judgments/EXPLAINER-I-183.md) brings together earlier pilots on training precision and quantization. Those runs suggested that stochastic rounding or a larger learning rate could help the unlearning effect survive some quantizers. However, the [larger-model pilot](../../judgments/MUSE-I-183.md) used one seed and shortened training, with imperfectly matched forgetting, and the [learning-rate control](../../judgments/EXT-I-183.md) used one seed. These observations motivate a more systematic sweep rather than establish the new hypothesis.

## Our experiment

Our starting recipe uses the [MUSE-News target model](https://huggingface.co/muse-bench/MUSE-news_target), based on Llama-2-7B, and the GA_GDR method from [Zhang's implementation](https://github.com/zzwjames/FailureLLMUnlearning). This method combines gradient ascent on forget data with gradient descent on retain data. The [MUSE implementation](https://github.com/swj0419/muse_bench) supplies the benchmark context and links to its data and target models. The initial configuration uses bf16 training and bitsandbytes FP4 quantization; the concrete settings live in [experiment.yaml](../configs/experiment.yaml).

We will sweep learning rates and seeds while keeping the other settings fixed. Every run will train for its configured budget and save checkpoints for validation. We will then select checkpoints with similar forget-validation NLL, using either a supplied target or a target chosen from the observed curves. The target selects checkpoints rather than stopping training, and selection will be frozen before we inspect quantized or blind-test results.

NLL measures how little probability a model assigns to the correct next tokens. A higher forget NLL indicates worse prediction of the forget text, while retain NLL lets us track the cost on retained data. We will evaluate each selected model before and after quantization and compare it with the matching original-model baselines. Our primary metric is the fraction of the unlearning-induced forget-NLL increase that remains after quantization. We call this retention, so a higher value means more of the unlearning effect survives.

The [main hypothesis](hypotheses.md) is that lower E(lr) will correspond to higher retention at similar forgetting levels. We will report achieved test forgetting, matching errors, retain NLL, and variation across seeds when multiple seeds are configured, and compare E(lr) with learning rate itself. A single-seed pilot provides a preliminary trend without establishing reproducibility. This is a focused NLL experiment rather than a reproduction of every MUSE metric or a claim that NLL alone establishes complete unlearning.

The planned implementation uses [PyTorch](https://github.com/pytorch/pytorch) for training, [Transformers](https://github.com/huggingface/transformers) for model loading and training APIs, and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for the initial quantization backend. The package responsibilities and interfaces are described in [design.md](design.md).
