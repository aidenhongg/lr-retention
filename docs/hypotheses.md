
## Main hypothesis

We expect runs with lower E(lr) to preserve more of their forgetting after quantization when their checkpoints have similar forget NLL before quantization. This prediction applies within a fixed model, dataset, training precision, and quantization method.

E(lr) measures the share of original linear weights that remain unchanged after both adding and subtracting the learning rate and rounding each result to the training precision. A lower value means that fewer weights resist these changes through rounding.

## How we test it

We will sweep learning rates and seeds while keeping the other settings fixed. Each run will use the configured training budget, and we will select checkpoints with similar forget-validation NLL using either a supplied target or one chosen from the validation curves. We will freeze this selection before looking at quantized or blind-test results.

Our primary metric will be the share of the unquantized forget-NLL increase that survives quantization, measured against the matching original-model baselines. Higher retention means that more of the forgetting survives. We will evaluate the selected checkpoints on the fixed tests and report their achieved test forget NLL alongside checkpoint matching errors and, when multiple seeds are configured, variation across seeds. We will also report retain NLL to show any loss of performance on retained data.

## How we interpret the results

Because E(lr) is calculated from the learning rate and the original weights, we will compare it with learning rate itself as a predictor of retention. A correlation between E(lr) and retention alone would not show that the predictor adds value beyond learning rate.

A single-seed pilot provides a preliminary trend and tests whether checkpoints can be matched. It cannot establish reproducibility or estimate variation across seeds.

The hypothesis would be weakened if lower E(lr) does not consistently correspond to higher retention, or if the relationship reverses, after accounting for seed variation and differences in achieved forgetting. If the runs cannot be matched at a common forgetting level, that limits the comparison rather than establishing or refuting the hypothesis.
