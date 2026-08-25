# Method

## 1. Problem formulation

For patient \(i\), let \(X_i^{[a,b]}\) denote the prenatal records visible in an arbitrary day range \([a,b]\), where days are measured from the last menstrual period. The model predicts:

\[
y_i = (d_i, w_i, l_i),
\]

where \(d_i\) is delivery day, \(w_i\) is birth weight in grams, and \(l_i\) is birth length in centimeters.

The input is intentionally not restricted to five canonical landmarks. It may be a cumulative prefix, a local window, or a partially observed prefix after content/visit removal.

## 2. Why CE alone is insufficient

Autoregressive SFT represents a number as a sequence of discrete tokens. A prediction of 274 and a prediction of 204 are both simply different token sequences; ordinary token CE does not explicitly encode that the former is numerically closer to a target of 275.

Continuous-v2 therefore keeps teacher-forced XML generation as an auxiliary language objective and adds three scalar regression heads. This preserves the existing VLM/SFT pipeline while giving the output space a continuous geometry.

## 3. Outcome query tokens

Three private special tokens are appended to the user prompt during training and numeric-head inference:

- `<|delivery_outcome_query|>`
- `<|birth_weight_outcome_query|>`
- `<|birth_length_outcome_query|>`

The plugin replaces their ordinary token embeddings with three trainable query embeddings. A forward hook captures the final normalized language hidden states, locates each query token exactly once, and sends the corresponding state through an independent head:

\[
\hat z_t = h_t^\top w_t + b_t.
\]

The released configuration uses LayerNorm followed by a linear scalar layer. The plugin also supports an MLP variant, but it is not the final experiment.

The query tokens occur before the assistant XML. Under causal attention, their hidden states cannot attend to the future ground-truth answer tokens. The assistant XML is used only by teacher forcing and CE.

## 4. Standardized numeric objective

Targets are standardized using fixed centers and scales:

| target | center | scale |
|---|---:|---:|
| delivery day | 275 d | 10 d |
| birth weight | 3250 g | 500 g |
| birth length | 50 cm | 2 cm |

\[
z_t=(y_t-\mu_t)/s_t.
\]

The regression loss is the average masked SmoothL1 loss over active tasks:

\[
\mathcal{L}_{reg} =
\frac{1}{|\mathcal{T}_{valid}|}
\sum_{t \in \mathcal{T}_{valid}}
\operatorname{SmoothL1}(\hat z_t,z_t;\beta=0.5).
\]

The final training objective is:

\[
\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{reg}.
\]

Invalid optional birth-weight or birth-length labels are represented by `NA` in the XML and zero in `outcome_mask`. Delivery day is required because it defines visibility and leakage boundaries.

## 5. LoRA integration

The VLM backbone is adapted with LoRA on all linear layers. The final configuration:

| item | value |
|---|---|
| base | Qwen3.5-9B |
| LoRA rank / alpha | 128 / 256 |
| vision encoder | frozen |
| aligner | frozen |
| outcome module | saved through `modules_to_save` |
| dtype | bfloat16 |
| attention | SDPA |
| optimizer schedule | cosine, 3% warmup |
| training launcher | ms-swift SFT + ZeRO-2 |

The plugin patches the Qwen3.5 template pre-hook so that custom regression targets, masks, and the original input IDs survive multimodal preprocessing.

## 6. Continuous temporal sampling

Five bins control coverage, not fixed views:

| bin | inclusive range | density anchor |
|---|---:|---:|
| d070_111 | 70–111 | 97 |
| d112_181 | 112–181 | 160 |
| d182_215 | 182–215 | 202 |
| d216_244 | 216–244 | 230 |
| d245_258 | 245–258 | 258 |

Within an eligible bin, 35% of cutoffs are sampled from a triangular distribution whose mode is the anchor; 65% are uniformly sampled. Every cutoff is strictly earlier than delivery.

The 80,000-draw schedule is balanced 1:1 by cohort, evenly across bins within condition, and uses four mutually exclusive conditions:

- 60% clean cumulative prefix
- 20% field-level content masking
- 10% whole-visit dropout
- 10% local window

Each source case receives at least one clean view before weighted oversampling. Outcome-tail weights are derived marginally across the three targets and capped at 3×.

## 7. Checkpoint selection

Token/regression validation loss is not the final temporal selector. `scripts/select_checkpoint.py` ranks checkpoints by:

1. fewest material population-level MAE reversals across the five cutoff bins;
2. lowest normalized upward-violation sum;
3. lowest mean normalized MAE.

Individual-patient strict monotonicity is reported but is not optimized. This avoids forcing an unrealistic pointwise constraint while retaining the desired group trend.
