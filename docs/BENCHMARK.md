# Model-agnostic robustness benchmark

## Principle

Every model receives the same clinical input stored in `test.jsonl`. Benchmark files contain one user message only and exclude:

- assistant labels;
- `outcome_targets` and `outcome_mask`;
- the three private outcome query tokens;
- future-derived hint or summary sections.

Ground truth lives in a separate `labels/outcomes.jsonl`, joined by `pair_id` only during scoring. Ours injects query tokens in memory; generation baselines use the unchanged user message.

## Splits

- **core:** one deterministic, balanced, non-anchor continuous cutoff per valid case;
- **full:** one deterministic non-anchor cutoff per eligible case and cutoff bin.

Core is an exact subset of full. Each perturbation is paired with its clean parent through `pair_id` and `clean_reference_id`.

## Variants

| family | variants |
|---|---|
| clean | `clean_continuous_prefix` |
| content mask | 15%, 30%, 50% |
| visit dropout | 20%, 40%, latest visit |
| modality drop | ultrasound, lab |
| local window | 28, 56, 84 days |
| compound | deterministic realistic combination of at least two perturbations |

The audit checks exact masking counts, visit-removal semantics, modality absence, window widths, unique IDs, subset identity, non-anchor cutoffs, label separation, and runtime-only query injection.

## Metrics

Clean performance:

- MAE, RMSE, bias, p90 absolute error;
- Pearson correlation;
- parse/coverage rate for generation models.

Paired robustness for variant \(v\):

\[
\Delta AE_{i,t,v} =
|\hat y_{i,t,v} - y_{i,t}| -
|\hat y_{i,t,clean} - y_{i,t}|.
\]

Prediction responsiveness:

\[
Drift_{i,t,v} =
|\hat y_{i,t,v} - \hat y_{i,t,clean}|.
\]

The normalized composite averages target errors after division by 10 days, 500 g, and 2 cm. Confidence intervals are paired bootstrap intervals.

## Interpretation

Low degradation alone is not sufficient evidence of robustness. A constant predictor may be perfectly invariant. Interpret robustness jointly with:

- clean accuracy;
- target correlation;
- output diversity/mode share;
- non-zero but bounded prediction drift;
- improvement across later cutoffs.

## Commands

```bash
python scripts/build_benchmark.py --source-root /private/source-data
python scripts/audit_benchmark.py \
  --root benchmarks/model_agnostic_robustness_v1

torchrun --nproc_per_node 4 scripts/run_inference.py \
  --model ours \
  --split full \
  --ours-base /path/to/base \
  --ours-checkpoint /path/to/adapter

python scripts/analyze_predictions.py
```
