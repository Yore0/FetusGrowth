# Data contract and leakage controls

## Expected private source tree

```text
SOURCE_ROOT/
├── huaxi/
│   ├── huaxi_train.jsonl
│   └── huaxi_test.jsonl
└── shenzhen/
    ├── shenzhen_train_all__full.jsonl
    └── shenzhen_internal_test_all__full.jsonl
```

Each source row must contain:

- `source_sample_id` or `sample_id`;
- one user message and one assistant message in `messages`;
- a user prompt containing `[孕妇基本信息]` and `[全周期检查记录（按时间排序）]`;
- dated timeline blocks with a parseable `DaysFromLMP=...`;
- assistant XML with `delivery_days`, `birth_weight_g`, and `birth_length_cm`.

The repository does not contain a real row example because even a small sample could expose clinical text or identifiers.

## Target validation

| target | accepted range | invalid behavior |
|---|---:|---|
| delivery day | 140–320 d | drop case |
| birth weight | 500–6000 g | set `NA`, mask regression |
| birth length | 20–65 cm | set `NA`, mask regression |

An additional cross-field check masks birth length below 40 cm when delivery is at least day 259 and weight is at least 2000 g.

All numeric parsing rejects booleans, non-finite values, empty strings, and zero values outside the accepted ranges.

## Timeline cleaning

The parser removes:

- negative-day records;
- inequality/ambiguous day expressions that cannot be placed safely;
- undated blocks;
- records at or after delivery;
- future-derived summary or hint sections;
- selected future-derived basic-information fields;
- ultrasound measurements whose inferred unit or value is implausible.

Visibility is then recomputed from cleaned blocks. A generated view must satisfy:

\[
0 \leq window\_start\_day \leq window\_end\_day < delivery\_days.
\]

## Huaxi seen-test exception

The original project requires Huaxi test IDs already present in `huaxi_train.jsonl` to remain in training. The implementation preserves this as an explicit seen-test policy:

1. `huaxi_test.jsonl` is never appended to the training pool;
2. matching IDs already in Huaxi train remain exactly through the train source;
3. internal Huaxi validation IDs are sampled only from non-test IDs;
4. all reports label this evaluation as train-seen, not external generalization.

## Generated training schema

```json
{
  "messages": [
    {"role": "user", "content": "...private query tokens appended..."},
    {
      "role": "assistant",
      "content": "<delivery_days>275</delivery_days>\\n<birth_weight_g>3250</birth_weight_g>\\n<birth_length_cm>50</birth_length_cm>"
    }
  ],
  "outcome_targets": [275.0, 3250.0, 50.0],
  "outcome_mask": [1, 1, 1]
}
```

Metadata is written to a parallel JSONL and includes case/cohort identifiers, window boundaries, visible modality counts, augmentation parameters, and target-mask reasons. It should remain private because linkage metadata can itself be sensitive.

## Safe publication policy

Never commit:

- source or generated JSONL;
- case IDs or row-level metadata;
- free-text prompts;
- patient-level predictions or errors;
- logs that echo data paths or sample content;
- model checkpoints.

The repository `.gitignore` excludes all standard locations and all `*.jsonl` files. Before publishing changes, run the audit commands in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
