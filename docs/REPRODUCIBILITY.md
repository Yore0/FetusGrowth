# Reproducibility and release audit

## Tested environment

The archived run was evaluated in:

```text
Python       3.12
ms-swift     4.4.0.dev0
torch        2.11.0
transformers 5.12.1
peft         0.19.1
numpy        2.3.5
pandas       2.3.3
matplotlib   3.11.0
scipy        1.18.0
scikit-learn 1.9.0
```

Qwen3.5 support and ms-swift internal template APIs may vary across releases. For exact reproduction, record the ms-swift commit, base-model revision, adapter revision, CUDA driver, and GPU model.

## Determinism

- data seed: `20260809`;
- benchmark seed: `20260810`;
- validation cutoffs: deterministic hash by cohort/case/bin;
- training draws: deterministic schedule, dataloader shuffled with the same data seed;
- paired bootstrap seeds are stored in analysis outputs.

Distributed kernels may still introduce small floating-point differences.

## Pre-training checks

```bash
python -m compileall -q src scripts plugins
pytest -q
python scripts/build_dataset.py --source-root /private/source-data
bash scripts/smoke_train.sh
```

Inspect `logs/build_continuous_v2_summary.json` and verify:

- all training cases have a clean-prefix view;
- cohort and condition quotas match the protocol;
- all validation cutoffs are non-anchor and before delivery;
- no forbidden hint section survives;
- cleaning counters are plausible for each source.

## Pre-publication checks

```bash
git status --short
git ls-files | grep -E '\\.(jsonl|safetensors|bin|pt|pth|ckpt)$' && exit 1 || true
git grep -nE '/root/|api[_-]?key|secret|password|token=' -- ':!README.md'
find . -type f -size +10M -print
```

Review every match manually. Query-token names are expected; credentials, absolute workstation paths, clinical text, and large binary artifacts are not.

## Checkpoint contract

A usable Hugging Face weight release needs:

- LoRA adapter configuration and tensors;
- the saved `outcome_regression` module;
- tokenizer files containing the three added special tokens;
- base-model name and immutable revision;
- the loss/head environment used during training;
- a small, synthetic inference smoke example.

The token IDs must match the plugin's expected IDs for the stated tokenizer. The inference script fails early if they differ.
