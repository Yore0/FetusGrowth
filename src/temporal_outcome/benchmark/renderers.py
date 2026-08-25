#!/usr/bin/env python3
"""Runtime-only renderers for the model-agnostic robustness benchmark.

Generic VLMs receive the benchmark messages unchanged. The v2 numeric-head
path adds its three private query tokens in memory and joins labels by pair_id;
no model-specific tokens or ground truth are written into test.jsonl.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterator

QUERY_TOKENS = (
    "<|delivery_outcome_query|>",
    "<|birth_weight_outcome_query|>",
    "<|birth_length_outcome_query|>",
)
QUERY_SUFFIX = "\n\n[内部结局回归查询]\n" + "\n".join(QUERY_TOKENS)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.open(encoding="utf-8"):
        yield json.loads(line)


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pair_id = str(row["pair_id"])
        if pair_id in labels:
            raise ValueError(f"duplicate label pair_id: {pair_id}")
        labels[pair_id] = row
    return labels


def generic_vlm_example(row: dict[str, Any]) -> dict[str, Any]:
    """Return a generation example shared by arbitrary chat/VLM baselines."""
    messages = copy.deepcopy(row["messages"])
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("benchmark rows must contain one user-only message")
    prompt = str(messages[0].get("content") or "")
    if any(token in prompt for token in QUERY_TOKENS):
        raise ValueError("model-specific query token found in generic benchmark data")
    return {
        "benchmark_id": row["benchmark_id"],
        "pair_id": row["pair_id"],
        "messages": messages,
    }


def v2_regression_head_example(
    row: dict[str, Any], labels: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Render a v2 head example in memory without modifying the benchmark."""
    example = generic_vlm_example(row)
    pair_id = str(example["pair_id"])
    if pair_id not in labels:
        raise KeyError(f"missing label for pair_id={pair_id}")
    example["messages"][-1]["content"] += QUERY_SUFFIX
    example["outcome_targets"] = labels[pair_id]["outcome_targets"]
    example["outcome_mask"] = labels[pair_id]["outcome_mask"]
    return example


def iter_generic_vlm(path: Path) -> Iterator[dict[str, Any]]:
    for row in read_jsonl(path):
        yield generic_vlm_example(row)


def iter_v2_regression_head(
    path: Path, labels_path: Path
) -> Iterator[dict[str, Any]]:
    labels = load_labels(labels_path)
    for row in read_jsonl(path):
        yield v2_regression_head_example(row, labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("generic_vlm", "v2_regression_head"),
        default="generic_vlm",
    )
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "v2_regression_head":
        if args.labels is None:
            raise ValueError("--labels is required for v2_regression_head")
        iterator = iter_v2_regression_head(args.test_file, args.labels)
    else:
        iterator = iter_generic_vlm(args.test_file)
    for index, row in enumerate(iterator):
        if index >= args.limit:
            break
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
