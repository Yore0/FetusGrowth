#!/usr/bin/env python3
"""Audit pairing, perturbation semantics, and model independence of the benchmark."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from temporal_outcome.benchmark import renderers

QUERY_TOKENS = (
    "<|delivery_outcome_query|>",
    "<|birth_weight_outcome_query|>",
    "<|birth_length_outcome_query|>",
)
LABEL_KEYS = {"outcome_targets", "outcome_mask", "target_mask_reasons"}


def jsonl_rows(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.rstrip("\n")
            try:
                yield raw, json.loads(raw)
            except json.JSONDecodeError as error:
                raise AssertionError(f"{path}:{line_number}: invalid JSON: {error}") from error


def check_row(row: dict[str, Any], variant: str, label_ids: set[str]) -> None:
    pair_id = str(row["pair_id"])
    assert row["benchmark_id"] == f"{pair_id}::{variant}"
    assert row["clean_reference_id"] == f"{pair_id}::clean_continuous_prefix"
    assert row["variant_name"] == variant
    assert pair_id in label_ids
    assert not (LABEL_KEYS & set(row))
    assert row["cutoff_day"] not in {97, 160, 202, 230, 258}
    messages = row["messages"]
    assert len(messages) == 1 and messages[0].get("role") == "user"
    serialized = json.dumps(row, ensure_ascii=False)
    assert not any(token in serialized for token in QUERY_TOKENS)

    aug = row["augmentation"]
    if variant.startswith("content_mask_"):
        requested = int(variant.rsplit("_", 1)[1]) / 100
        eligible = int(aug["content_mask_eligible_fields"])
        masked = int(aug["content_masked_fields"])
        assert math.isclose(float(aug["mask_level"]), requested)
        assert math.isclose(float(aug["content_mask_fraction_requested"]), requested)
        assert masked == max(1, round(eligible * requested))
    elif variant in {"visit_dropout_20", "visit_dropout_40"}:
        requested = int(variant.rsplit("_", 1)[1]) / 100
        visits = int(aug["source_visit_count_before_dropout"])
        removed = int(aug["removed_visit_count"])
        assert removed == min(max(1, round(visits * requested)), visits - 1)
        assert math.isclose(float(aug["actual_visit_dropout_fraction"]), removed / visits)
    elif variant == "visit_dropout_latest":
        assert aug["dropout_strategy"] == "latest"
        assert int(aug["removed_visit_count"]) == 1
        removed_days = aug["removed_visit_end_days"]
        assert removed_days and row["visible_last_day"] <= max(removed_days)
    elif variant.startswith("modality_drop_"):
        modality = variant.removeprefix("modality_drop_")
        assert aug["removed_modality"] == modality
        assert int(aug["removed_record_count"]) > 0
        assert modality not in row["visible_modality_counts"]
    elif variant.startswith("local_window_"):
        width = int(variant.rsplit("_", 1)[1])
        assert int(aug["local_window_width_days"]) == width
        assert row["window_start_day"] == max(0, row["cutoff_day"] - width + 1)
        assert int(aug["removed_record_count"]) > 0
    elif variant == "compound_realistic":
        components = aug["compound_components"]
        assert int(aug["compound_component_count"]) == len(components)
        assert len(components) >= 2 and len(components) == len(set(components))


def load_labels(path: Path) -> tuple[dict[str, str], Counter[str]]:
    labels: dict[str, str] = {}
    cohorts: Counter[str] = Counter()
    for raw, row in jsonl_rows(path):
        pair_id = str(row["pair_id"])
        assert pair_id not in labels
        labels[pair_id] = raw
        cohorts[str(row["cohort"])] += 1
    return labels, cohorts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    variants = list(manifest["variants"])

    full_labels, full_label_cohorts = load_labels(root / "full/labels/outcomes.jsonl")
    full_label_ids = set(full_labels)
    core_labels, core_label_cohorts = load_labels(root / "core/labels/outcomes.jsonl")
    core_label_ids = set(core_labels)
    assert core_label_ids <= full_label_ids
    assert all(full_labels[pair_id] == raw for pair_id, raw in core_labels.items())

    report: dict[str, Any] = {
        "status": "passed",
        "labels": {
            "core": {"rows": len(core_labels), "cohorts": dict(core_label_cohorts)},
            "full": {"rows": len(full_labels), "cohorts": dict(full_label_cohorts)},
            "core_exact_subset_of_full": True,
        },
        "variants": {},
        "invariants": {
            "model_specific_query_tokens_absent_from_test_jsonl": True,
            "ground_truth_absent_from_test_jsonl": True,
            "user_only_messages": True,
            "all_cutoffs_non_anchor": True,
            "every_test_pair_has_separate_label": True,
        },
    }

    for variant in variants:
        core_path = root / "core" / variant / "test.jsonl"
        full_path = root / "full" / variant / "test.jsonl"
        core_raw: dict[str, str] = {}
        core_rates: list[float] = []
        for raw, row in jsonl_rows(core_path):
            benchmark_id = str(row["benchmark_id"])
            assert benchmark_id not in core_raw
            check_row(row, variant, core_label_ids)
            core_raw[benchmark_id] = raw
            if variant.startswith("content_mask_"):
                aug = row["augmentation"]
                core_rates.append(aug["content_masked_fields"] / aug["content_mask_eligible_fields"])

        unmatched = dict(core_raw)
        full_seen: set[str] = set()
        full_count = 0
        full_rates: list[float] = []
        for raw, row in jsonl_rows(full_path):
            full_count += 1
            benchmark_id = str(row["benchmark_id"])
            assert benchmark_id not in full_seen
            full_seen.add(benchmark_id)
            check_row(row, variant, full_label_ids)
            if benchmark_id in unmatched:
                assert unmatched.pop(benchmark_id) == raw
            if variant.startswith("content_mask_"):
                aug = row["augmentation"]
                full_rates.append(aug["content_masked_fields"] / aug["content_mask_eligible_fields"])
        assert not unmatched

        slot: dict[str, Any] = {
            "core_rows": len(core_raw),
            "full_rows": full_count,
            "core_exact_subset_of_full": True,
        }
        if core_rates:
            slot["core_actual_mask_fraction_mean"] = sum(core_rates) / len(core_rates)
            slot["full_actual_mask_fraction_mean"] = sum(full_rates) / len(full_rates)
        report["variants"][variant] = slot

    _, sample = next(jsonl_rows(root / "core/clean_continuous_prefix/test.jsonl"))
    generic = renderers.generic_vlm_example(sample)
    rendered = renderers.v2_regression_head_example(
        sample,
        {sample["pair_id"]: json.loads(core_labels[sample["pair_id"]])},
    )
    generic_prompt = generic["messages"][0]["content"]
    rendered_prompt = rendered["messages"][0]["content"]
    assert not any(token in generic_prompt for token in QUERY_TOKENS)
    assert all(rendered_prompt.count(token) == 1 for token in QUERY_TOKENS)
    assert "outcome_targets" not in generic and "outcome_targets" in rendered
    report["renderer_smoke_test"] = {
        "generic_vlm_messages_unchanged": generic["messages"] == sample["messages"],
        "v2_query_tokens_injected_at_runtime_only": True,
        "v2_labels_joined_at_runtime_only": True,
    }
    assert report["renderer_smoke_test"]["generic_vlm_messages_unchanged"]

    rendered_report = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered_report, encoding="utf-8")
    print(rendered_report, end="")


if __name__ == "__main__":
    main()
