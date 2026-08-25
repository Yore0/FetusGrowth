#!/usr/bin/env python3
"""Build paired, model-agnostic continuous/missingness evaluation JSONL files.

The emitted test files contain exactly one generic user message and never
contain assistant labels, regression targets, or the three internal outcome
query tokens. Model-specific rendering belongs in benchmark_renderers.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence, TextIO

from temporal_outcome.data import flexible, temporal_views

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks/model_agnostic_robustness_v1"
SEED = 20260810

VARIANTS = (
    "clean_continuous_prefix",
    "content_mask_15",
    "content_mask_30",
    "content_mask_50",
    "visit_dropout_20",
    "visit_dropout_40",
    "visit_dropout_latest",
    "modality_drop_ultrasound",
    "modality_drop_lab",
    "local_window_28",
    "local_window_56",
    "local_window_84",
    "compound_realistic",
)
TIERS = ("core", "full")
INTERNAL_QUERY_MARKERS = (
    "[内部结局回归查询]",
    "<|delivery_outcome_query|>",
    "<|birth_weight_outcome_query|>",
    "<|birth_length_outcome_query|>",
)
FORBIDDEN_PROMPT_SECTIONS = (
    "[分娩前摘要]",
    "[重点异常与诊断提示]",
    "[关键异常对齐摘要]",
    "[软性提示]",
    "[建模辅助提示]",
)
LABEL_KEYS = {
    "outcome_targets",
    "outcome_mask",
    "targets",
    "actual_delivery_days",
    "actual_birth_weight_g",
    "actual_birth_length_cm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Private directory containing huaxi/ and shenzhen/ evaluation JSONL.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def stable_int(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def write_jsonl_row(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def choose_non_landmark_cutoff(
    flexible: Any,
    case: dict[str, Any],
    cutoff_bin: Any,
    seed: int,
) -> int | None:
    delivery = int(case["targets"]["delivery_days"])
    minimum = max(cutoff_bin.low, flexible.min_block_end(case))
    maximum = min(cutoff_bin.high, delivery - 1)
    candidates = [
        value
        for value in range(minimum, maximum + 1)
        if value != cutoff_bin.anchor
    ]
    if not candidates:
        return None
    rng = random.Random(
        stable_int(seed, f"cutoff:{case['cohort']}:{case['case_id']}:{cutoff_bin.name}")
    )
    return candidates[rng.randrange(len(candidates))]


def build_full_parents(
    temporal: Any,
    flexible: Any,
    cases: Sequence[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    for case in cases:
        for slot, cutoff_bin in enumerate(flexible.CUTOFF_BINS):
            cutoff = choose_non_landmark_cutoff(flexible, case, cutoff_bin, seed)
            if cutoff is None:
                continue
            blocks = temporal.visible_blocks(case, window_start_day=0, cutoff_day=cutoff)
            if not blocks:
                continue
            pair_id = (
                f"{case['cohort']}:{case['case_id']}:"
                f"{cutoff_bin.name}:d{cutoff:03d}"
            )
            parents.append(
                {
                    "pair_id": pair_id,
                    "case": case,
                    "cutoff_bin": cutoff_bin,
                    "continuous_slot": slot,
                    "cutoff_day": cutoff,
                    "clean_blocks": [dict(block) for block in blocks],
                }
            )
    return parents


def select_core_parent_ids(
    parents: Sequence[dict[str, Any]], seed: int
) -> set[str]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        case = parent["case"]
        by_case[f"{case['cohort']}:{case['case_id']}"].append(parent)
    counts: Counter[tuple[str, str]] = Counter()
    selected: set[str] = set()
    ordered_cases = sorted(
        by_case,
        key=lambda key: (stable_int(seed, f"core-order:{key}"), key),
    )
    for case_key in ordered_cases:
        candidates = by_case[case_key]
        cohort = str(candidates[0]["case"]["cohort"])
        minimum = min(counts[(cohort, parent["cutoff_bin"].name)] for parent in candidates)
        balanced = [
            parent
            for parent in candidates
            if counts[(cohort, parent["cutoff_bin"].name)] == minimum
        ]
        selected_parent = min(
            balanced,
            key=lambda parent: (
                stable_int(seed, f"core-choice:{case_key}:{parent['pair_id']}"),
                parent["pair_id"],
            ),
        )
        selected.add(selected_parent["pair_id"])
        counts[(cohort, selected_parent["cutoff_bin"].name)] += 1
    return selected


def dropout_groups(
    temporal: Any,
    blocks: Sequence[dict[str, Any]],
    *,
    fraction: float | None,
    latest: bool,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    groups = temporal.encounter_groups(blocks)
    if len(groups) <= 1:
        return None
    if latest:
        removed_indices = {len(groups) - 1}
        requested = None
    else:
        assert fraction is not None
        remove_count = max(1, int(round(len(groups) * fraction)))
        remove_count = min(remove_count, len(groups) - 1)
        removed_indices = set(rng.sample(range(len(groups)), remove_count))
        requested = fraction
    retained = [
        dict(block)
        for index, group in enumerate(groups)
        if index not in removed_indices
        for block in group
    ]
    removed = [
        block
        for index, group in enumerate(groups)
        if index in removed_indices
        for block in group
    ]
    return retained, {
        "dropout_fraction_requested": requested,
        "source_visit_count_before_dropout": len(groups),
        "removed_visit_count": len(removed_indices),
        "removed_record_count": len(removed),
        "removed_visit_end_days": sorted({int(block["end_day"]) for block in removed}),
        "actual_visit_dropout_fraction": len(removed_indices) / len(groups),
    }


def modality_drop(
    blocks: Sequence[dict[str, Any]], modality: str
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    removed = [block for block in blocks if block["modality"] == modality]
    retained = [dict(block) for block in blocks if block["modality"] != modality]
    if not removed or not retained:
        return None
    return retained, {
        "removed_modality": modality,
        "removed_record_count": len(removed),
        "source_record_count_before_modality_drop": len(blocks),
    }


def local_window(
    temporal: Any,
    case: dict[str, Any],
    clean_blocks: Sequence[dict[str, Any]],
    cutoff: int,
    width: int,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]] | None:
    start = max(0, cutoff - width + 1)
    blocks = temporal.visible_blocks(case, window_start_day=start, cutoff_day=cutoff)
    if not blocks or len(blocks) >= len(clean_blocks):
        return None
    return start, [dict(block) for block in blocks], {
        "local_window_width_days": width,
        "removed_record_count": len(clean_blocks) - len(blocks),
    }


def variant_blocks(
    temporal: Any,
    flexible: Any,
    parent: dict[str, Any],
    variant: str,
    seed: int,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]] | None:
    case = parent["case"]
    cutoff = int(parent["cutoff_day"])
    clean = [dict(block) for block in parent["clean_blocks"]]
    rng = random.Random(stable_int(seed, f"variant:{variant}:{parent['pair_id']}"))
    if variant == "clean_continuous_prefix":
        return 0, clean, {"variant_applied": True}

    if variant.startswith("content_mask_"):
        fraction = int(variant.rsplit("_", 1)[1]) / 100
        blocks, metadata = flexible.mask_visit_content(
            clean, rng=rng, minimum=fraction, maximum=fraction
        )
        if not metadata["content_mask_applied"]:
            return None
        metadata.update({"variant_applied": True, "mask_level": fraction})
        return 0, blocks, metadata

    if variant in {"visit_dropout_20", "visit_dropout_40"}:
        fraction = int(variant.rsplit("_", 1)[1]) / 100
        result = dropout_groups(
            temporal, clean, fraction=fraction, latest=False, rng=rng
        )
        if result is None:
            return None
        blocks, metadata = result
        metadata["variant_applied"] = True
        return 0, blocks, metadata

    if variant == "visit_dropout_latest":
        result = dropout_groups(temporal, clean, fraction=None, latest=True, rng=rng)
        if result is None:
            return None
        blocks, metadata = result
        metadata.update({"variant_applied": True, "dropout_strategy": "latest"})
        return 0, blocks, metadata

    if variant.startswith("modality_drop_"):
        modality = variant.removeprefix("modality_drop_")
        result = modality_drop(clean, modality)
        if result is None:
            return None
        blocks, metadata = result
        metadata["variant_applied"] = True
        return 0, blocks, metadata

    if variant.startswith("local_window_"):
        width = int(variant.rsplit("_", 1)[1])
        result = local_window(temporal, case, clean, cutoff, width)
        if result is None:
            return None
        start, blocks, metadata = result
        metadata["variant_applied"] = True
        return start, blocks, metadata

    if variant == "compound_realistic":
        components: list[str] = []
        metadata: dict[str, Any] = {"variant_applied": True}
        start = 0
        blocks = clean

        local = local_window(temporal, case, blocks, cutoff, 84)
        if local is not None:
            start, blocks, local_meta = local
            components.append("local_window_84")
            metadata["local_window"] = local_meta

        available_modalities = sorted({str(block["modality"]) for block in blocks})
        droppable = [
            modality
            for modality in ("ultrasound", "lab")
            if modality in available_modalities
            and any(block["modality"] != modality for block in blocks)
        ]
        if droppable and rng.random() < 0.50:
            selected = droppable[rng.randrange(len(droppable))]
            dropped = modality_drop(blocks, selected)
            assert dropped is not None
            blocks, modality_meta = dropped
            components.append(f"modality_drop_{selected}")
            metadata["modality_drop"] = modality_meta

        visit_result = dropout_groups(
            temporal, blocks, fraction=0.20, latest=False, rng=rng
        )
        if visit_result is not None:
            blocks, visit_meta = visit_result
            components.append("visit_dropout_20")
            metadata["visit_dropout"] = visit_meta

        masked, mask_meta = flexible.mask_visit_content(
            blocks, rng=rng, minimum=0.15, maximum=0.15
        )
        if mask_meta["content_mask_applied"]:
            blocks = masked
            components.append("content_mask_15")
            metadata["content_mask"] = mask_meta

        if len(components) < 2 or not blocks:
            return None
        metadata["compound_components"] = components
        metadata["compound_component_count"] = len(components)
        return start, blocks, metadata

    raise ValueError(f"unsupported variant: {variant}")


def make_test_row(
    temporal: Any,
    parent: dict[str, Any],
    variant: str,
    start: int,
    blocks: Sequence[dict[str, Any]],
    augmentation: dict[str, Any],
) -> dict[str, Any]:
    case = parent["case"]
    pair_id = str(parent["pair_id"])
    cutoff = int(parent["cutoff_day"])
    view = temporal.make_view(
        case,
        blocks,
        split="model_agnostic_benchmark",
        view_type="benchmark_variant",
        view_name=f"{pair_id}:{variant}",
        stage=parent["cutoff_bin"].name,
        window_start_day=start,
        cutoff_day=cutoff,
        is_huaxi_seen_test=case["cohort"] == "huaxi",
        extra=None,
    )
    prompt = str(view["messages"][0]["content"])
    if any(marker in prompt for marker in INTERNAL_QUERY_MARKERS):
        raise AssertionError("model-specific query token leaked into benchmark prompt")
    if any(section in prompt for section in FORBIDDEN_PROMPT_SECTIONS):
        raise AssertionError("future/hint section leaked into benchmark prompt")
    row: dict[str, Any] = {
        "benchmark_id": f"{pair_id}::{variant}",
        "pair_id": pair_id,
        "clean_reference_id": f"{pair_id}::clean_continuous_prefix",
        "case_id": str(case["case_id"]),
        "cohort": str(case["cohort"]),
        "source_name": str(case["source_name"]),
        "variant_family": variant.split("_", 1)[0],
        "variant_name": variant,
        "cutoff_bin": parent["cutoff_bin"].name,
        "continuous_slot": int(parent["continuous_slot"]),
        "cutoff_day": cutoff,
        "window_start_day": start,
        "window_end_day": cutoff,
        "visible_record_count": int(view["visible_record_count"]),
        "visible_visit_count": int(view["visible_visit_count"]),
        "visible_first_day": view["visible_first_day"],
        "visible_last_day": view["visible_last_day"],
        "days_since_last_record": view["days_since_last_record"],
        "visible_modality_counts": view["visible_modality_counts"],
        "augmentation": augmentation,
        "messages": [{"role": "user", "content": prompt}],
    }
    if LABEL_KEYS & set(row):
        raise AssertionError("label key leaked into benchmark row")
    return row


def label_row(flexible: Any, parent: dict[str, Any]) -> dict[str, Any]:
    values, mask, reasons = flexible.validated_targets(parent["case"])
    return {
        "pair_id": parent["pair_id"],
        "case_id": str(parent["case"]["case_id"]),
        "cohort": str(parent["case"]["cohort"]),
        "cutoff_bin": parent["cutoff_bin"].name,
        "continuous_slot": int(parent["continuous_slot"]),
        "cutoff_day": int(parent["cutoff_day"]),
        "outcome_targets": values,
        "outcome_mask": mask,
        "target_mask_reasons": reasons,
    }


def increment_summary(
    summary: dict[str, Any], tier: str, variant: str, row: dict[str, Any]
) -> None:
    slot = summary["files"][tier][variant]
    slot["rows"] += 1
    slot["cohorts"][row["cohort"]] += 1
    slot["cutoff_bins"][row["cutoff_bin"]] += 1
    slot["visible_visit_counts"][str(row["visible_visit_count"])] += 1
    components = row["augmentation"].get("compound_components", [])
    for component in components:
        slot["compound_components"][component] += 1


def json_ready(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, defaultdict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def audit_output(
    output_root: Path,
    parent_ids: dict[str, set[str]],
) -> dict[str, Any]:
    audit: dict[str, Any] = {"files": {}, "errors": Counter()}
    clean_ids: dict[str, set[str]] = {}
    for tier in TIERS:
        clean_path = output_root / tier / "clean_continuous_prefix/test.jsonl"
        clean_ids[tier] = {
            json.loads(line)["pair_id"] for line in clean_path.open(encoding="utf-8")
        }
        if clean_ids[tier] != parent_ids[tier]:
            audit["errors"][f"{tier}_clean_parent_mismatch"] += 1
        for variant in VARIANTS:
            path = output_root / tier / variant / "test.jsonl"
            seen_benchmark_ids: set[str] = set()
            rows = 0
            cohort_counts: Counter[str] = Counter()
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                rows += 1
                cohort_counts[row["cohort"]] += 1
                benchmark_id = row["benchmark_id"]
                if benchmark_id in seen_benchmark_ids:
                    audit["errors"]["duplicate_benchmark_id"] += 1
                seen_benchmark_ids.add(benchmark_id)
                if row["pair_id"] not in clean_ids[tier]:
                    audit["errors"]["variant_missing_clean_reference"] += 1
                if LABEL_KEYS & set(row):
                    audit["errors"]["label_key_in_test_row"] += 1
                messages = row.get("messages")
                if (
                    not isinstance(messages, list)
                    or len(messages) != 1
                    or messages[0].get("role") != "user"
                ):
                    audit["errors"]["non_user_only_message"] += 1
                    continue
                prompt = str(messages[0].get("content") or "")
                if any(marker in prompt for marker in INTERNAL_QUERY_MARKERS):
                    audit["errors"]["internal_query_token_in_prompt"] += 1
                if any(section in prompt for section in FORBIDDEN_PROMPT_SECTIONS):
                    audit["errors"]["forbidden_section_in_prompt"] += 1
            audit["files"][f"{tier}/{variant}"] = {
                "rows": rows,
                "cohorts": dict(cohort_counts),
            }
    audit["errors"] = dict(audit["errors"])
    return audit


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing benchmark: {args.output_root}"
        )
    temporal = temporal_views
    source_paths = {
        "huaxi": args.source_root / "huaxi/huaxi_test.jsonl",
        "shenzhen": args.source_root
        / "shenzhen/shenzhen_internal_test_all__full.jsonl",
    }
    raw = {cohort: read_jsonl(path) for cohort, path in source_paths.items()}
    cases: list[dict[str, Any]] = []
    cleaning: dict[str, Counter[str]] = {}
    for cohort in ("huaxi", "shenzhen"):
        parsed, cohort_audit = temporal.parse_source_rows(
            raw[cohort], cohort=cohort, source_name=source_paths[cohort].name
        )
        cases.extend(parsed)
        cleaning[cohort] = cohort_audit

    full_parents = build_full_parents(temporal, flexible, cases, args.seed)
    if not full_parents:
        raise RuntimeError("no eligible continuous parents")
    core_ids = select_core_parent_ids(full_parents, args.seed)
    full_ids = {parent["pair_id"] for parent in full_parents}
    if not core_ids <= full_ids:
        raise AssertionError("core IDs are not a subset of full IDs")

    args.output_root.mkdir(parents=True)
    for tier in TIERS:
        (args.output_root / tier / "labels").mkdir(parents=True)
        for variant in VARIANTS:
            (args.output_root / tier / variant).mkdir(parents=True)

    summary: dict[str, Any] = {
        "protocol": "model_agnostic_continuous_missingness_benchmark_v1",
        "seed": args.seed,
        "definitions": {
            "core": "one deterministic balanced non-landmark continuous cutoff per valid case",
            "full": "one deterministic non-landmark continuous cutoff per eligible case and cutoff bin",
            "paired_variants": True,
            "model_agnostic_user_only_test_files": True,
            "internal_query_tokens_in_data": False,
            "assistant_ground_truth_in_data": False,
            "labels_stored_separately": True,
        },
        "sources": {
            cohort: {
                "path": str(source_paths[cohort]),
                "raw_rows": len(raw[cohort]),
                "valid_cases": sum(case["cohort"] == cohort for case in cases),
                "dropped_cases": len(raw[cohort])
                - sum(case["cohort"] == cohort for case in cases),
                "cleaning_audit": dict(cleaning[cohort]),
            }
            for cohort in ("huaxi", "shenzhen")
        },
        "cutoff_bins": [asdict(item) for item in flexible.CUTOFF_BINS],
        "variants": list(VARIANTS),
        "parent_counts": {
            "core": len(core_ids),
            "full": len(full_ids),
        },
        "files": {
            tier: {
                variant: {
                    "path": str(args.output_root / tier / variant / "test.jsonl"),
                    "rows": 0,
                    "cohorts": Counter(),
                    "cutoff_bins": Counter(),
                    "visible_visit_counts": Counter(),
                    "compound_components": Counter(),
                }
                for variant in VARIANTS
            }
            for tier in TIERS
        },
        "skipped_ineligible": {variant: 0 for variant in VARIANTS},
    }

    with ExitStack() as stack:
        writers: dict[tuple[str, str], TextIO] = {}
        label_writers: dict[str, TextIO] = {}
        for tier in TIERS:
            label_writers[tier] = stack.enter_context(
                (args.output_root / tier / "labels/outcomes.jsonl").open(
                    "w", encoding="utf-8"
                )
            )
            for variant in VARIANTS:
                writers[(tier, variant)] = stack.enter_context(
                    (args.output_root / tier / variant / "test.jsonl").open(
                        "w", encoding="utf-8"
                    )
                )

        for position, parent in enumerate(full_parents, start=1):
            pair_id = parent["pair_id"]
            tiers = ["full"] + (["core"] if pair_id in core_ids else [])
            label = label_row(flexible, parent)
            for tier in tiers:
                write_jsonl_row(label_writers[tier], label)
            for variant in VARIANTS:
                result = variant_blocks(
                    temporal, flexible, parent, variant, args.seed
                )
                if result is None:
                    summary["skipped_ineligible"][variant] += 1
                    continue
                start, blocks, augmentation = result
                row = make_test_row(
                    temporal,
                    parent,
                    variant,
                    start,
                    blocks,
                    augmentation,
                )
                for tier in tiers:
                    write_jsonl_row(writers[(tier, variant)], row)
                    increment_summary(summary, tier, variant, row)
            if position % 1000 == 0:
                print(f"[build] parents={position}/{len(full_parents)}", flush=True)

    parent_ids = {"core": core_ids, "full": full_ids}
    output_audit = audit_output(args.output_root, parent_ids)
    summary["output_audit"] = output_audit
    summary["invariants"] = {
        "core_is_subset_of_full": core_ids <= full_ids,
        "all_test_rows_have_clean_reference": not output_audit["errors"].get(
            "variant_missing_clean_reference", 0
        ),
        "no_label_keys_in_test_rows": not output_audit["errors"].get(
            "label_key_in_test_row", 0
        ),
        "user_only_messages": not output_audit["errors"].get(
            "non_user_only_message", 0
        ),
        "no_internal_query_tokens": not output_audit["errors"].get(
            "internal_query_token_in_prompt", 0
        ),
        "no_forbidden_future_or_hint_sections": not output_audit["errors"].get(
            "forbidden_section_in_prompt", 0
        ),
        "all_cutoffs_non_landmark": all(
            parent["cutoff_day"] != parent["cutoff_bin"].anchor
            for parent in full_parents
        ),
        "all_cutoffs_before_delivery": all(
            parent["cutoff_day"]
            < int(parent["case"]["targets"]["delivery_days"])
            for parent in full_parents
        ),
    }
    ready_summary = json_ready(summary)
    (args.output_root / "manifest.json").write_text(
        json.dumps(ready_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ready_summary["parent_counts"], ensure_ascii=False))
    print(json.dumps(ready_summary["invariants"], ensure_ascii=False))
    print(args.output_root)


if __name__ == "__main__":
    main()
