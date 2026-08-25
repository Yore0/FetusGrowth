#!/usr/bin/env python3
"""Shared targets, sampling, and augmentation helpers for continuous training.

The source parsing, leakage removal, anomaly cleaning, train/validation split,
and Huaxi seen-test policy are reused from the audited temporal-v1 builder.
Only training-view construction changes:

* fixed landmark prefix anchors;
* day-level random prefix cutoffs;
* day-level random local windows;
* random whole-visit dropout;
* optional value-level masking inside retained visits.

The output is already in the three-query regression schema consumed by the
Qwen3.5 outcome-regression plugin. Invalid optional outcomes are written as NA
in the CE target and masked in the regression target, keeping both losses
consistent.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPORAL_ROOT = PROJECT_ROOT
REGRESSION_V7_ROOT = PROJECT_ROOT
SOURCE_ROOT = PROJECT_ROOT / "data/raw"

SEED = 20260806
TOTAL_DRAWS = 80_000
COHORTS = ("huaxi", "shenzhen")
TARGET_NAMES = ("delivery_days", "birth_weight_g", "birth_length_cm")
QUERY_TOKENS = (
    "<|delivery_outcome_query|>",
    "<|birth_weight_outcome_query|>",
    "<|birth_length_outcome_query|>",
)
QUERY_SUFFIX = "\n\n[内部结局回归查询]\n" + "\n".join(QUERY_TOKENS)
FORBIDDEN_USER_SECTIONS = (
    "[分娩前摘要]",
    "[重点异常与诊断提示]",
    "[关键异常对齐摘要]",
    "[软性提示]",
    "[建模辅助提示]",
)


@dataclass(frozen=True)
class CutoffBin:
    name: str
    low: int
    high: int
    anchor: int


CUTOFF_BINS = (
    CutoffBin("d070_111", 70, 111, 97),
    CutoffBin("d112_181", 112, 181, 160),
    CutoffBin("d182_215", 182, 215, 202),
    CutoffBin("d216_244", 216, 244, 230),
    CutoffBin("d245_258", 245, 258, 258),
)
VIEW_RATIOS = {
    "landmark_prefix": 0.15,
    "flexible_prefix": 0.35,
    "flexible_local_window": 0.30,
    "visit_bundle_dropout": 0.20,
}

FIELD_RE = re.compile(
    r"(?P<label>(?:^|[,，;；]\s*)[^=,，;；\n]{1,64}=)"
    r"(?P<value>[^,，;；\n]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--temporal-root", type=Path, default=TEMPORAL_ROOT)
    parser.add_argument("--regression-v7-root", type=Path, default=REGRESSION_V7_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--draws", type=int, default=TOTAL_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--content-mask-probability", type=float, default=0.35)
    parser.add_argument("--content-mask-min", type=float, default=0.15)
    parser.add_argument("--content-mask-max", type=float, default=0.35)
    parser.add_argument("--visit-dropout-min", type=float, default=0.10)
    parser.add_argument("--visit-dropout-max", type=float, default=0.30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_int(seed: int, text: str) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def equal_counts(total: int, labels: Sequence[str]) -> dict[str, int]:
    if not labels:
        raise ValueError("labels must not be empty")
    base, remainder = divmod(total, len(labels))
    return {
        label: base + int(index < remainder)
        for index, label in enumerate(labels)
    }


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validated_targets(case: dict[str, Any]) -> tuple[list[float], list[int], list[str]]:
    raw = case["targets"]
    delivery = finite(raw.get("delivery_days"))
    weight = finite(raw.get("birth_weight_g"))
    length = finite(raw.get("birth_length_cm"))
    values = [0.0, 0.0, 0.0]
    mask = [0, 0, 0]
    reasons: list[str] = []

    if delivery is not None and 140 <= delivery <= 320:
        values[0], mask[0] = delivery, 1
    else:
        reasons.append("delivery_missing_or_invalid")
    if weight is not None and 500 <= weight <= 6000:
        values[1], mask[1] = weight, 1
    else:
        reasons.append("birth_weight_missing_or_invalid")

    length_valid = length is not None and 20 <= length <= 65
    if (
        length_valid
        and delivery is not None
        and weight is not None
        and delivery >= 259
        and weight >= 2000
        and length < 40
    ):
        length_valid = False
        reasons.append("birth_length_cross_field_implausible")
    if length_valid:
        values[2], mask[2] = length, 1
    elif "birth_length_cross_field_implausible" not in reasons:
        reasons.append("birth_length_missing_or_invalid")
    if not mask[0]:
        raise ValueError(f"invalid delivery target for case {case['case_id']}")
    return values, mask, reasons


def target_response(values: Sequence[float], mask: Sequence[int]) -> str:
    rendered = []
    for name, value, valid in zip(TARGET_NAMES, values, mask, strict=True):
        text = "NA" if not valid else str(int(round(float(value))))
        rendered.append(f"<{name}>{text}</{name}>")
    return "\n".join(rendered)


def copy_blocks(blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(block) for block in blocks]


def candidate_fields(blocks: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for block_index, block in enumerate(blocks):
        lines = str(block["text"]).splitlines()
        for line_index, line in enumerate(lines):
            if line_index == 0:
                continue
            for match_index, match in enumerate(FIELD_RE.finditer(line)):
                value = match.group("value").strip().lower()
                if value in {"na", "暂无", "未知"} or value.startswith("na "):
                    continue
                candidates.append((block_index, line_index * 10_000 + match_index))
    return candidates


def mask_visit_content(
    blocks: Sequence[dict[str, Any]],
    *,
    rng: random.Random,
    minimum: float,
    maximum: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masked_blocks = copy_blocks(blocks)
    indexed: list[tuple[int, int, int]] = []
    matches_by_line: dict[tuple[int, int], list[re.Match[str]]] = {}
    for block_index, block in enumerate(masked_blocks):
        lines = str(block["text"]).splitlines()
        for line_index, line in enumerate(lines):
            if line_index == 0:
                continue
            matches = [
                match
                for match in FIELD_RE.finditer(line)
                if match.group("value").strip().lower() not in {"na", "暂无", "未知"}
                and not match.group("value").strip().lower().startswith("na ")
            ]
            if matches:
                matches_by_line[(block_index, line_index)] = matches
                indexed.extend(
                    (block_index, line_index, match_index)
                    for match_index in range(len(matches))
                )
    if not indexed:
        return masked_blocks, {
            "content_mask_applied": False,
            "content_mask_eligible_fields": 0,
            "content_masked_fields": 0,
        }

    fraction = rng.uniform(minimum, maximum)
    mask_count = max(1, int(round(len(indexed) * fraction)))
    if len(indexed) > 1:
        mask_count = min(mask_count, len(indexed) - 1)
    selected = set(rng.sample(indexed, mask_count))

    for block_index, block in enumerate(masked_blocks):
        lines = str(block["text"]).splitlines()
        for line_index in range(1, len(lines)):
            matches = matches_by_line.get((block_index, line_index))
            if not matches:
                continue
            selected_indices = {
                match_index
                for match_index in range(len(matches))
                if (block_index, line_index, match_index) in selected
            }
            if not selected_indices:
                continue
            pieces: list[str] = []
            cursor = 0
            for match_index, match in enumerate(matches):
                pieces.append(lines[line_index][cursor:match.start()])
                if match_index in selected_indices:
                    pieces.append(match.group("label") + "NA")
                else:
                    pieces.append(match.group(0))
                cursor = match.end()
            pieces.append(lines[line_index][cursor:])
            lines[line_index] = "".join(pieces)
        block["text"] = "\n".join(lines)

    return masked_blocks, {
        "content_mask_applied": True,
        "content_mask_fraction_requested": round(fraction, 6),
        "content_mask_eligible_fields": len(indexed),
        "content_masked_fields": len(selected),
    }


class WeightedPool:
    def __init__(self, indices: Sequence[int], weights: Sequence[float]) -> None:
        self.indices = list(indices)
        self.cumulative: list[float] = []
        running = 0.0
        for index in self.indices:
            running += float(weights[index])
            self.cumulative.append(running)
        if not self.indices or running <= 0:
            raise ValueError("weighted pool must be non-empty")
        self.total = running

    def draw(self, rng: random.Random) -> int:
        position = rng.random() * self.total
        selected = bisect.bisect_right(self.cumulative, position)
        return self.indices[min(selected, len(self.indices) - 1)]


def min_block_end(case: dict[str, Any]) -> int:
    return min(int(block["end_day"]) for block in case["blocks"])


def eligible_for_bin(case: dict[str, Any], cutoff_bin: CutoffBin) -> bool:
    delivery = int(case["targets"]["delivery_days"])
    maximum = min(cutoff_bin.high, delivery - 1)
    return maximum >= cutoff_bin.low and min_block_end(case) <= maximum


def choose_cutoff(
    case: dict[str, Any],
    cutoff_bin: CutoffBin,
    rng: random.Random,
) -> int:
    delivery = int(case["targets"]["delivery_days"])
    maximum = min(cutoff_bin.high, delivery - 1)
    minimum = max(cutoff_bin.low, min_block_end(case))
    if minimum > maximum:
        raise ValueError("case is not eligible for requested cutoff bin")
    return rng.randint(minimum, maximum)


def local_window_blocks(
    temporal: Any,
    case: dict[str, Any],
    cutoff: int,
    rng: random.Random,
) -> tuple[int, list[dict[str, Any]]]:
    prefix = temporal.visible_blocks(case, window_start_day=0, cutoff_day=cutoff)
    groups = temporal.encounter_groups(prefix)
    valid_group_indices = [
        index
        for index, group in enumerate(groups)
        if min(int(block["start_day"]) for block in group) > 0
    ]
    if not valid_group_indices:
        raise ValueError("no positive-day encounter for local window")
    group_index = valid_group_indices[rng.randrange(len(valid_group_indices))]
    group = groups[group_index]
    first_start = min(int(block["start_day"]) for block in group)
    previous_end = (
        max(int(block["end_day"]) for block in groups[group_index - 1])
        if group_index > 0
        else 0
    )
    lower = max(1, min(previous_end + 1, first_start))
    start = rng.randint(lower, first_start)
    blocks = temporal.visible_blocks(case, window_start_day=start, cutoff_day=cutoff)
    if not blocks:
        raise ValueError("sampled local window is empty")
    return start, blocks


def generate_blocks(
    temporal: Any,
    case: dict[str, Any],
    *,
    view_type: str,
    cutoff_bin: CutoffBin,
    rng: random.Random,
    visit_dropout_min: float,
    visit_dropout_max: float,
) -> tuple[int, int, list[dict[str, Any]], dict[str, Any]]:
    if view_type == "landmark_prefix":
        cutoff = cutoff_bin.anchor
        if int(case["targets"]["delivery_days"]) <= cutoff:
            raise ValueError("delivery precedes landmark")
        blocks = temporal.visible_blocks(case, window_start_day=0, cutoff_day=cutoff)
        if not blocks:
            raise ValueError("empty landmark prefix")
        return 0, cutoff, blocks, {}

    cutoff = choose_cutoff(case, cutoff_bin, rng)
    if view_type == "flexible_prefix":
        blocks = temporal.visible_blocks(case, window_start_day=0, cutoff_day=cutoff)
        if not blocks:
            raise ValueError("empty flexible prefix")
        return 0, cutoff, blocks, {}

    if view_type == "flexible_local_window":
        start, blocks = local_window_blocks(temporal, case, cutoff, rng)
        return start, cutoff, blocks, {}

    if view_type != "visit_bundle_dropout":
        raise ValueError(f"unknown view type: {view_type}")
    source = temporal.visible_blocks(case, window_start_day=0, cutoff_day=cutoff)
    groups = temporal.encounter_groups(source)
    if len(groups) <= 1:
        raise ValueError("visit dropout requires at least two encounters")
    fraction = rng.uniform(visit_dropout_min, visit_dropout_max)
    remove_count = max(1, int(round(len(groups) * fraction)))
    remove_count = min(remove_count, len(groups) - 1)
    removed_indices = set(rng.sample(range(len(groups)), remove_count))
    retained = [
        block
        for group_index, group in enumerate(groups)
        if group_index not in removed_indices
        for block in group
    ]
    removed_days = sorted(
        {
            int(block["end_day"])
            for group_index, group in enumerate(groups)
            if group_index in removed_indices
            for block in group
        }
    )
    return 0, cutoff, retained, {
        "dropout_fraction_requested": round(fraction, 6),
        "source_visit_count_before_dropout": len(groups),
        "removed_visit_count": remove_count,
        "removed_visit_end_days": removed_days,
    }


def make_regression_row(
    temporal: Any,
    case: dict[str, Any],
    *,
    position: int,
    view_type: str,
    cutoff_bin: CutoffBin,
    rng: random.Random,
    huaxi_test_ids: set[str],
    content_mask_probability: float,
    content_mask_min: float,
    content_mask_max: float,
    visit_dropout_min: float,
    visit_dropout_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start, cutoff, blocks, augmentation = generate_blocks(
        temporal,
        case,
        view_type=view_type,
        cutoff_bin=cutoff_bin,
        rng=rng,
        visit_dropout_min=visit_dropout_min,
        visit_dropout_max=visit_dropout_max,
    )
    mask_requested = rng.random() < content_mask_probability
    if mask_requested:
        blocks, content_meta = mask_visit_content(
            blocks,
            rng=rng,
            minimum=content_mask_min,
            maximum=content_mask_max,
        )
    else:
        blocks = copy_blocks(blocks)
        content_meta = {
            "content_mask_applied": False,
            "content_mask_eligible_fields": len(candidate_fields(blocks)),
            "content_masked_fields": 0,
        }
    augmentation.update(content_meta)

    view_name = f"flex_{position:06d}_{view_type}_d{start}_{cutoff}"
    stage = cutoff_bin.name
    source_view = temporal.make_view(
        case,
        blocks,
        split="train",
        view_type=view_type,
        view_name=view_name,
        stage=stage,
        window_start_day=start,
        cutoff_day=cutoff,
        is_huaxi_seen_test=(
            case["cohort"] == "huaxi" and str(case["case_id"]) in huaxi_test_ids
        ),
        extra=augmentation,
    )
    values, mask, reasons = validated_targets(case)
    messages = [dict(message) for message in source_view["messages"]]
    messages[0]["content"] = str(messages[0]["content"]) + QUERY_SUFFIX
    messages[1]["content"] = target_response(values, mask)
    row = {
        "messages": messages,
        "outcome_targets": values,
        "outcome_mask": mask,
    }
    metadata = {
        key: source_view[key]
        for key in (
            "view_id",
            "case_id",
            "cohort",
            "hospital",
            "is_huaxi_seen_test",
            "view_type",
            "stage",
            "window_start_day",
            "window_end_day",
            "visible_record_count",
            "visible_visit_count",
            "visible_first_day",
            "visible_last_day",
            "days_since_last_record",
            "visible_modality_counts",
        )
    }
    metadata.update(augmentation)
    metadata.update(
        {
            "schedule_position": position,
            "cutoff_bin": cutoff_bin.name,
            "target_mask_reasons": reasons,
            "outcome_targets": values,
            "outcome_mask": mask,
        }
    )
    return row, metadata


def load_training_and_validation_cases(
    temporal: Any,
    *,
    source_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], dict[str, Any]]:
    paths = {
        "huaxi_train": source_root / "huaxi/huaxi_train.jsonl",
        "huaxi_test": source_root / "huaxi/huaxi_test.jsonl",
        "shenzhen_train": source_root / "shenzhen/shenzhen_train_all__full.jsonl",
        "shenzhen_test": source_root / "shenzhen/shenzhen_internal_test_all__full.jsonl",
    }
    raw = {name: temporal.read_jsonl(path) for name, path in paths.items()}
    huaxi_test_ids = temporal.raw_id_set(raw["huaxi_test"])
    parsed: dict[str, list[dict[str, Any]]] = {}
    cleaning: dict[str, Counter[str]] = {}
    for name, cohort in (
        ("huaxi_train", "huaxi"),
        ("huaxi_test", "huaxi"),
        ("shenzhen_train", "shenzhen"),
        ("shenzhen_test", "shenzhen"),
    ):
        parsed[name], cleaning[name] = temporal.parse_source_rows(
            raw[name], cohort=cohort, source_name=paths[name].name
        )
    reconciliation = temporal.reconcile_huaxi_seen_test_cases(
        parsed["huaxi_train"], parsed["huaxi_test"]
    )
    train_cases, validation_cases, validation_ids = temporal.partition_training_cases(
        parsed["huaxi_train"],
        parsed["shenzhen_train"],
        huaxi_test_ids=huaxi_test_ids,
        val_fraction=0.10,
        seed=20260727,
    )
    report = {
        "paths": {name: str(path) for name, path in paths.items()},
        "raw_rows": {name: len(rows) for name, rows in raw.items()},
        "valid_cases": {name: len(rows) for name, rows in parsed.items()},
        "training_cases": len(train_cases),
        "validation_cases": len(validation_cases),
        "cohort_training_cases": dict(Counter(case["cohort"] for case in train_cases)),
        "huaxi_seen_test_ids": len(huaxi_test_ids),
        "huaxi_reconciliation": dict(reconciliation),
        "validation_ids": {name: len(values) for name, values in validation_ids.items()},
        "cleaning_audit": {name: dict(counter) for name, counter in cleaning.items()},
    }
    return train_cases, validation_cases, huaxi_test_ids, report


def load_training_cases(
    temporal: Any,
    *,
    source_root: Path,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Backward-compatible view used by the original experiment scripts."""
    train_cases, _, huaxi_test_ids, report = load_training_and_validation_cases(
        temporal, source_root=source_root
    )
    return train_cases, huaxi_test_ids, report


def normalize_validation(
    source: Path,
    destination: Path,
    smoke_destination: Path,
) -> dict[str, Any]:
    rows = 0
    corrected = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader:
        with destination.open("w", encoding="utf-8") as writer:
            with smoke_destination.open("w", encoding="utf-8") as smoke_writer:
                for line in reader:
                    row = json.loads(line)
                    messages = [dict(message) for message in row["messages"]]
                    values = row["outcome_targets"]
                    mask = row["outcome_mask"]
                    expected = target_response(values, mask)
                    if messages[-1].get("role") == "assistant" and messages[-1].get("content") != expected:
                        messages[-1]["content"] = expected
                        corrected["assistant_target_aligned_to_mask"] += 1
                    row["messages"] = messages
                    payload = json.dumps(row, ensure_ascii=False) + "\n"
                    writer.write(payload)
                    if rows < 32:
                        smoke_writer.write(payload)
                    rows += 1
    return {"rows": rows, "corrections": dict(corrected)}


def build_quotas(draws: int) -> dict[tuple[str, str, str], int]:
    cohort_counts = equal_counts(draws, list(COHORTS))
    quotas: dict[tuple[str, str, str], int] = {}
    for cohort, cohort_total in cohort_counts.items():
        type_counts = allocate_counts(cohort_total, VIEW_RATIOS)
        for view_type, type_total in type_counts.items():
            for bin_name, count in equal_counts(
                type_total, [cutoff_bin.name for cutoff_bin in CUTOFF_BINS]
            ).items():
                quotas[(cohort, view_type, bin_name)] = count
    if sum(quotas.values()) != draws:
        raise AssertionError("quota allocation does not sum to requested draws")
    return quotas


def audit_row(row: dict[str, Any], metadata: dict[str, Any]) -> None:
    messages = row["messages"]
    user = messages[0]["content"]
    assistant = messages[-1]["content"]
    if any(section in user for section in FORBIDDEN_USER_SECTIONS):
        raise AssertionError("future-derived section leaked into user prompt")
    if any(user.count(token) != 1 for token in QUERY_TOKENS):
        raise AssertionError("query tokens must occur exactly once")
    if int(metadata["window_end_day"]) >= int(row["outcome_targets"][0]):
        raise AssertionError("cutoff must precede delivery")
    if int(metadata["window_start_day"]) < 0:
        raise AssertionError("negative window start")
    if not assistant.startswith("<delivery_days>"):
        raise AssertionError("assistant XML is malformed")
    if len(row["outcome_targets"]) != 3 or len(row["outcome_mask"]) != 3:
        raise AssertionError("outcome vector shape mismatch")


def main() -> None:
    args = parse_args()
    if args.draws <= 0 or args.draws % 2:
        raise ValueError("--draws must be a positive even number")
    if not 0 <= args.content_mask_probability <= 1:
        raise ValueError("content-mask-probability must be in [0, 1]")
    if not 0 < args.content_mask_min <= args.content_mask_max < 1:
        raise ValueError("invalid content masking fractions")
    if not 0 < args.visit_dropout_min <= args.visit_dropout_max < 1:
        raise ValueError("invalid visit dropout fractions")

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    logs_dir = project_root / "logs"
    train_path = data_dir / "train_flexible_regression.jsonl"
    metadata_path = data_dir / "train_flexible_metadata.jsonl"
    smoke_path = data_dir / "smoke_train_regression.jsonl"
    val_path = data_dir / "val_regression.jsonl"
    smoke_val_path = data_dir / "smoke_val_regression.jsonl"
    summary_path = logs_dir / "build_flexible_dataset_summary.json"
    targets = (train_path, metadata_path, smoke_path, val_path, smoke_val_path, summary_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"derived outputs already exist: {existing[:3]}; pass --force")
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    temporal = import_module(
        args.temporal_root.resolve() / "scripts/prepare_temporal_views.py",
        "audited_temporal_v1_builder",
    )
    schedule = import_module(
        args.temporal_root.resolve() / "scripts/build_balanced_schedule.py",
        "audited_temporal_v1_schedule",
    )
    cases, huaxi_test_ids, source_report = load_training_cases(
        temporal, source_root=args.source_root.resolve()
    )
    cases.sort(key=lambda case: (str(case["cohort"]), str(case["case_id"])))
    weights, tail_report = schedule.build_tail_factors(cases, max_weight=3.0)
    index_by_cohort: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        index_by_cohort[str(case["cohort"])].append(index)

    pools: dict[tuple[str, str], WeightedPool] = {}
    for cohort in COHORTS:
        for cutoff_bin in CUTOFF_BINS:
            eligible = [
                index
                for index in index_by_cohort[cohort]
                if eligible_for_bin(cases[index], cutoff_bin)
            ]
            pools[(cohort, cutoff_bin.name)] = WeightedPool(eligible, weights)

    quotas = build_quotas(args.draws)
    remaining = Counter(quotas)
    rng = random.Random(args.seed)
    draw_specs: list[tuple[int, str, CutoffBin]] = []
    covered_cases: set[tuple[str, str]] = set()

    # Hard case coverage uses the flexible-prefix quota, which exceeds the
    # number of cases in each cohort. Select the least-filled eligible cutoff
    # bin for every case before weighted oversampling.
    coverage_order = sorted(
        range(len(cases)),
        key=lambda index: (
            sum(eligible_for_bin(cases[index], cutoff_bin) for cutoff_bin in CUTOFF_BINS),
            stable_int(args.seed, f"coverage:{cases[index]['cohort']}:{cases[index]['case_id']}"),
        ),
    )
    for case_index in coverage_order:
        case = cases[case_index]
        cohort = str(case["cohort"])
        eligible_bins = [
            cutoff_bin
            for cutoff_bin in CUTOFF_BINS
            if eligible_for_bin(case, cutoff_bin)
            and remaining[(cohort, "flexible_prefix", cutoff_bin.name)] > 0
        ]
        if not eligible_bins:
            raise RuntimeError(f"no coverage quota for case {cohort}:{case['case_id']}")
        cutoff_bin = max(
            eligible_bins,
            key=lambda item: (
                remaining[(cohort, "flexible_prefix", item.name)],
                -stable_int(args.seed, f"coverage-bin:{cohort}:{case['case_id']}:{item.name}"),
            ),
        )
        draw_specs.append((case_index, "flexible_prefix", cutoff_bin))
        remaining[(cohort, "flexible_prefix", cutoff_bin.name)] -= 1
        covered_cases.add((cohort, str(case["case_id"])))

    for key in sorted(remaining):
        cohort, view_type, bin_name = key
        cutoff_bin = next(item for item in CUTOFF_BINS if item.name == bin_name)
        for _ in range(remaining[key]):
            draw_specs.append((pools[(cohort, bin_name)].draw(rng), view_type, cutoff_bin))
    if len(draw_specs) != args.draws:
        raise AssertionError(f"built {len(draw_specs)} specs, expected {args.draws}")
    rng.shuffle(draw_specs)

    counters: dict[str, Counter[Any]] = defaultdict(Counter)
    unique_users: set[str] = set()
    unique_windows: set[tuple[int, int]] = set()
    unique_cutoffs: set[int] = set()
    generated = 0
    with train_path.open("w", encoding="utf-8") as train_writer:
        with metadata_path.open("w", encoding="utf-8") as metadata_writer:
            with smoke_path.open("w", encoding="utf-8") as smoke_writer:
                for position, (initial_index, view_type, cutoff_bin) in enumerate(draw_specs):
                    last_error: Exception | None = None
                    case_index = initial_index
                    for attempt in range(100):
                        case = cases[case_index]
                        row_rng = random.Random(
                            stable_int(
                                args.seed,
                                f"draw:{position}:{attempt}:{case['cohort']}:{case['case_id']}:"
                                f"{view_type}:{cutoff_bin.name}",
                            )
                        )
                        try:
                            row, metadata = make_regression_row(
                                temporal,
                                case,
                                position=position,
                                view_type=view_type,
                                cutoff_bin=cutoff_bin,
                                rng=row_rng,
                                huaxi_test_ids=huaxi_test_ids,
                                content_mask_probability=args.content_mask_probability,
                                content_mask_min=args.content_mask_min,
                                content_mask_max=args.content_mask_max,
                                visit_dropout_min=args.visit_dropout_min,
                                visit_dropout_max=args.visit_dropout_max,
                            )
                            audit_row(row, metadata)
                            break
                        except (ValueError, AssertionError) as exc:
                            last_error = exc
                            cohort = str(cases[initial_index]["cohort"])
                            case_index = pools[(cohort, cutoff_bin.name)].draw(row_rng)
                    else:
                        raise RuntimeError(
                            f"failed to generate draw {position} after retries: {last_error}"
                        )

                    payload = json.dumps(row, ensure_ascii=False) + "\n"
                    train_writer.write(payload)
                    metadata_writer.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    if position < 256:
                        smoke_writer.write(payload)
                    generated += 1
                    user = str(row["messages"][0]["content"])
                    unique_users.add(hashlib.sha256(user.encode("utf-8")).hexdigest())
                    start = int(metadata["window_start_day"])
                    end = int(metadata["window_end_day"])
                    unique_windows.add((start, end))
                    unique_cutoffs.add(end)
                    counters["cohort"][metadata["cohort"]] += 1
                    counters["view_type"][metadata["view_type"]] += 1
                    counters["cutoff_bin"][metadata["cutoff_bin"]] += 1
                    counters["content_mask"][str(metadata["content_mask_applied"])] += 1
                    counters["target_reasons"].update(metadata["target_mask_reasons"])
                    counters["window_start_kind"]["prefix" if start == 0 else "local"] += 1

    validation_report = normalize_validation(
        args.regression_v7_root.resolve() / "data/val_regression.jsonl",
        val_path,
        smoke_val_path,
    )
    summary = {
        "status": "complete",
        "protocol_version": "temporal_outcome_regression_flexible_v1_20260806",
        "seed": args.seed,
        "draws": generated,
        "source": source_report,
        "case_coverage": {
            "covered": len(covered_cases),
            "training_cases": len(cases),
            "complete": len(covered_cases) == len(cases),
        },
        "view_ratios_requested": VIEW_RATIOS,
        "cutoff_bins": [cutoff_bin.__dict__ for cutoff_bin in CUTOFF_BINS],
        "quota": {":".join(key): value for key, value in sorted(quotas.items())},
        "realized": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(counters.items())
        },
        "diversity": {
            "unique_user_prompts": len(unique_users),
            "unique_user_prompt_rate": len(unique_users) / generated,
            "unique_window_boundaries": len(unique_windows),
            "unique_cutoff_days": len(unique_cutoffs),
            "minimum_cutoff": min(unique_cutoffs),
            "maximum_cutoff": max(unique_cutoffs),
        },
        "augmentation": {
            "content_mask_probability": args.content_mask_probability,
            "content_mask_fraction_range": [args.content_mask_min, args.content_mask_max],
            "visit_dropout_fraction_range": [args.visit_dropout_min, args.visit_dropout_max],
            "content_mask_scope": "non-header visit field values only; replaced with NA",
        },
        "tail_balance": tail_report,
        "validation": validation_report,
        "outputs": {
            "train": str(train_path),
            "metadata": str(metadata_path),
            "smoke_train": str(smoke_path),
            "validation": str(val_path),
            "smoke_validation": str(smoke_val_path),
        },
        "invariants": {
            "train_internal_validation_split_reused_exactly": True,
            "huaxi_seen_test_rows_appended": 0,
            "gt_hint_removed": True,
            "postdelivery_records_removed": True,
            "cutoff_strictly_before_delivery": True,
            "query_tokens_are_label_free": True,
            "ce_and_regression_optional_target_masks_aligned": True,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
