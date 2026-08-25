#!/usr/bin/env python3
"""Build v2 training data with one unified continuous-cutoff policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from temporal_outcome.data import balancing, flexible, temporal_views

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 20260809
TOTAL_DRAWS = 80_000
COHORTS = ("huaxi", "shenzhen")
CONDITION_RATIOS = {
    "clean_prefix": 0.60,
    "content_mask": 0.20,
    "visit_dropout": 0.10,
    "local_window": 0.10,
}
ANCHOR_CENTERED_PROBABILITY = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Private directory containing huaxi/ and shenzhen/ source JSONL files.",
    )
    parser.add_argument("--draws", type=int, default=TOTAL_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--anchor-centered-probability",
        type=float,
        default=ANCHOR_CENTERED_PROBABILITY,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    result = {name: int(math.floor(value)) for name, value in raw.items()}
    order = sorted(raw, key=lambda name: (raw[name] - result[name], name), reverse=True)
    for name in order[: total - sum(result.values())]:
        result[name] += 1
    return result


def equal_counts(total: int, labels: Sequence[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(labels))
    return {
        label: base + int(index < remainder)
        for index, label in enumerate(labels)
    }


def build_quotas(flexible: Any, draws: int) -> dict[tuple[str, str, str], int]:
    quotas: dict[tuple[str, str, str], int] = {}
    for cohort, cohort_total in equal_counts(draws, list(COHORTS)).items():
        condition_counts = allocate_counts(cohort_total, CONDITION_RATIOS)
        for condition, condition_total in condition_counts.items():
            bin_counts = equal_counts(
                condition_total, [item.name for item in flexible.CUTOFF_BINS]
            )
            for bin_name, count in bin_counts.items():
                quotas[(cohort, condition, bin_name)] = count
    if sum(quotas.values()) != draws:
        raise AssertionError("quota allocation does not match requested draws")
    return quotas


def choose_continuous_cutoff(
    flexible: Any,
    case: dict[str, Any],
    cutoff_bin: Any,
    rng: random.Random,
    anchor_centered_probability: float,
) -> tuple[int, str]:
    delivery = int(case["targets"]["delivery_days"])
    minimum = max(cutoff_bin.low, flexible.min_block_end(case))
    maximum = min(cutoff_bin.high, delivery - 1)
    if minimum > maximum:
        raise ValueError("case is not eligible for cutoff bin")
    if rng.random() < anchor_centered_probability:
        mode = min(max(cutoff_bin.anchor, minimum), maximum)
        cutoff = int(round(rng.triangular(minimum, maximum, mode)))
        sampling_mode = "anchor_centered_triangular"
    else:
        cutoff = rng.randint(minimum, maximum)
        sampling_mode = "uniform_within_bin"
    return min(max(cutoff, minimum), maximum), sampling_mode


def visit_dropout(
    temporal: Any,
    blocks: Sequence[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = temporal.encounter_groups(blocks)
    if len(groups) <= 1:
        raise ValueError("visit dropout requires at least two encounters")
    fraction = rng.uniform(0.10, 0.20)
    remove_count = min(
        len(groups) - 1,
        max(1, int(round(len(groups) * fraction))),
    )
    removed = set(rng.sample(range(len(groups)), remove_count))
    retained = [
        block
        for group_index, group in enumerate(groups)
        if group_index not in removed
        for block in group
    ]
    return retained, {
        "dropout_fraction_requested": round(fraction, 6),
        "source_visit_count_before_dropout": len(groups),
        "removed_visit_count": remove_count,
        "removed_visit_end_days": sorted(
            {
                int(block["end_day"])
                for group_index, group in enumerate(groups)
                if group_index in removed
                for block in group
            }
        ),
    }


def generate_blocks(
    temporal: Any,
    flexible: Any,
    case: dict[str, Any],
    condition: str,
    cutoff_bin: Any,
    rng: random.Random,
    anchor_centered_probability: float,
) -> tuple[int, int, list[dict[str, Any]], dict[str, Any]]:
    cutoff, sampling_mode = choose_continuous_cutoff(
        flexible,
        case,
        cutoff_bin,
        rng,
        anchor_centered_probability,
    )
    prefix = temporal.visible_blocks(case, window_start_day=0, cutoff_day=cutoff)
    if not prefix:
        raise ValueError("empty continuous prefix")
    metadata: dict[str, Any] = {
        "augmentation_type": condition,
        "cutoff_sampling_mode": sampling_mode,
    }
    if condition == "clean_prefix":
        return 0, cutoff, [dict(block) for block in prefix], metadata
    if condition == "content_mask":
        masked, mask_meta = flexible.mask_visit_content(
            prefix,
            rng=rng,
            minimum=0.20,
            maximum=0.30,
        )
        if not mask_meta["content_mask_applied"]:
            raise ValueError("content mask has no eligible fields")
        metadata.update(mask_meta)
        return 0, cutoff, masked, metadata
    if condition == "visit_dropout":
        dropped, dropout_meta = visit_dropout(temporal, prefix, rng)
        metadata.update(dropout_meta)
        return 0, cutoff, dropped, metadata
    if condition == "local_window":
        start, local = flexible.local_window_blocks(temporal, case, cutoff, rng)
        return start, cutoff, local, metadata
    raise ValueError(f"unsupported augmentation condition: {condition}")


def make_row(
    temporal: Any,
    flexible: Any,
    case: dict[str, Any],
    *,
    position: int,
    condition: str,
    cutoff_bin: Any,
    rng: random.Random,
    huaxi_test_ids: set[str],
    anchor_centered_probability: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start, cutoff, blocks, augmentation = generate_blocks(
        temporal,
        flexible,
        case,
        condition,
        cutoff_bin,
        rng,
        anchor_centered_probability,
    )
    view_type = (
        "continuous_local_window"
        if condition == "local_window"
        else "continuous_prefix"
    )
    view_name = f"continuous_v2_{position:06d}_{condition}_d{start}_{cutoff}"
    source_view = temporal.make_view(
        case,
        blocks,
        split="train",
        view_type=view_type,
        view_name=view_name,
        stage=cutoff_bin.name,
        window_start_day=start,
        cutoff_day=cutoff,
        is_huaxi_seen_test=(
            case["cohort"] == "huaxi" and str(case["case_id"]) in huaxi_test_ids
        ),
        extra=augmentation,
    )
    values, mask, reasons = flexible.validated_targets(case)
    messages = [dict(message) for message in source_view["messages"]]
    messages[0]["content"] = str(messages[0]["content"]) + flexible.QUERY_SUFFIX
    messages[1]["content"] = flexible.target_response(values, mask)
    row = {
        "messages": messages,
        "outcome_targets": values,
        "outcome_mask": mask,
    }
    metadata_keys = (
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
    metadata = {key: source_view[key] for key in metadata_keys}
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
    flexible.audit_row(row, metadata)
    return row, metadata


def build_continuous_validation(
    temporal: Any,
    flexible: Any,
    cases: Sequence[dict[str, Any]],
    output: Path,
    metadata_output: Path,
    smoke: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    rows = 0
    cohorts = Counter()
    bins = Counter()
    unique_cases = set()
    slots_per_case: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as writer:
        with metadata_output.open("w", encoding="utf-8") as metadata_writer:
            with smoke.open("w", encoding="utf-8") as smoke_writer:
                ordered = sorted(
                    cases, key=lambda item: (str(item["cohort"]), str(item["case_id"]))
                )
                for case in ordered:
                    case_key = f"{case['cohort']}:{case['case_id']}"
                    for slot, cutoff_bin in enumerate(flexible.CUTOFF_BINS):
                        delivery = int(case["targets"]["delivery_days"])
                        minimum = max(cutoff_bin.low, flexible.min_block_end(case))
                        maximum = min(cutoff_bin.high, delivery - 1)
                        candidates = [
                            day
                            for day in range(minimum, maximum + 1)
                            if day != cutoff_bin.anchor
                        ]
                        if not candidates:
                            raise ValueError(
                                f"held-out case {case_key} has no non-anchor cutoff "
                                f"for {cutoff_bin.name}"
                            )
                        selector = flexible.stable_int(
                            seed, f"validation:{case_key}:{cutoff_bin.name}"
                        )
                        cutoff = candidates[selector % len(candidates)]
                        blocks = temporal.visible_blocks(
                            case, window_start_day=0, cutoff_day=cutoff
                        )
                        source_view = temporal.make_view(
                            case,
                            blocks,
                            split="internal_val",
                            view_type="continuous_prefix",
                            view_name=f"val_{case_key}_{cutoff_bin.name}_d{cutoff}",
                            stage=cutoff_bin.name,
                            window_start_day=0,
                            cutoff_day=cutoff,
                            is_huaxi_seen_test=False,
                            extra={
                                "eval_condition": "clean_prefix",
                                "continuous_slot": slot,
                                "cutoff_bin": cutoff_bin.name,
                                "cutoff_sampling_mode": "deterministic_non_anchor",
                            },
                        )
                        values, mask, reasons = flexible.validated_targets(case)
                        messages = [dict(message) for message in source_view["messages"]]
                        messages[0]["content"] = (
                            str(messages[0]["content"]) + flexible.QUERY_SUFFIX
                        )
                        messages[1]["content"] = flexible.target_response(values, mask)
                        row = {
                            "messages": messages,
                            "outcome_targets": values,
                            "outcome_mask": mask,
                        }
                        metadata_keys = (
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
                        metadata = {key: source_view[key] for key in metadata_keys}
                        metadata.update(
                            {
                                "eval_condition": "clean_prefix",
                                "continuous_slot": slot,
                                "cutoff_bin": cutoff_bin.name,
                                "cutoff_sampling_mode": "deterministic_non_anchor",
                                "target_mask_reasons": reasons,
                                "outcome_targets": values,
                                "outcome_mask": mask,
                            }
                        )
                        flexible.audit_row(row, metadata)
                        row_line = json.dumps(row, ensure_ascii=False) + "\n"
                        writer.write(row_line)
                        metadata_writer.write(
                            json.dumps(metadata, ensure_ascii=False) + "\n"
                        )
                        if rows < 32:
                            smoke_writer.write(row_line)
                        rows += 1
                        slots_per_case[case_key] += 1
                        cohorts[str(case["cohort"])] += 1
                        bins[cutoff_bin.name] += 1
                        unique_cases.add(case_key)
    incomplete = [key for key, count in slots_per_case.items() if count != 5]
    if incomplete:
        raise AssertionError(f"validation cases without five slots: {incomplete[:5]}")
    return {
        "rows": rows,
        "unique_cases": len(unique_cases),
        "cohorts": dict(cohorts),
        "cutoff_bins": dict(bins),
        "source": "deterministically generated from the held-out case partition",
        "all_cutoffs_non_landmark": True,
    }


def main() -> None:
    args = parse_args()
    if args.draws <= 0 or args.draws % 2:
        raise ValueError("--draws must be a positive even number")
    if not 0 <= args.anchor_centered_probability <= 1:
        raise ValueError("anchor-centered probability must be in [0, 1]")
    root = args.project_root.resolve()
    data_dir = root / "data"
    logs_dir = root / "logs"
    train_path = data_dir / "train_continuous_v2_regression.jsonl"
    metadata_path = data_dir / "train_continuous_v2_metadata.jsonl"
    smoke_path = data_dir / "smoke_train_regression.jsonl"
    val_path = data_dir / "val_continuous_clean_regression.jsonl"
    val_metadata_path = data_dir / "val_continuous_clean_metadata.jsonl"
    smoke_val_path = data_dir / "smoke_val_regression.jsonl"
    summary_path = logs_dir / "build_continuous_v2_summary.json"
    outputs = (
        train_path,
        metadata_path,
        smoke_path,
        val_path,
        val_metadata_path,
        smoke_val_path,
        summary_path,
    )
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("v2 outputs exist; pass --force")
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cases, validation_cases, huaxi_test_ids, source_report = (
        flexible.load_training_and_validation_cases(
            temporal_views, source_root=args.source_root.resolve()
        )
    )
    cases.sort(key=lambda case: (str(case["cohort"]), str(case["case_id"])))
    weights, tail_report = balancing.build_tail_factors(cases, max_weight=3.0)
    indices_by_cohort: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        indices_by_cohort[str(case["cohort"])].append(index)

    pools: dict[tuple[str, str], Any] = {}
    for cohort in COHORTS:
        for cutoff_bin in flexible.CUTOFF_BINS:
            eligible = [
                index
                for index in indices_by_cohort[cohort]
                if flexible.eligible_for_bin(cases[index], cutoff_bin)
            ]
            pools[(cohort, cutoff_bin.name)] = flexible.WeightedPool(eligible, weights)

    quotas = build_quotas(flexible, args.draws)
    remaining = Counter(quotas)
    draw_specs: list[tuple[int, str, Any]] = []
    covered_cases: set[tuple[str, str]] = set()
    coverage_order = sorted(
        range(len(cases)),
        key=lambda index: (
            sum(
                flexible.eligible_for_bin(cases[index], cutoff_bin)
                for cutoff_bin in flexible.CUTOFF_BINS
            ),
            flexible.stable_int(
                args.seed,
                f"coverage:{cases[index]['cohort']}:{cases[index]['case_id']}",
            ),
        ),
    )
    for case_index in coverage_order:
        case = cases[case_index]
        cohort = str(case["cohort"])
        eligible_bins = [
            cutoff_bin
            for cutoff_bin in flexible.CUTOFF_BINS
            if flexible.eligible_for_bin(case, cutoff_bin)
            and remaining[(cohort, "clean_prefix", cutoff_bin.name)] > 0
        ]
        if not eligible_bins:
            raise RuntimeError(f"no clean coverage quota for {cohort}:{case['case_id']}")
        cutoff_bin = max(
            eligible_bins,
            key=lambda item: (
                remaining[(cohort, "clean_prefix", item.name)],
                -flexible.stable_int(
                    args.seed,
                    f"coverage-bin:{cohort}:{case['case_id']}:{item.name}",
                ),
            ),
        )
        draw_specs.append((case_index, "clean_prefix", cutoff_bin))
        remaining[(cohort, "clean_prefix", cutoff_bin.name)] -= 1
        covered_cases.add((cohort, str(case["case_id"])))

    schedule_rng = random.Random(args.seed)
    for key in sorted(remaining):
        cohort, condition, bin_name = key
        cutoff_bin = next(item for item in flexible.CUTOFF_BINS if item.name == bin_name)
        for _ in range(remaining[key]):
            draw_specs.append(
                (pools[(cohort, bin_name)].draw(schedule_rng), condition, cutoff_bin)
            )
    if len(draw_specs) != args.draws:
        raise AssertionError("draw schedule length mismatch")
    schedule_rng.shuffle(draw_specs)

    counters: dict[str, Counter[Any]] = defaultdict(Counter)
    unique_users: set[str] = set()
    unique_cutoffs: set[int] = set()
    unique_windows: set[tuple[int, int]] = set()
    failures = Counter()
    with train_path.open("w", encoding="utf-8") as train_writer:
        with metadata_path.open("w", encoding="utf-8") as metadata_writer:
            with smoke_path.open("w", encoding="utf-8") as smoke_writer:
                for position, (initial_index, condition, cutoff_bin) in enumerate(draw_specs):
                    case_index = initial_index
                    last_error: Exception | None = None
                    for attempt in range(100):
                        case = cases[case_index]
                        row_rng = random.Random(
                            flexible.stable_int(
                                args.seed,
                                f"draw:{position}:{attempt}:{case['cohort']}:"
                                f"{case['case_id']}:{condition}:{cutoff_bin.name}",
                            )
                        )
                        try:
                            row, metadata = make_row(
                                temporal_views,
                                flexible,
                                case,
                                position=position,
                                condition=condition,
                                cutoff_bin=cutoff_bin,
                                rng=row_rng,
                                huaxi_test_ids=huaxi_test_ids,
                                anchor_centered_probability=args.anchor_centered_probability,
                            )
                            break
                        except (ValueError, AssertionError) as exc:
                            last_error = exc
                            failures[f"{condition}:{type(exc).__name__}"] += 1
                            cohort = str(cases[initial_index]["cohort"])
                            case_index = pools[(cohort, cutoff_bin.name)].draw(row_rng)
                    else:
                        raise RuntimeError(
                            f"failed draw {position} after retries: {last_error}"
                        )
                    payload = json.dumps(row, ensure_ascii=False) + "\n"
                    train_writer.write(payload)
                    metadata_writer.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    if position < 256:
                        smoke_writer.write(payload)
                    user = str(row["messages"][0]["content"])
                    unique_users.add(hashlib.sha256(user.encode("utf-8")).hexdigest())
                    start = int(metadata["window_start_day"])
                    cutoff = int(metadata["window_end_day"])
                    unique_cutoffs.add(cutoff)
                    unique_windows.add((start, cutoff))
                    counters["cohort"][metadata["cohort"]] += 1
                    counters["augmentation_type"][metadata["augmentation_type"]] += 1
                    counters["cutoff_bin"][metadata["cutoff_bin"]] += 1
                    counters["cutoff_sampling_mode"][metadata["cutoff_sampling_mode"]] += 1
                    counters["view_type"][metadata["view_type"]] += 1
                    counters["target_reasons"].update(metadata["target_mask_reasons"])

    validation_report = build_continuous_validation(
        temporal_views,
        flexible,
        validation_cases,
        val_path,
        val_metadata_path,
        smoke_val_path,
        seed=args.seed,
    )
    anchors = {item.anchor for item in flexible.CUTOFF_BINS}
    summary = {
        "status": "complete",
        "protocol": "continuous_cutoff_v2_20260809",
        "seed": args.seed,
        "draws": args.draws,
        "source": source_report,
        "case_coverage": {
            "covered": len(covered_cases),
            "training_cases": len(cases),
            "complete": len(covered_cases) == len(cases),
        },
        "condition_ratios_requested": CONDITION_RATIOS,
        "anchor_centered_probability": args.anchor_centered_probability,
        "cutoff_bins": [item.__dict__ for item in flexible.CUTOFF_BINS],
        "realized": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(counters.items())
        },
        "diversity": {
            "unique_user_prompts": len(unique_users),
            "unique_user_prompt_rate": len(unique_users) / args.draws,
            "unique_window_boundaries": len(unique_windows),
            "unique_cutoff_days": len(unique_cutoffs),
            "minimum_cutoff": min(unique_cutoffs),
            "maximum_cutoff": max(unique_cutoffs),
            "anchor_cutoff_rows": sum(
                1
                for line in metadata_path.open(encoding="utf-8")
                if int(json.loads(line)["window_end_day"]) in anchors
            ),
        },
        "generation_failures_retried": dict(failures),
        "tail_balance": tail_report,
        "validation": validation_report,
        "outputs": {
            "train": str(train_path),
            "metadata": str(metadata_path),
            "smoke_train": str(smoke_path),
            "validation": str(val_path),
            "validation_metadata": str(val_metadata_path),
            "smoke_validation": str(smoke_val_path),
        },
        "invariants": {
            "single_unified_continuous_cutoff_policy": True,
            "separate_landmark_view_type": False,
            "anchors_only_shape_sampling_density": True,
            "held_out_validation_only": True,
            "huaxi_test_train_seen_policy_preserved": True,
            "gt_hint_removed": True,
            "postdelivery_records_removed": True,
            "cutoff_strictly_before_delivery": True,
            "model_and_loss_unchanged_from_v1": True,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
