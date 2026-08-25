#!/usr/bin/env python3
"""Build a deterministic, oversampled SFT draw schedule from temporal views.

Balancing is marginal rather than a Cartesian product:

* Huaxi/Shenzhen receive approximately equal draws.
* Canonical stages are uniform within cohort; local windows are uniform over
  the stages for which they exist.
* Canonical/local/dropout views default to a 70/15/15 mixture.
* Delivery, birth-weight, and birth-length tails contribute a geometric-mean
  sampling factor capped at 3, avoiding domination by tiny joint cells.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

PROTOCOL_VERSION = "temporal_balanced_schedule_v1_20260727"
DEFAULT_SEED = 20260727
STAGE_ORDER = ("w13", "w22", "w28", "w32", "w36")
VIEW_TYPE_RATIOS = {
    "canonical_prefix": 0.70,
    "local_window": 0.15,
    "visit_bundle_dropout": 0.15,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overfit-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=60000)
    parser.add_argument("--overfit-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-tail-weight", type=float, default=3.0)
    parser.add_argument("--canonical-ratio", type=float, default=0.70)
    parser.add_argument("--local-window-ratio", type=float, default=0.15)
    parser.add_argument("--dropout-ratio", type=float, default=0.15)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(seed: int, text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()[:8], "big"
    )


def delivery_bin(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    if value < 259:
        return "early_lt259"
    if value > 287:
        return "late_gt287"
    return "middle_259_287"


def weight_bin(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    if value < 2500:
        return "low"
    if value >= 4000:
        return "high"
    return "normal"


def length_bin(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    if value < 48:
        return "short"
    if value > 52:
        return "long"
    return "middle"


BIN_FUNCTIONS = {
    "delivery_days": delivery_bin,
    "birth_weight_g": weight_bin,
    "birth_length_cm": length_bin,
}


def row_targets(row: dict[str, Any]) -> dict[str, int | float | None]:
    targets = row.get("targets")
    if isinstance(targets, dict):
        return {
            name: targets.get(name)
            for name in ("delivery_days", "birth_weight_g", "birth_length_cm")
        }
    return {
        "delivery_days": row.get("actual_delivery_days"),
        "birth_weight_g": row.get("actual_birth_weight_g"),
        "birth_length_cm": row.get("actual_birth_length_cm"),
    }


def unique_case_targets(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, int | float | None]]:
    result: dict[tuple[str, str], dict[str, int | float | None]] = {}
    for row in rows:
        key = (str(row["cohort"]), str(row["case_id"]))
        targets = row_targets(row)
        if key in result and result[key] != targets:
            raise AssertionError(f"Targets differ across views for case {key}")
        result[key] = targets
    return result


def build_tail_factors(
    rows: Sequence[dict[str, Any]],
    *,
    max_weight: float,
) -> tuple[list[float], dict[str, Any]]:
    if max_weight < 1:
        raise ValueError("max-tail-weight must be >= 1")
    cases = unique_case_targets(rows)
    counts: Counter[tuple[str, str, str]] = Counter()
    for (cohort, _), targets in cases.items():
        for target_name, function in BIN_FUNCTIONS.items():
            category = function(targets.get(target_name))
            if category != "unknown":
                counts[(cohort, target_name, category)] += 1

    maximums: dict[tuple[str, str], int] = {}
    for (cohort, target_name, _), count in counts.items():
        maximums[(cohort, target_name)] = max(
            count, maximums.get((cohort, target_name), 0)
        )

    factor_table: dict[tuple[str, str, str], float] = {}
    for key, count in counts.items():
        cohort, target_name, _ = key
        factor_table[key] = min(
            max_weight,
            math.sqrt(maximums[(cohort, target_name)] / max(count, 1)),
        )

    weights: list[float] = []
    for row in rows:
        cohort = str(row["cohort"])
        factors: list[float] = []
        for target_name, function in BIN_FUNCTIONS.items():
            category = function(row_targets(row).get(target_name))
            if category == "unknown":
                continue
            factors.append(factor_table[(cohort, target_name, category)])
        if not factors:
            weight = 1.0
        else:
            weight = math.prod(factors) ** (1.0 / len(factors))
        weights.append(min(max_weight, max(1.0, weight)))

    report_counts: dict[str, dict[str, int]] = defaultdict(dict)
    report_factors: dict[str, dict[str, float]] = defaultdict(dict)
    for (cohort, target, category), count in sorted(counts.items()):
        key = f"{cohort}:{target}"
        report_counts[key][category] = count
        report_factors[key][category] = round(
            factor_table[(cohort, target, category)], 6
        )
    return weights, {
        "marginal_case_counts": dict(report_counts),
        "marginal_tail_factors": dict(report_factors),
        "combination": "geometric_mean",
        "maximum_weight": max_weight,
    }


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    positive = {key: value for key, value in ratios.items() if value > 0}
    denominator = sum(positive.values())
    if denominator <= 0:
        raise ValueError("At least one positive ratio is required")
    exact = {key: total * value / denominator for key, value in positive.items()}
    result = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(
        positive,
        key=lambda key: (exact[key] - result[key], key),
        reverse=True,
    )
    for key in order[:remainder]:
        result[key] += 1
    return result


def equal_counts(total: int, labels: Sequence[str]) -> dict[str, int]:
    if not labels:
        raise ValueError("Cannot allocate over an empty label set")
    return allocate_counts(total, {label: 1.0 for label in labels})


def weighted_choice_index(
    pool: Sequence[int],
    weights: Sequence[float],
    rng: random.Random,
) -> int:
    if not pool:
        raise ValueError("Cannot sample an empty pool")
    cumulative: list[float] = []
    running = 0.0
    for index in pool:
        running += weights[index]
        cumulative.append(running)
    position = rng.random() * running
    selected = bisect.bisect_right(cumulative, position)
    return pool[min(selected, len(pool) - 1)]


def normalized_view_ratios(
    canonical: float,
    local_window: float,
    dropout: float,
) -> dict[str, float]:
    ratios = {
        "canonical_prefix": canonical,
        "local_window": local_window,
        "visit_bundle_dropout": dropout,
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("View-type ratios must be non-negative")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("View-type ratios must sum to 1")
    return ratios


def build_draw_specs(
    rows: Sequence[dict[str, Any]],
    weights: Sequence[float],
    *,
    draws: int,
    seed: int,
    view_ratios: dict[str, float],
) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    if draws <= 0:
        raise ValueError("draws must be > 0")
    if len(rows) != len(weights):
        raise ValueError("rows and weights lengths differ")

    pools: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (str(row["view_type"]), str(row["cohort"]), str(row["stage"]))
        pools[key].append(index)

    available_types = {key[0] for key in pools}
    requested_missing = {
        view_type
        for view_type, ratio in view_ratios.items()
        if ratio > 0 and view_type not in available_types
    }
    if requested_missing:
        raise ValueError(f"Requested view types are absent: {sorted(requested_missing)}")

    rng = random.Random(seed)
    requested_counts = allocate_counts(draws, view_ratios)
    target_allocation: Counter[tuple[str, str, str]] = Counter()
    for view_type, type_count in requested_counts.items():
        cohorts = sorted(
            {cohort for candidate_type, cohort, _ in pools if candidate_type == view_type}
        )
        cohort_counts = equal_counts(type_count, cohorts)
        for cohort, cohort_count in cohort_counts.items():
            stages = [
                stage
                for stage in STAGE_ORDER
                if pools.get((view_type, cohort, stage))
            ]
            extra_stages = sorted(
                {
                    stage
                    for candidate_type, candidate_cohort, stage in pools
                    if candidate_type == view_type
                    and candidate_cohort == cohort
                    and stage not in STAGE_ORDER
                }
            )
            stages.extend(extra_stages)
            for stage, stage_count in equal_counts(cohort_count, stages).items():
                target_allocation[(view_type, cohort, stage)] = stage_count

    # Coverage is a hard constraint, not a probabilistic side effect of
    # oversampling. Assign every training case one canonical draw first, while
    # respecting the already-computed cohort/stage quotas. Remaining slots are
    # then sampled with replacement using the tail weights.
    all_case_keys = {
        (str(row["cohort"]), str(row["case_id"]))
        for row in rows
    }
    canonical_by_case_stage: dict[
        tuple[str, str], dict[str, list[int]]
    ] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if str(row["view_type"]) != "canonical_prefix":
            continue
        case_key = (str(row["cohort"]), str(row["case_id"]))
        canonical_by_case_stage[case_key][str(row["stage"])].append(index)

    cases_without_canonical = sorted(all_case_keys - set(canonical_by_case_stage))
    if cases_without_canonical:
        raise ValueError(
            "Every scheduled training case must have a canonical view; "
            f"missing={cases_without_canonical[:5]}"
        )

    draw_specs: list[tuple[int, float]] = []
    allocation_audit: Counter[tuple[str, str, str]] = Counter()
    coverage_case_keys: set[tuple[str, str]] = set()
    cohort_case_counts = Counter(cohort for cohort, _ in all_case_keys)
    cohort_canonical_quotas = Counter(
        {
            cohort: sum(
                count
                for (view_type, candidate_cohort, _), count in target_allocation.items()
                if view_type == "canonical_prefix"
                and candidate_cohort == cohort
            )
            for cohort in cohort_case_counts
        }
    )
    insufficient = {
        cohort: {
            "cases": cohort_case_counts[cohort],
            "canonical_quota": cohort_canonical_quotas[cohort],
        }
        for cohort in cohort_case_counts
        if cohort_canonical_quotas[cohort] < cohort_case_counts[cohort]
    }
    if insufficient:
        raise ValueError(
            "Canonical draw quota is too small to cover every case: "
            f"{insufficient}"
        )

    ordered_cases = sorted(
        all_case_keys,
        key=lambda case_key: (
            len(canonical_by_case_stage[case_key]),
            stable_int(seed, f"coverage-case:{case_key[0]}:{case_key[1]}"),
            case_key,
        ),
    )
    for cohort, case_id in ordered_cases:
        case_key = (cohort, case_id)
        eligible_stages = [
            stage
            for stage in canonical_by_case_stage[case_key]
            if allocation_audit[("canonical_prefix", cohort, stage)]
            < target_allocation[("canonical_prefix", cohort, stage)]
        ]
        if not eligible_stages:
            raise ValueError(
                "Canonical stage quotas cannot cover every case; "
                f"failed_case={case_key}"
            )
        stage = min(
            eligible_stages,
            key=lambda candidate_stage: (
                allocation_audit[
                    ("canonical_prefix", cohort, candidate_stage)
                ]
                / target_allocation[
                    ("canonical_prefix", cohort, candidate_stage)
                ],
                stable_int(
                    seed,
                    f"coverage-stage:{cohort}:{case_id}:{candidate_stage}",
                ),
                candidate_stage,
            ),
        )
        candidates = canonical_by_case_stage[case_key][stage]
        index = min(
            candidates,
            key=lambda candidate_index: (
                stable_int(
                    seed,
                    "coverage-view:"
                    + str(
                        rows[candidate_index].get("view_id")
                        or rows[candidate_index].get("sample_id")
                    ),
                ),
                candidate_index,
            ),
        )
        draw_specs.append((index, weights[index]))
        allocation_audit[("canonical_prefix", cohort, stage)] += 1
        coverage_case_keys.add(case_key)

    for key, target_count in sorted(target_allocation.items()):
        pool = pools[key]
        remaining = target_count - allocation_audit[key]
        if remaining < 0:
            raise AssertionError(
                f"Mandatory coverage exceeded allocation for {key}: "
                f"{allocation_audit[key]} > {target_count}"
            )
        for _ in range(remaining):
            index = weighted_choice_index(pool, weights, rng)
            draw_specs.append((index, weights[index]))
            allocation_audit[key] += 1

    if len(draw_specs) != draws:
        raise AssertionError(f"Allocated {len(draw_specs)} rows, expected {draws}")
    if coverage_case_keys != all_case_keys:
        raise AssertionError(
            f"Coverage audit failed: {len(coverage_case_keys)} of "
            f"{len(all_case_keys)} cases"
        )
    rng.shuffle(draw_specs)
    return draw_specs, {
        "requested_view_type_counts": requested_counts,
        "allocated_strata": {
            ":".join(key): count for key, count in sorted(allocation_audit.items())
        },
        "mandatory_canonical_coverage": {
            "covered_cases": len(coverage_case_keys),
            "input_cases": len(all_case_keys),
            "complete": coverage_case_keys == all_case_keys,
            "covered_cases_by_cohort": dict(
                sorted(Counter(cohort for cohort, _ in coverage_case_keys).items())
            ),
        },
    }

def scheduled_row(
    base: dict[str, Any],
    *,
    position: int,
    sampling_weight: float,
    seed: int,
) -> dict[str, Any]:
    row = dict(base)
    base_view_id = str(base.get("view_id") or base.get("sample_id"))
    draw_id = f"draw_{position:06d}_{stable_int(seed, base_view_id + ':' + str(position)):016x}"
    row["base_view_id"] = base_view_id
    row["sample_id"] = draw_id
    row["schedule_draw_id"] = draw_id
    row["schedule_position"] = position
    row["sampling_tail_weight"] = round(float(sampling_weight), 6)
    row["schedule_protocol_version"] = PROTOCOL_VERSION
    return row


def write_schedule(
    path: Path,
    rows: Sequence[dict[str, Any]],
    specs: Sequence[tuple[int, float]],
    *,
    seed: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cohorts: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    view_types: Counter[str] = Counter()
    base_views: set[str] = set()
    cases: set[tuple[str, str]] = set()
    with path.open("w", encoding="utf-8") as handle:
        for position, (index, weight) in enumerate(specs):
            row = scheduled_row(
                rows[index],
                position=position,
                sampling_weight=weight,
                seed=seed,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            cohorts[str(row["cohort"])] += 1
            stages[str(row["stage"])] += 1
            view_types[str(row["view_type"])] += 1
            base_views.add(str(row["base_view_id"]))
            cases.add((str(row["cohort"]), str(row["case_id"])))
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "draws": len(specs),
        "unique_base_views": len(base_views),
        "unique_cases": len(cases),
        "cohorts": dict(sorted(cohorts.items())),
        "stages": dict(sorted(stages.items())),
        "view_types": dict(sorted(view_types.items())),
    }


def build_overfit_specs(
    schedule_specs: Sequence[tuple[int, float]],
    rows: Sequence[dict[str, Any]],
    *,
    size: int,
    seed: int,
) -> list[tuple[int, float]]:
    if size <= 0:
        return []
    unique: dict[str, tuple[int, float]] = {}
    for index, weight in schedule_specs:
        view_id = str(rows[index].get("view_id") or rows[index].get("sample_id"))
        unique.setdefault(view_id, (index, weight))
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            stable_int(
                seed + 256,
                str(rows[item[0]].get("view_id") or rows[item[0]].get("sample_id")),
            ),
            str(rows[item[0]].get("view_id") or rows[item[0]].get("sample_id")),
        ),
    )
    return ordered[: min(size, len(ordered))]


def target_scaling(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    cases = unique_case_targets(rows)
    result: dict[str, dict[str, float | int]] = {}
    for target_name in ("delivery_days", "birth_weight_g", "birth_length_cm"):
        values = [
            float(targets[target_name])
            for targets in cases.values()
            if targets.get(target_name) is not None
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        result[target_name] = {
            "count": len(values),
            "mean": mean,
            "std": max(math.sqrt(variance), 1e-6),
            "median": median,
            "minimum": min(values),
            "maximum": max(values),
        }
    return result


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_path = (
        args.input.resolve()
        if args.input is not None
        else project_root / "data" / "temporal_views" / "train_views.jsonl"
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else project_root / "data" / "train_schedule" / "train_balanced.jsonl"
    )
    overfit_path = (
        args.overfit_output.resolve()
        if args.overfit_output is not None
        else project_root / "data" / "train_schedule" / "overfit_256.jsonl"
    )
    summary_path = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else project_root / "logs" / "build_balanced_schedule_summary.json"
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("Input temporal view file is empty")
    required = {"messages", "view_id", "case_id", "cohort", "stage", "view_type"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Input schema is missing fields: {missing}")

    ratios = normalized_view_ratios(
        args.canonical_ratio,
        args.local_window_ratio,
        args.dropout_ratio,
    )
    weights, tail_report = build_tail_factors(
        rows, max_weight=args.max_tail_weight
    )
    specs, allocation_report = build_draw_specs(
        rows,
        weights,
        draws=args.draws,
        seed=args.seed,
        view_ratios=ratios,
    )
    schedule_summary = write_schedule(
        output_path, rows, specs, seed=args.seed
    )
    overfit_specs = build_overfit_specs(
        specs, rows, size=args.overfit_size, seed=args.seed
    )
    overfit_summary = write_schedule(
        overfit_path, rows, overfit_specs, seed=args.seed + 256
    )

    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "seed": args.seed,
        "input": {
            "path": str(input_path),
            "sha256": file_sha256(input_path),
            "views": len(rows),
            "unique_cases": len(unique_case_targets(rows)),
        },
        "requested_draws": args.draws,
        "view_type_ratios": ratios,
        "tail_balance": tail_report,
        "allocation": allocation_report,
        "numeric_target_scaling": target_scaling(rows),
        "schedule": schedule_summary,
        "overfit": overfit_summary,
        "training_contract": {
            "schedule_order_required": False,
            "train_dataloader_shuffle_safe": True,
            "sampling_is_with_replacement": True,
            "joint_label_cartesian_balancing": False,
            "missing_target_loss_mask": "assistant XML value NA",
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
