#!/usr/bin/env python3
"""Summarize numeric-head checkpoints on the paired continuous validation set."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("delivery_days", "birth_weight_g", "birth_length_cm")
SCALES = np.asarray([10.0, 500.0, 2.0], dtype=np.float64)
BINS = ("d070_111", "d112_181", "d182_215", "d216_244", "d245_258")
# Changes smaller than 5% of the training normalization scale are reported as
# sampling-level noise rather than a material population reversal.
MATERIAL_TOLERANCE = 0.05 * SCALES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regression",
        type=Path,
        default=PROJECT_ROOT / "data/val_continuous_clean_regression.jsonl",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data/val_continuous_clean_metadata.jsonl",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation/checkpoint_selection_continuous",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[250, 500, 750, 1000, 1250, 1500, 1667]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "evaluation/checkpoint_selection_continuous/continuous_metrics_summary.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def read_predictions(directory: Path, step: int) -> dict[int, list[float]]:
    paths = sorted(glob.glob(str(directory / f"step{step}.rank*.jsonl")))
    if not paths:
        raise ValueError(f"step {step}: no prediction shards matched {directory}")
    result: dict[int, list[float]] = {}
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            row = json.loads(line)
            index = int(row["source_index"])
            if index in result:
                raise ValueError(f"step {step}: duplicate source index {index}")
            result[index] = [float(value) for value in row["prediction"]]
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def target_metric(rows: list[dict[str, Any]], target_index: int) -> dict[str, Any]:
    valid = [row for row in rows if row["mask"][target_index]]
    target = np.asarray([row["targets"][target_index] for row in valid], dtype=np.float64)
    prediction = np.asarray(
        [row["prediction"][target_index] for row in valid], dtype=np.float64
    )
    error = prediction - target
    absolute = np.abs(error)
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": len(valid),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "median_ae": float(np.median(absolute)),
        "p90_ae": float(np.quantile(absolute, 0.90)),
        "pearson_r": correlation(target, prediction),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else None,
        "prediction_std": float(prediction.std()),
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: target_metric(rows, index) for index, name in enumerate(TARGETS)
    }


def trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        cases[row["case_key"]][row["slot"]] = row
    output: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGETS):
        complete: list[list[float]] = []
        for slots in cases.values():
            if set(slots) != set(range(5)):
                continue
            if any(not slots[index]["mask"][target_index] for index in range(5)):
                continue
            complete.append(
                [
                    abs(
                        slots[index]["prediction"][target_index]
                        - slots[index]["targets"][target_index]
                    )
                    for index in range(5)
                ]
            )
        values = np.asarray(complete, dtype=np.float64)
        adjacent = {}
        for index in range(4):
            changes = values[:, index + 1] - values[:, index]
            adjacent[f"slot{index}_to_slot{index + 1}"] = {
                "mean_ae_change": float(changes.mean()),
                "later_not_worse_rate": float(np.mean(changes <= 0)),
            }
        first_last = values[:, -1] - values[:, 0]
        output[target_name] = {
            "complete_cases": len(values),
            "strict_individual_monotonic_rate": float(
                np.mean(np.all(values[:, 1:] <= values[:, :-1], axis=1))
            ),
            "slot0_to_slot4_mean_ae_change": float(first_last.mean()),
            "slot4_not_worse_than_slot0_rate": float(np.mean(first_last <= 0)),
            "adjacent": adjacent,
        }
    return output


def bootstrap_population_differences(
    rows: list[dict[str, Any]], replicates: int, rng: np.random.Generator
) -> dict[str, Any]:
    by_case: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_key"]][row["slot"]] = row
    case_slots = [slots for slots in by_case.values() if set(slots) == set(range(5))]
    output: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGETS):
        errors = np.asarray(
            [
                [
                    abs(
                        slots[slot]["prediction"][target_index]
                        - slots[slot]["targets"][target_index]
                    )
                    for slot in range(5)
                ]
                for slots in case_slots
            ],
            dtype=np.float64,
        )
        sample_indices = rng.integers(0, len(errors), size=(replicates, len(errors)))
        boot_means = errors[sample_indices].mean(axis=1)
        adjacent = {}
        for slot in range(4):
            observed = float(errors[:, slot + 1].mean() - errors[:, slot].mean())
            boot_delta = boot_means[:, slot + 1] - boot_means[:, slot]
            adjacent[f"slot{slot}_to_slot{slot + 1}"] = {
                "mean_mae_change": observed,
                "cluster_bootstrap_95ci": [
                    float(value) for value in np.quantile(boot_delta, [0.025, 0.975])
                ],
                "probability_change_le_zero": float(np.mean(boot_delta <= 0)),
            }
        output[target_name] = adjacent
    return output


def selection_statistics(by_bin: dict[str, Any], overall: dict[str, Any]) -> dict[str, Any]:
    sequences = np.asarray(
        [
            [by_bin[cutoff_bin][target]["mae"] for cutoff_bin in BINS]
            for target in TARGETS
        ],
        dtype=np.float64,
    )
    changes = sequences[:, 1:] - sequences[:, :-1]
    strict_positive = np.maximum(changes, 0.0)
    normalized_positive = strict_positive / SCALES[:, None]
    material = changes > MATERIAL_TOLERANCE[:, None]
    normalized_mae = np.asarray(
        [overall[target]["mae"] for target in TARGETS], dtype=np.float64
    ) / SCALES
    return {
        "mae_sequences": {
            target: [float(value) for value in sequences[index]]
            for index, target in enumerate(TARGETS)
        },
        "strict_reversal_count": int(np.sum(changes > 0)),
        "material_reversal_count": int(np.sum(material)),
        "normalized_upward_violation_sum": float(normalized_positive.sum()),
        "largest_normalized_upward_violation": float(normalized_positive.max()),
        "mean_normalized_mae": float(normalized_mae.mean()),
        "slot0_to_slot4_normalized_change": {
            target: float((sequences[index, -1] - sequences[index, 0]) / SCALES[index])
            for index, target in enumerate(TARGETS)
        },
    }


def main() -> None:
    args = parse_args()
    regression = read_jsonl(args.regression)
    metadata = read_jsonl(args.metadata)
    if len(regression) != len(metadata):
        raise ValueError("regression and metadata lengths differ")
    expected = set(range(len(regression)))
    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "protocol": "continuous_v2_population_checkpoint_selection_v1",
        "rows": len(regression),
        "unique_cases": len({f"{row['cohort']}:{row['case_id']}" for row in metadata}),
        "bins": list(BINS),
        "material_reversal_tolerance": dict(zip(TARGETS, MATERIAL_TOLERANCE.tolist())),
        "checkpoints": {},
    }
    for step in args.steps:
        predictions = read_predictions(args.prediction_dir, step)
        if set(predictions) != expected:
            raise ValueError(f"step {step}: incomplete prediction indices")
        rows = []
        for index, (regression_row, metadata_row) in enumerate(
            zip(regression, metadata, strict=True)
        ):
            rows.append(
                {
                    "case_key": f"{metadata_row['cohort']}:{metadata_row['case_id']}",
                    "cohort": metadata_row["cohort"],
                    "cutoff_bin": metadata_row["cutoff_bin"],
                    "slot": int(metadata_row["continuous_slot"]),
                    "targets": [float(value) for value in regression_row["outcome_targets"]],
                    "mask": [int(value) for value in regression_row["outcome_mask"]],
                    "prediction": predictions[index],
                }
            )
        overall = metrics(rows)
        by_bin = {
            cutoff_bin: metrics([row for row in rows if row["cutoff_bin"] == cutoff_bin])
            for cutoff_bin in BINS
        }
        by_cohort = {
            cohort: metrics([row for row in rows if row["cohort"] == cohort])
            for cohort in ("huaxi", "shenzhen")
        }
        summary["checkpoints"][str(step)] = {
            "overall": overall,
            "by_bin": by_bin,
            "by_cohort": by_cohort,
            "population_trend": selection_statistics(by_bin, overall),
            "population_adjacent_bootstrap": bootstrap_population_differences(
                rows, args.bootstrap_replicates, rng
            ),
            "individual_trajectory_diagnostic": trajectory(rows),
        }

    ranked = sorted(
        args.steps,
        key=lambda step: (
            summary["checkpoints"][str(step)]["population_trend"][
                "material_reversal_count"
            ],
            summary["checkpoints"][str(step)]["population_trend"][
                "normalized_upward_violation_sum"
            ],
            summary["checkpoints"][str(step)]["population_trend"][
                "mean_normalized_mae"
            ],
        ),
    )
    summary["selection"] = {
        "rule": [
            "minimize_material_population_reversal_count",
            "minimize_normalized_upward_violation_sum",
            "minimize_mean_normalized_mae",
        ],
        "ranking": ranked,
        "selected_step": ranked[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["selection"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
