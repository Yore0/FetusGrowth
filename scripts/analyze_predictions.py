#!/usr/bin/env python3
"""Score numeric-head predictions on the paired robustness benchmark."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks/model_agnostic_robustness_v1"
OUTPUT_ROOT = PROJECT_ROOT / "evaluation/benchmark_run"
TARGETS = ("delivery_days", "birth_weight_g", "birth_length_cm")
TARGET_LABELS = {
    "delivery_days": "Delivery day",
    "birth_weight_g": "Birth weight (g)",
    "birth_length_cm": "Birth length (cm)",
}
SCALES = {"delivery_days": 10.0, "birth_weight_g": 500.0, "birth_length_cm": 2.0}
VARIANT_ORDER = (
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
PERTURBATIONS = VARIANT_ORDER[1:]
BIN_ORDER = ("d070_111", "d112_181", "d182_215", "d216_244", "d245_258")
COLOR = "#2b6cb0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--predictions-name", type=str, default="merged_all_variants"
    )
    parser.add_argument("--model-label", type=str, default="continuous-v2")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        pair_id = str(row["pair_id"])
        if pair_id in result:
            raise ValueError(f"duplicate label pair_id={pair_id}")
        result[pair_id] = row
    return result


def ci_mean(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame[frame["label_valid"]]
    valid = eligible[eligible["prediction_valid"]]
    result: dict[str, Any] = {
        "n_eligible": int(len(eligible)),
        "n_predicted": int(len(valid)),
        "coverage": float(len(valid) / len(eligible)) if len(eligible) else math.nan,
    }
    if len(valid) == 0:
        result.update(
            {
                "mae": math.nan,
                "rmse": math.nan,
                "bias": math.nan,
                "p90_ae": math.nan,
                "pearson_r": None,
            }
        )
        return result
    errors = valid["error"].to_numpy(np.float64)
    abs_errors = valid["absolute_error"].to_numpy(np.float64)
    result.update(
        {
            "mae": float(abs_errors.mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "bias": float(errors.mean()),
            "p90_ae": float(np.quantile(abs_errors, 0.9)),
            "pearson_r": safe_corr(
                valid["prediction"].to_numpy(np.float64),
                valid["target_value"].to_numpy(np.float64),
            ),
        }
    )
    return result


def grouped_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys, strict=True))
        row.update(metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def load_long_frame(args: argparse.Namespace) -> pd.DataFrame:
    labels = load_labels(args.benchmark_root / "full/labels/outcomes.jsonl")
    pattern = str(
        args.output_dir / "predictions" / "ours" / f"{args.predictions_name}.rank*.jsonl"
    )
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no prediction shards: {pattern}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for prediction_row in read_jsonl(path):
            pair_id = str(prediction_row["pair_id"])
            variant = str(prediction_row["variant_name"])
            key = (variant, pair_id)
            if key in seen:
                raise ValueError(f"duplicate ours/{variant}/{pair_id}")
            seen.add(key)
            label = labels[pair_id]
            for index, target_name in enumerate(TARGETS):
                value = prediction_row["predictions"][index]
                prediction_valid = (
                    bool(prediction_row["parse_mask"][index])
                    and value is not None
                    and math.isfinite(float(value))
                )
                target_value = float(label["outcome_targets"][index])
                label_valid = bool(label["outcome_mask"][index])
                rows.append(
                    {
                        "model": "ours",
                        "variant": variant,
                        "pair_id": pair_id,
                        "case_id": str(prediction_row["case_id"]),
                        "cohort": prediction_row["cohort"],
                        "cutoff_bin": prediction_row["cutoff_bin"],
                        "cutoff_day": int(prediction_row["cutoff_day"]),
                        "visible_visit_count": int(prediction_row["visible_visit_count"]),
                        "visible_record_count": int(prediction_row["visible_record_count"]),
                        "target": target_name,
                        "target_value": target_value,
                        "label_valid": label_valid,
                        "prediction": float(value) if prediction_valid else math.nan,
                        "prediction_valid": prediction_valid,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["error"] = frame["prediction"] - frame["target_value"]
    frame["absolute_error"] = frame["error"].abs()
    frame["normalized_absolute_error"] = frame.apply(
        lambda row: row["absolute_error"] / SCALES[row["target"]], axis=1
    )
    return frame


def paired_robustness(
    frame: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = frame[frame["variant"] == "clean_continuous_prefix"][
        ["model", "pair_id", "target", "prediction", "prediction_valid", "absolute_error"]
    ].rename(
        columns={
            "prediction": "clean_prediction",
            "prediction_valid": "clean_prediction_valid",
            "absolute_error": "clean_absolute_error",
        }
    )
    perturbed = frame[frame["variant"] != "clean_continuous_prefix"].merge(
        clean, on=["model", "pair_id", "target"], how="left", validate="many_to_one"
    )
    paired = perturbed[
        perturbed["label_valid"]
        & perturbed["prediction_valid"]
        & perturbed["clean_prediction_valid"].fillna(False)
    ].copy()
    paired["ae_delta"] = paired["absolute_error"] - paired["clean_absolute_error"]
    paired["prediction_drift"] = (paired["prediction"] - paired["clean_prediction"]).abs()
    paired["ae_delta_normalized"] = paired.apply(
        lambda row: row["ae_delta"] / SCALES[row["target"]], axis=1
    )
    paired["prediction_drift_normalized"] = paired.apply(
        lambda row: row["prediction_drift"] / SCALES[row["target"]], axis=1
    )

    rows = []
    for group_index, ((model, variant, target), group) in enumerate(
        paired.groupby(["model", "variant", "target"], sort=False)
    ):
        delta = group["ae_delta"].to_numpy(np.float64)
        drift = group["prediction_drift"].to_numpy(np.float64)
        low, high = ci_mean(delta, replicates, seed + group_index)
        rows.append(
            {
                "model": model,
                "variant": variant,
                "target": target,
                "n_pairs": int(len(group)),
                "mean_ae_delta": float(delta.mean()),
                "ae_delta_ci95_low": low,
                "ae_delta_ci95_high": high,
                "mean_ae_delta_normalized": float(group["ae_delta_normalized"].mean()),
                "mean_prediction_drift": float(drift.mean()),
                "mean_prediction_drift_normalized": float(
                    group["prediction_drift_normalized"].mean()
                ),
                "not_worse_rate": float(np.mean(delta <= 0)),
            }
        )
    paired_table = pd.DataFrame(rows)

    composite_source = paired.copy()
    composite_source["normalized_composite"] = composite_source["ae_delta_normalized"]
    composite_rows = []
    for group_index, ((model, variant), group) in enumerate(
        composite_source.groupby(["model", "variant"], sort=False)
    ):
        # mean across targets per pair then across pairs
        per_pair = (
            group.groupby("pair_id", sort=False)["ae_delta_normalized"].mean().to_numpy()
        )
        low, high = ci_mean(per_pair, replicates, seed + 10_000 + group_index)
        composite_rows.append(
            {
                "model": model,
                "variant": variant,
                "n_pairs": int(len(per_pair)),
                "mean_normalized_composite_ae_delta": float(per_pair.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return paired_table, pd.DataFrame(composite_rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: f"{value:.{digits}f}" if pd.notna(value) else "NA"
        )
    return view.to_markdown(index=False)


def plots(
    overall: pd.DataFrame,
    by_cohort: pd.DataFrame,
    by_cutoff: pd.DataFrame,
    paired: pd.DataFrame,
    composite: pd.DataFrame,
    output: Path,
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    clean = overall[overall["variant"] == "clean_continuous_prefix"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, target in zip(axes, TARGETS, strict=True):
        value = float(clean[clean.target == target].mae.iloc[0])
        bars = axis.bar(["Ours ckpt-1667"], [value], color=COLOR)
        axis.bar_label(bars, fmt="%.2f", fontsize=9)
        axis.set_title(TARGET_LABELS[target])
        axis.set_ylabel("MAE")
    fig.suptitle("Full clean continuous-prefix performance")
    fig.savefig(figure_dir / "clean_mae.png", dpi=180)
    plt.close(fig)

    matrix = (
        paired.pivot(index="variant", columns="target", values="mean_ae_delta_normalized")
        .reindex(index=PERTURBATIONS, columns=TARGETS)
    )
    fig, axis = plt.subplots(figsize=(6.5, 7), constrained_layout=True)
    data = matrix.to_numpy()
    maximum = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    image = axis.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    axis.set_xticks(range(3), ["Delivery", "Weight", "Length"])
    axis.set_yticks(range(len(PERTURBATIONS)), PERTURBATIONS)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            number = matrix.iloc[y, x]
            if np.isfinite(number):
                axis.text(x, y, f"{number:.3f}", ha="center", va="center", fontsize=7)
    axis.set_title("Paired error degradation vs clean (normalized)")
    fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(figure_dir / "robustness_degradation_heatmap.png", dpi=180)
    plt.close(fig)

    drift = (
        paired.pivot(
            index="variant", columns="target", values="mean_prediction_drift_normalized"
        ).reindex(index=PERTURBATIONS, columns=TARGETS)
    )
    fig, axis = plt.subplots(figsize=(6.5, 7), constrained_layout=True)
    data = drift.to_numpy()
    maximum = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
    image = axis.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=maximum)
    axis.set_xticks(range(3), ["Delivery", "Weight", "Length"])
    axis.set_yticks(range(len(PERTURBATIONS)), PERTURBATIONS)
    for y in range(drift.shape[0]):
        for x in range(drift.shape[1]):
            number = drift.iloc[y, x]
            if np.isfinite(number):
                axis.text(x, y, f"{number:.3f}", ha="center", va="center", fontsize=7)
    axis.set_title("Prediction drift from paired clean view (normalized)")
    fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(figure_dir / "prediction_drift_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    cohort_clean = by_cohort[by_cohort["variant"] == "clean_continuous_prefix"]
    for axis, target in zip(axes, TARGETS, strict=True):
        part = cohort_clean[cohort_clean["target"] == target]
        vals = [
            float(part[part.cohort == cohort].mae.iloc[0])
            for cohort in ("huaxi", "shenzhen")
        ]
        bars = axis.bar(["Huaxi", "Shenzhen"], vals, color=COLOR)
        axis.bar_label(bars, fmt="%.2f", fontsize=9)
        axis.set_title(TARGET_LABELS[target])
        axis.set_ylabel("MAE")
    fig.suptitle("Clean MAE by cohort")
    fig.savefig(figure_dir / "clean_mae_by_cohort.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    cutoff_clean = by_cutoff[by_cutoff["variant"] == "clean_continuous_prefix"]
    for axis, target in zip(axes, TARGETS, strict=True):
        part = cutoff_clean[cutoff_clean.target == target]
        values = [float(part[part.cutoff_bin == bin_name].mae.iloc[0]) for bin_name in BIN_ORDER]
        axis.plot(range(5), values, marker="o", color=COLOR)
        axis.set_xticks(
            range(5), ["70–111", "112–181", "182–215", "216–244", "245–258"], rotation=25
        )
        axis.set_title(TARGET_LABELS[target])
        axis.set_ylabel("MAE")
    fig.suptitle("Clean performance across continuous cutoff bins")
    fig.savefig(figure_dir / "clean_mae_by_cutoff_bin.png", dpi=180)
    plt.close(fig)

    severity = {
        "Content mask": (
            "clean_continuous_prefix",
            "content_mask_15",
            "content_mask_30",
            "content_mask_50",
        ),
        "Visit dropout": (
            "clean_continuous_prefix",
            "visit_dropout_20",
            "visit_dropout_40",
        ),
        "Local context": (
            "clean_continuous_prefix",
            "local_window_84",
            "local_window_56",
            "local_window_28",
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (family, variants) in zip(axes, severity.items(), strict=True):
        values = [0.0]
        for variant in variants[1:]:
            match = composite[composite.variant == variant]
            values.append(float(match.mean_normalized_composite_ae_delta.iloc[0]))
        axis.plot(range(len(variants)), values, marker="o", color=COLOR)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(
            range(len(variants)),
            [
                name.replace("clean_continuous_prefix", "clean")
                for name in variants
            ],
            rotation=25,
        )
        axis.set_title(family)
        axis.set_ylabel("Paired normalized error delta")
    fig.suptitle("Robustness as perturbation severity increases")
    fig.savefig(figure_dir / "severity_curves.png", dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    composite: pd.DataFrame,
    model_label: str,
) -> None:
    clean = overall[overall.variant == "clean_continuous_prefix"].copy()
    clean["target"] = clean["target"].map(TARGET_LABELS)
    clean_table = markdown_table(
        clean,
        [
            "target",
            "n_eligible",
            "n_predicted",
            "coverage",
            "mae",
            "rmse",
            "bias",
            "p90_ae",
            "pearson_r",
        ],
    )
    composite_table = markdown_table(
        composite,
        [
            "variant",
            "n_pairs",
            "mean_normalized_composite_ae_delta",
            "ci95_low",
            "ci95_high",
        ],
    )
    paired_view = paired.copy()
    paired_view["target"] = paired_view["target"].map(TARGET_LABELS)
    paired_table = markdown_table(
        paired_view,
        [
            "variant",
            "target",
            "n_pairs",
            "mean_ae_delta",
            "ae_delta_ci95_low",
            "ae_delta_ci95_high",
            "mean_prediction_drift",
            "not_worse_rate",
        ],
    )
    text = f"""# Full robustness benchmark: {model_label}

## Protocol

- Model: `{model_label}`, numeric predictions from three regression heads.
- Split: full model-agnostic robustness benchmark (13 variants, merged inference).
- Labels: `full/labels/outcomes.jsonl`, joined by `pair_id`.
- Robustness deltas are paired against the identical clean parent and cutoff.

## Clean performance

{clean_table}

## Normalized composite robustness degradation

{composite_table}

## Target-level degradation and prediction drift

{paired_table}

## Figures

- `figures/clean_mae.png`
- `figures/clean_mae_by_cohort.png`
- `figures/clean_mae_by_cutoff_bin.png`
- `figures/robustness_degradation_heatmap.png`
- `figures/prediction_drift_heatmap.png`
- `figures/severity_curves.png`
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_long_frame(args)
    overall = grouped_metrics(frame, ["model", "variant", "target"])
    by_cohort = grouped_metrics(frame, ["model", "variant", "cohort", "target"])
    by_cutoff = grouped_metrics(frame, ["model", "variant", "cutoff_bin", "target"])
    paired, composite = paired_robustness(
        frame, args.bootstrap_replicates, args.seed
    )

    tables = {
        "metrics_overall.csv": overall,
        "metrics_by_cohort.csv": by_cohort,
        "metrics_by_cutoff_bin.csv": by_cutoff,
        "paired_robustness.csv": paired,
        "paired_composite_robustness.csv": composite,
    }
    for filename, table in tables.items():
        table.to_csv(args.output_dir / filename, index=False)
    plots(overall, by_cohort, by_cutoff, paired, composite, args.output_dir)
    write_report(args.output_dir, overall, paired, composite, args.model_label)

    clean = overall[overall.variant == "clean_continuous_prefix"]
    summary = {
        "protocol": "model_agnostic_numeric_head_robustness_v1",
        "model": args.model_label,
        "predictions_name": args.predictions_name,
        "rows_long": len(frame),
        "bootstrap_replicates": args.bootstrap_replicates,
        "clean_metrics": clean.to_dict(orient="records"),
        "artifacts": list(tables) + ["REPORT.md", "figures/"],
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
