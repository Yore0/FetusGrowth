#!/usr/bin/env python3
"""Prepare leakage-safe temporal SFT views from the Huaxi/Shenzhen JSONL files.

The protocol deliberately treats the Huaxi test cohort as a *seen test*: the
test cases already present in ``huaxi_train.jsonl`` remain in training exactly
once.  The test file is never appended to the training pool, and Huaxi
validation IDs are selected exclusively from the non-test-ID subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PROTOCOL_VERSION = "temporal_outcomes_v1_20260727"
DEFAULT_SEED = 20260727

DELIVERY_RANGE = (140, 320)
BIRTH_WEIGHT_RANGE = (500, 6000)
BIRTH_LENGTH_RANGE = (20, 65)


@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    cutoff_day: int


# These are inclusive 13w6d/22w6d/28w6d/32w6d/36w6d boundaries.
CANONICAL_STAGES: tuple[Stage, ...] = (
    Stage("prefix_w13", "w13", 97),
    Stage("prefix_w22", "w22", 160),
    Stage("prefix_w28", "w28", 202),
    Stage("prefix_w32", "w32", 230),
    Stage("prefix_w36", "w36", 258),
)

# T1 ends at day 97, so an isolated post-T1 window begins at day 98.
LOCAL_WINDOWS: tuple[tuple[str, str, int, int], ...] = (
    ("window_w13_w22", "w22", 98, 160),
    ("window_w13_w28", "w28", 98, 202),
)

TIMELINE_MARKER = "[全周期检查记录（按时间排序）]"
OUTPUT_TIMELINE_MARKER = "[当前可见检查记录（按时间排序）]"
FUTURE_SECTION_HEADERS = {
    "[分娩前摘要]",
    "[重点异常与诊断提示]",
    "[关键异常对齐摘要]",
    "[软性提示]",
    "[建模辅助提示]",
}
FUTURE_BASIC_INFO_PATTERNS = (
    re.compile(r"^\s*-\s*孕期糖尿病(?:\(GDM\))?\s*:", re.I),
    re.compile(r"^\s*-\s*孕期高血压\s*:", re.I),
)

XML_RE = re.compile(r"<([A-Za-z_]+)>(.*?)</\1>", re.S)
DAY_RE = re.compile(r"DaysFromLMP\s*=\s*([^\s|]+)", re.I)
SECTION_SPLIT_RE = re.compile(r"(?m)(?=^\[[^\n]+\])")
INEQUALITY_RE = re.compile(r"(?:≥|≤|>|<|>=|<=)")

TEMPORAL_PROMPT = """请仅基于下方当前实际可见的部分孕期信息，预测最终分娩结局。
最后仅输出以下 XML 标签，不要输出思考过程或额外解释：
<delivery_days>...</delivery_days>
<birth_weight_g>...</birth_weight_g>
<birth_length_cm>...</birth_length_cm>

说明：
1. <delivery_days>是从末次月经起算的最终分娩孕周天数（DaysFromLMP），不是剩余天数。
2. 当前输入可能只覆盖某个时间窗口，也可能缺失整次就诊；窗口外或未出现的信息均为未知，不能假设为正常或已经发生。
3. 预测时只能使用当前可见检查，不能推断存在任何分娩后记录或结局提示。"""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument(
        "--dropout-copies",
        type=int,
        default=1,
        help="Deterministic visit-bundle dropout candidates per training case.",
    )
    parser.add_argument("--dropout-min", type=float, default=0.10)
    parser.add_argument("--dropout-max", type=float, default=0.30)
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
    payload = f"{seed}:{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def source_case_id(row: dict[str, Any]) -> str:
    raw = row.get("source_sample_id")
    if raw is None or str(raw).strip() == "":
        raw = row.get("sample_id")
    if raw is None or str(raw).strip() == "":
        raise ValueError("Source row has neither source_sample_id nor sample_id")
    value = str(raw).strip()
    return re.sub(r"__(?:full|cutoff.*|prefix.*|window.*)$", "", value)


def get_message(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError(f"Missing {role} message for {row.get('sample_id')}")


def parse_xml(text: str) -> dict[str, str]:
    return {name: value.strip() for name, value in XML_RE.findall(text or "")}


def parse_rounded_number(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(round(number))


def clean_targets(assistant_text: str) -> tuple[dict[str, int | None] | None, Counter[str]]:
    """Return clean targets; an invalid delivery label invalidates the case."""

    audit: Counter[str] = Counter()
    fields = parse_xml(assistant_text)
    delivery = parse_rounded_number(fields.get("delivery_days"))
    if delivery is None:
        audit["case_drop_missing_delivery"] += 1
        return None, audit
    if not DELIVERY_RANGE[0] <= delivery <= DELIVERY_RANGE[1]:
        audit["case_drop_invalid_delivery_range"] += 1
        return None, audit

    weight = parse_rounded_number(fields.get("birth_weight_g"))
    if weight is None or not BIRTH_WEIGHT_RANGE[0] <= weight <= BIRTH_WEIGHT_RANGE[1]:
        if weight is None:
            audit["birth_weight_set_na_missing_or_zero"] += 1
        else:
            audit["birth_weight_set_na_invalid_range"] += 1
        weight = None

    length = parse_rounded_number(fields.get("birth_length_cm"))
    if length is None or not BIRTH_LENGTH_RANGE[0] <= length <= BIRTH_LENGTH_RANGE[1]:
        if length is None:
            audit["birth_length_set_na_missing_or_zero"] += 1
        else:
            audit["birth_length_set_na_invalid_range"] += 1
        length = None

    return {
        "delivery_days": delivery,
        "birth_weight_g": weight,
        "birth_length_cm": length,
    }, audit


def parse_day_expression(expr: str) -> tuple[tuple[int, int] | None, str | None]:
    text = expr.strip().replace("～", "~")
    if INEQUALITY_RE.search(text):
        return None, "ambiguous_inequality_day"
    range_match = re.fullmatch(r"(-?\d+)\s*(?:-|~|至)\s*(-?\d+)", text)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        return (min(start, end), max(start, end)), None
    single_match = re.fullmatch(r"-?\d+", text)
    if single_match:
        value = int(text)
        return (value, value), None
    return None, "unparseable_day"


def infer_modality(title: str, text: str) -> str:
    value = f"{title}\n{text[:300]}".lower()
    if any(token in value for token in ("超声", "b超", "ultrasound", "彩超")):
        return "ultrasound"
    if any(token in value for token in ("化验", "检验", "血常规", "尿常规", "实验室", "lab")):
        return "lab"
    if any(token in value for token in ("随访", "复诊", "产检", "门诊")):
        return "followup"
    if any(token in value for token in ("入组", "建档", "初诊", "登记")):
        return "enrollment"
    return "other"


ULTRASOUND_VALUE_RE = re.compile(
    r"(?P<label>"
    r"头臀长\(CRL\)|双顶径\(BPD\)|头围\(HC\)|腹围\(AC\)|股骨长\(FL\)|"
    r"估计胎儿体重\(EFW\)|CRL|BPD|HC|AC|FL|FW|EFW"
    r")(?P<separator>\s*=\s*)(?P<value>-?\d+(?:\.\d+)?)"
    r"(?P<unit>\s*(?:mm|g)\b)?",
    re.I,
)
AFI_VALUE_RE = re.compile(
    r"(?P<label>羊水指数\(AFI\)|AFI)(?P<separator>\s*=\s*)"
    r"(?P<value>-?\d+(?:\.\d+)?)(?P<unit>\s*cm\b)?",
    re.I,
)
UMBILICAL_VEIN_DIAMETER_RE = re.compile(
    r"(?P<label>脐静脉内径)(?P<separator>\s*=\s*)"
    r"(?P<value>-?\d+(?:\.\d+)?)(?P<unit>\s*mm\b)?",
    re.I,
)
UMBILICAL_VEIN_FLOW_RE = re.compile(
    r"(?P<label>脐静脉血流量)(?P<separator>\s*=\s*)"
    r"(?P<value>-?\d+(?:\.\d+)?)(?P<unit>\s*ml/min\b)?",
    re.I,
)
# Deliberately broad data-quality guards, not clinical/diagnostic thresholds.
UMBILICAL_VEIN_DIAMETER_QUALITY_RANGE_MM = (0.0, 30.0)
UMBILICAL_VEIN_FLOW_QUALITY_RANGE_ML_MIN = (0.0, 2000.0)
MEASUREMENT_LIMITS = {
    "crl": (1.0, 150.0),
    "bpd": (5.0, 120.0),
    "hc": (20.0, 420.0),
    "ac": (20.0, 450.0),
    "fl": (2.0, 90.0),
    "fw": (30.0, 6000.0),
    "efw": (30.0, 6000.0),
}


def _measurement_key(label: str) -> str:
    upper = label.upper()
    for key in ("CRL", "BPD", "HC", "AC", "FL", "EFW", "FW"):
        if key in upper:
            return key.lower()
    raise ValueError(label)


def sanitize_ultrasound_text(text: str) -> tuple[str, Counter[str]]:
    """Conservatively mask impossible core fetal measurements.

    Zero is not removed globally: only core ultrasound growth fields treat zero
    as missing.  Laboratory zeroes and a true AFI of zero remain untouched.
    """

    audit: Counter[str] = Counter()

    def replace_measurement(match: re.Match[str]) -> str:
        key = _measurement_key(match.group("label"))
        value = float(match.group("value"))
        minimum, maximum = MEASUREMENT_LIMITS[key]
        if value == 0:
            audit[f"ultrasound_{key}_zero_set_na"] += 1
        elif value < minimum or value > maximum:
            audit[f"ultrasound_{key}_implausible_set_na"] += 1
        else:
            return match.group(0)
        return (
            match.group("label")
            + match.group("separator")
            + "NA"
            + (match.group("unit") or "")
        )

    def replace_afi(match: re.Match[str]) -> str:
        value = float(match.group("value"))
        # Source AFI is documented in cm. Values over 50 cm are unit/entry errors.
        if value <= 50:
            return match.group(0)
        audit["ultrasound_afi_implausible_set_na"] += 1
        return (
            match.group("label")
            + match.group("separator")
            + "NA"
            + (match.group("unit") or "")
        )

    def replace_umbilical_vein_diameter(match: re.Match[str]) -> str:
        value = float(match.group("value"))
        low, high = UMBILICAL_VEIN_DIAMETER_QUALITY_RANGE_MM
        if low < value <= high:
            return match.group(0)
        audit["ultrasound_umbilical_vein_diameter_quality_set_na"] += 1
        return (
            match.group("label")
            + match.group("separator")
            + "NA"
            + (match.group("unit") or "")
        )

    def replace_umbilical_vein_flow(match: re.Match[str]) -> str:
        value = float(match.group("value"))
        low, high = UMBILICAL_VEIN_FLOW_QUALITY_RANGE_ML_MIN
        if low < value <= high:
            return match.group(0)
        audit["ultrasound_umbilical_vein_flow_quality_set_na"] += 1
        return (
            match.group("label")
            + match.group("separator")
            + "NA"
            + (match.group("unit") or "")
        )

    cleaned = ULTRASOUND_VALUE_RE.sub(replace_measurement, text)
    cleaned = AFI_VALUE_RE.sub(replace_afi, cleaned)
    cleaned = UMBILICAL_VEIN_DIAMETER_RE.sub(
        replace_umbilical_vein_diameter, cleaned
    )
    cleaned = UMBILICAL_VEIN_FLOW_RE.sub(replace_umbilical_vein_flow, cleaned)
    return cleaned, audit


def sanitize_basic_info(text: str) -> tuple[str, Counter[str]]:
    audit: Counter[str] = Counter()
    kept: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in FUTURE_BASIC_INFO_PATTERNS):
            audit["future_derived_basic_field_removed"] += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip(), audit


def parse_prompt_sections(
    user_text: str,
    *,
    cohort: str,
    delivery_days: int,
) -> tuple[str, list[dict[str, Any]], Counter[str]]:
    """Extract and clean basic information and dated timeline blocks."""

    audit: Counter[str] = Counter()
    basic_marker = "[孕妇基本信息]"
    basic_index = user_text.find(basic_marker)
    timeline_index = user_text.find(TIMELINE_MARKER)
    if basic_index < 0 or timeline_index <= basic_index:
        raise ValueError("Cannot locate basic information and full-cycle timeline")

    # Anything before [孕妇基本信息], including a GT soft hint, is discarded.
    prefix = user_text[:basic_index]
    if any(header in prefix for header in FUTURE_SECTION_HEADERS):
        audit["future_hint_prefix_removed"] += 1

    basic_raw = user_text[basic_index + len(basic_marker):timeline_index].strip()
    basic_info, basic_audit = sanitize_basic_info(basic_raw)
    audit.update(basic_audit)

    timeline = user_text[timeline_index + len(TIMELINE_MARKER):].strip()
    blocks: list[dict[str, Any]] = []
    for chunk in SECTION_SPLIT_RE.split(timeline):
        chunk = chunk.strip()
        if not chunk:
            continue
        title = chunk.splitlines()[0].strip()
        header = title.split(" DaysFromLMP", 1)[0].strip()
        if header in FUTURE_SECTION_HEADERS or title in FUTURE_SECTION_HEADERS:
            audit["future_summary_section_removed"] += 1
            continue
        day_match = DAY_RE.search(chunk)
        if not day_match:
            audit["undated_block_removed"] += 1
            continue
        expression = day_match.group(1)
        parsed, reason = parse_day_expression(expression)
        if parsed is None:
            if cohort == "huaxi" and INEQUALITY_RE.search(expression):
                audit["huaxi_ambiguous_ge34_block_removed"] += 1
            else:
                audit[f"{reason}_block_removed"] += 1
            continue
        start_day, end_day = parsed
        if start_day < 0 or end_day < 0:
            audit["negative_day_block_removed"] += 1
            continue
        # At-delivery records can directly encode the outcome and are also hidden.
        if end_day >= delivery_days:
            audit["at_or_after_delivery_block_removed"] += 1
            continue
        modality = infer_modality(title, chunk)
        cleaned_text = chunk
        if modality == "ultrasound":
            cleaned_text, measurement_audit = sanitize_ultrasound_text(chunk)
            audit.update(measurement_audit)
        blocks.append(
            {
                "title": title,
                "text": cleaned_text,
                "start_day": start_day,
                "end_day": end_day,
                "modality": modality,
            }
        )
    blocks.sort(key=lambda block: (block["end_day"], block["start_day"], block["title"]))
    return basic_info, blocks, audit


def parse_source_case(
    row: dict[str, Any],
    *,
    cohort: str,
    source_name: str,
) -> tuple[dict[str, Any] | None, Counter[str]]:
    audit: Counter[str] = Counter()
    try:
        assistant = get_message(row, "assistant")
        user = get_message(row, "user")
    except ValueError:
        audit["case_drop_missing_message"] += 1
        return None, audit

    targets, target_audit = clean_targets(assistant)
    audit.update(target_audit)
    if targets is None:
        return None, audit
    assert targets["delivery_days"] is not None

    try:
        basic_info, blocks, block_audit = parse_prompt_sections(
            user,
            cohort=cohort,
            delivery_days=targets["delivery_days"],
        )
    except ValueError:
        audit["case_drop_unparseable_prompt"] += 1
        return None, audit
    audit.update(block_audit)
    if not blocks:
        audit["case_drop_no_clean_dated_blocks"] += 1
        return None, audit

    case_id = source_case_id(row)
    return {
        "case_id": case_id,
        "source_sample_id": case_id,
        "source_row_sample_id": str(row.get("sample_id") or case_id),
        "cohort": cohort,
        "hospital": str(row.get("hospital") or cohort),
        "source_name": source_name,
        "basic_info": basic_info,
        "blocks": blocks,
        "source_record_count": int(row.get("visible_record_count") or len(blocks)),
        "clean_record_count": len(blocks),
        "targets": targets,
    }, audit


def parse_source_rows(
    rows: Sequence[dict[str, Any]],
    *,
    cohort: str,
    source_name: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    cases: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()
    for row in rows:
        case, row_audit = parse_source_case(row, cohort=cohort, source_name=source_name)
        audit.update(row_audit)
        if case is not None:
            cases.append(case)
    assert_unique_case_ids(cases, source_name)
    return cases, audit


def reconcile_huaxi_seen_test_cases(
    huaxi_train_cases: Sequence[dict[str, Any]],
    huaxi_test_cases: Sequence[dict[str, Any]],
) -> Counter[str]:
    """Reconcile the user-specified Huaxi train-seen duplicate cohort.

    Test rows are never appended. Cleaned inputs must be byte-identical. A
    valid seen-test target may fill a missing/invalid raw-train target, while
    conflicting valid targets fail loudly instead of selecting one silently.
    """

    audit: Counter[str] = Counter()
    train_by_id = {str(case["case_id"]): case for case in huaxi_train_cases}
    for test_case in huaxi_test_cases:
        case_id = str(test_case["case_id"])
        train_case = train_by_id.get(case_id)
        if train_case is None:
            raise AssertionError(
                f"Valid Huaxi seen-test case is absent from raw train: {case_id}"
            )
        train_input = (
            train_case["basic_info"],
            train_case["blocks"],
        )
        test_input = (
            test_case["basic_info"],
            test_case["blocks"],
        )
        if train_input != test_input:
            raise AssertionError(
                "Cleaned Huaxi train/seen-test inputs differ for "
                f"case_id={case_id}"
            )
        audit["cleaned_input_exact_match_cases"] += 1

        for target_name in (
            "delivery_days",
            "birth_weight_g",
            "birth_length_cm",
        ):
            train_value = train_case["targets"].get(target_name)
            test_value = test_case["targets"].get(target_name)
            if (
                train_value is not None
                and test_value is not None
                and train_value != test_value
            ):
                raise AssertionError(
                    "Conflicting valid Huaxi train/seen-test targets: "
                    f"case_id={case_id}, target={target_name}, "
                    f"train={train_value}, seen_test={test_value}"
                )
            if train_value is None and test_value is not None:
                train_case["targets"][target_name] = test_value
                audit[f"train_{target_name}_filled_from_seen_test"] += 1
        audit["seen_test_cases_reconciled_without_append"] += 1
    return audit


def assert_unique_case_ids(cases: Sequence[dict[str, Any]], label: str) -> None:
    counts = Counter(str(case["case_id"]) for case in cases)
    duplicates = [case_id for case_id, count in counts.items() if count > 1]
    if duplicates:
        raise AssertionError(f"Duplicate case IDs in {label}: {duplicates[:5]}")


def select_validation_ids(
    cases: Sequence[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
    cohort: str,
    excluded_ids: set[str] | None = None,
) -> set[str]:
    if not 0 <= fraction < 1:
        raise ValueError("val-fraction must be in [0, 1)")
    excluded_ids = excluded_ids or set()
    eligible = sorted(
        {str(case["case_id"]) for case in cases if str(case["case_id"]) not in excluded_ids},
        key=lambda case_id: (stable_int(seed, f"{cohort}:{case_id}"), case_id),
    )
    count = int(round(len(eligible) * fraction))
    if eligible and fraction > 0:
        count = max(1, count)
    return set(eligible[:count])


def partition_training_cases(
    huaxi_cases: Sequence[dict[str, Any]],
    shenzhen_cases: Sequence[dict[str, Any]],
    *,
    huaxi_test_ids: set[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]]]:
    """Split internal validation while retaining Huaxi seen-test IDs in train."""

    huaxi_val_ids = select_validation_ids(
        huaxi_cases,
        fraction=val_fraction,
        seed=seed,
        cohort="huaxi",
        excluded_ids=huaxi_test_ids,
    )
    shenzhen_val_ids = select_validation_ids(
        shenzhen_cases,
        fraction=val_fraction,
        seed=seed,
        cohort="shenzhen",
    )
    if huaxi_val_ids & huaxi_test_ids:
        raise AssertionError("Huaxi internal validation contains a seen-test ID")

    train = [
        case
        for case in huaxi_cases
        if str(case["case_id"]) not in huaxi_val_ids
    ] + [
        case
        for case in shenzhen_cases
        if str(case["case_id"]) not in shenzhen_val_ids
    ]
    validation = [
        case
        for case in huaxi_cases
        if str(case["case_id"]) in huaxi_val_ids
    ] + [
        case
        for case in shenzhen_cases
        if str(case["case_id"]) in shenzhen_val_ids
    ]
    train_ids = {(case["cohort"], str(case["case_id"])) for case in train}
    validation_ids = {(case["cohort"], str(case["case_id"])) for case in validation}
    if train_ids & validation_ids:
        raise AssertionError("Internal train/validation case overlap")
    return train, validation, {
        "huaxi_val_ids": huaxi_val_ids,
        "shenzhen_val_ids": shenzhen_val_ids,
    }


def encounter_groups(blocks: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for block in blocks:
        key = (int(block["start_day"]), int(block["end_day"]))
        grouped.setdefault(key, []).append(block)
    return [
        grouped[key]
        for key in sorted(grouped, key=lambda item: (item[1], item[0]))
    ]


def visible_blocks(
    case: dict[str, Any],
    *,
    window_start_day: int,
    cutoff_day: int,
) -> list[dict[str, Any]]:
    """Use only complete blocks wholly contained in the requested window."""

    return [
        block
        for block in case["blocks"]
        if int(block["start_day"]) >= window_start_day
        and int(block["end_day"]) <= cutoff_day
    ]


def availability(blocks: Sequence[dict[str, Any]], cutoff_day: int) -> dict[str, Any]:
    visits = encounter_groups(blocks)
    if not blocks:
        return {
            "visible_record_count": 0,
            "visible_visit_count": 0,
            "visible_first_day": None,
            "visible_last_day": None,
            "days_since_last_record": None,
            "visible_modality_counts": {},
        }
    counts = Counter(str(block["modality"]) for block in blocks)
    first = min(int(block["start_day"]) for block in blocks)
    last = max(int(block["end_day"]) for block in blocks)
    return {
        "visible_record_count": len(blocks),
        "visible_visit_count": len(visits),
        "visible_first_day": first,
        "visible_last_day": last,
        "days_since_last_record": cutoff_day - last,
        "visible_modality_counts": dict(sorted(counts.items())),
    }


def build_user_prompt(
    case: dict[str, Any],
    blocks: Sequence[dict[str, Any]],
    *,
    window_start_day: int,
    cutoff_day: int,
) -> tuple[str, dict[str, Any]]:
    info = availability(blocks, cutoff_day)
    modality_text = ", ".join(
        f"{name}={count}" for name, count in info["visible_modality_counts"].items()
    )
    availability_text = "\n".join(
        [
            "[数据可用性概况]",
            f"- observation_window_start_day: {window_start_day}",
            f"- observation_window_end_day: {cutoff_day}",
            f"- available_visit_count: {info['visible_visit_count']}",
            f"- available_record_count: {info['visible_record_count']}",
            f"- first_observed_day: {info['visible_first_day']}",
            f"- last_observed_day: {info['visible_last_day']}",
            f"- days_since_last_record: {info['days_since_last_record']}",
            f"- modality_counts: {modality_text or 'none'}",
            "- 上述信息只描述当前可见数据；窗口外和缺失检查均为未知。",
        ]
    )
    prompt = "\n\n".join(
        [
            TEMPORAL_PROMPT,
            availability_text,
            "[孕妇基本信息]\n" + str(case["basic_info"]).strip(),
            OUTPUT_TIMELINE_MARKER + "\n" + "\n\n".join(block["text"] for block in blocks),
        ]
    ).strip()
    return prompt, info


def target_response(targets: dict[str, int | None]) -> str:
    def display(name: str) -> str:
        value = targets.get(name)
        return "NA" if value is None else str(value)

    return "\n".join(
        [
            f"<delivery_days>{display('delivery_days')}</delivery_days>",
            f"<birth_weight_g>{display('birth_weight_g')}</birth_weight_g>",
            f"<birth_length_cm>{display('birth_length_cm')}</birth_length_cm>",
        ]
    )


def make_view(
    case: dict[str, Any],
    blocks: Sequence[dict[str, Any]],
    *,
    split: str,
    view_type: str,
    view_name: str,
    stage: str,
    window_start_day: int,
    cutoff_day: int,
    is_huaxi_seen_test: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not blocks:
        raise ValueError("A temporal view must contain at least one visible block")
    if int(case["targets"]["delivery_days"]) <= cutoff_day:
        raise ValueError("delivery_days must be strictly greater than cutoff_day")
    if any(
        int(block["start_day"]) < window_start_day
        or int(block["end_day"]) > cutoff_day
        for block in blocks
    ):
        raise AssertionError("A block crosses the temporal view boundary")
    prompt, info = build_user_prompt(
        case,
        blocks,
        window_start_day=window_start_day,
        cutoff_day=cutoff_day,
    )
    case_id = str(case["case_id"])
    view_id = f"{case['cohort']}:{case_id}:{view_name}"
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_id": view_id,
        "view_id": view_id,
        "source_sample_id": case_id,
        "case_id": case_id,
        "cohort": case["cohort"],
        "hospital": case["hospital"],
        "split": split,
        "source_name": case["source_name"],
        "source_row_sample_id": case["source_row_sample_id"],
        "is_huaxi_seen_test": bool(is_huaxi_seen_test),
        "view_type": view_type,
        "view_name": view_name,
        "stage": stage,
        "window_start_day": window_start_day,
        "window_end_day": cutoff_day,
        "cutoff_day": cutoff_day,
        **info,
        "source_record_count": case["source_record_count"],
        "clean_record_count": case["clean_record_count"],
        "targets": dict(case["targets"]),
        "actual_delivery_days": case["targets"]["delivery_days"],
        "actual_birth_weight_g": case["targets"]["birth_weight_g"],
        "actual_birth_length_cm": case["targets"]["birth_length_cm"],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_response(case["targets"])},
        ],
    }
    if extra:
        result.update(extra)
    return result


def canonical_views_for_case(
    case: dict[str, Any],
    *,
    split: str,
    huaxi_test_ids: set[str],
) -> Iterator[dict[str, Any]]:
    delivery = int(case["targets"]["delivery_days"])
    seen_test = case["cohort"] == "huaxi" and str(case["case_id"]) in huaxi_test_ids
    for stage in CANONICAL_STAGES:
        if delivery <= stage.cutoff_day:
            continue
        blocks = visible_blocks(case, window_start_day=0, cutoff_day=stage.cutoff_day)
        if not blocks:
            continue
        yield make_view(
            case,
            blocks,
            split=split,
            view_type="canonical_prefix",
            view_name=stage.name,
            stage=stage.label,
            window_start_day=0,
            cutoff_day=stage.cutoff_day,
            is_huaxi_seen_test=seen_test,
        )


def local_window_views_for_case(
    case: dict[str, Any],
    *,
    split: str,
    huaxi_test_ids: set[str],
) -> Iterator[dict[str, Any]]:
    delivery = int(case["targets"]["delivery_days"])
    seen_test = case["cohort"] == "huaxi" and str(case["case_id"]) in huaxi_test_ids
    for view_name, stage, start, cutoff in LOCAL_WINDOWS:
        if delivery <= cutoff:
            continue
        blocks = visible_blocks(case, window_start_day=start, cutoff_day=cutoff)
        if not blocks:
            continue
        yield make_view(
            case,
            blocks,
            split=split,
            view_type="local_window",
            view_name=view_name,
            stage=stage,
            window_start_day=start,
            cutoff_day=cutoff,
            is_huaxi_seen_test=seen_test,
        )


def dropout_views_for_case(
    case: dict[str, Any],
    *,
    split: str,
    huaxi_test_ids: set[str],
    copies: int,
    seed: int,
    dropout_min: float,
    dropout_max: float,
) -> Iterator[dict[str, Any]]:
    if copies <= 0:
        return
    if not 0 < dropout_min <= dropout_max < 1:
        raise ValueError("dropout fractions must satisfy 0 < min <= max < 1")
    candidates: list[tuple[Stage, list[dict[str, Any]]]] = []
    delivery = int(case["targets"]["delivery_days"])
    for stage in CANONICAL_STAGES:
        if delivery <= stage.cutoff_day:
            continue
        blocks = visible_blocks(case, window_start_day=0, cutoff_day=stage.cutoff_day)
        if len(encounter_groups(blocks)) > 1:
            candidates.append((stage, blocks))
    if not candidates:
        return

    seen_test = case["cohort"] == "huaxi" and str(case["case_id"]) in huaxi_test_ids
    for copy_index in range(copies):
        rng = random.Random(
            stable_int(seed, f"{case['cohort']}:{case['case_id']}:dropout:{copy_index}")
        )
        stage, source_blocks = candidates[rng.randrange(len(candidates))]
        groups = encounter_groups(source_blocks)
        requested_fraction = rng.uniform(dropout_min, dropout_max)
        remove_count = max(1, int(round(len(groups) * requested_fraction)))
        remove_count = min(len(groups) - 1, remove_count)
        removed_indices = set(rng.sample(range(len(groups)), remove_count))
        retained = [
            block
            for index, group in enumerate(groups)
            if index not in removed_indices
            for block in group
        ]
        removed_days = sorted(
            {
                int(block["end_day"])
                for index, group in enumerate(groups)
                if index in removed_indices
                for block in group
            }
        )
        view_name = f"visit_dropout_{stage.label}_{copy_index + 1}"
        yield make_view(
            case,
            retained,
            split=split,
            view_type="visit_bundle_dropout",
            view_name=view_name,
            stage=stage.label,
            window_start_day=0,
            cutoff_day=stage.cutoff_day,
            is_huaxi_seen_test=seen_test,
            extra={
                "dropout_fraction_requested": round(requested_fraction, 6),
                "source_visit_count_before_dropout": len(groups),
                "removed_visit_count": remove_count,
                "removed_visit_end_days": removed_days,
            },
        )


def training_views(
    cases: Sequence[dict[str, Any]],
    *,
    huaxi_test_ids: set[str],
    dropout_copies: int,
    seed: int,
    dropout_min: float,
    dropout_max: float,
) -> Iterator[dict[str, Any]]:
    for case in cases:
        yield from canonical_views_for_case(
            case, split="train", huaxi_test_ids=huaxi_test_ids
        )
        yield from local_window_views_for_case(
            case, split="train", huaxi_test_ids=huaxi_test_ids
        )
        yield from dropout_views_for_case(
            case,
            split="train",
            huaxi_test_ids=huaxi_test_ids,
            copies=dropout_copies,
            seed=seed,
            dropout_min=dropout_min,
            dropout_max=dropout_max,
        )


def canonical_views(
    cases: Sequence[dict[str, Any]],
    *,
    split: str,
    huaxi_test_ids: set[str],
) -> Iterator[dict[str, Any]]:
    for case in cases:
        yield from canonical_views_for_case(
            case, split=split, huaxi_test_ids=huaxi_test_ids
        )


def write_views(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    cases: set[tuple[str, str]] = set()
    cohorts: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    view_types: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            cases.add((str(row["cohort"]), str(row["case_id"])))
            cohorts[str(row["cohort"])] += 1
            stages[str(row["stage"])] += 1
            view_types[str(row["view_type"])] += 1
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "views": count,
        "unique_cases": len(cases),
        "cohorts": dict(sorted(cohorts.items())),
        "stages": dict(sorted(stages.items())),
        "view_types": dict(sorted(view_types.items())),
    }


def raw_id_set(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {source_case_id(row) for row in rows}


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_root = args.source_data_dir.resolve()
    paths = {
        "huaxi_train": source_root / "huaxi" / "huaxi_train.jsonl",
        "huaxi_test": source_root / "huaxi" / "huaxi_test.jsonl",
        "shenzhen_train": source_root / "shenzhen" / "shenzhen_train_all__full.jsonl",
        "shenzhen_test": source_root / "shenzhen" / "shenzhen_internal_test_all__full.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = {name: read_jsonl(path) for name, path in paths.items()}
    huaxi_test_ids = raw_id_set(raw["huaxi_test"])
    huaxi_raw_train_ids = raw_id_set(raw["huaxi_train"])
    raw_missing_seen_test = huaxi_test_ids - huaxi_raw_train_ids
    if raw_missing_seen_test:
        raise AssertionError(
            "This protocol requires every Huaxi test ID to already exist in raw train; "
            f"missing={len(raw_missing_seen_test)}"
        )

    parsed: dict[str, list[dict[str, Any]]] = {}
    cleaning_audit: dict[str, Counter[str]] = {}
    for name, cohort in (
        ("huaxi_train", "huaxi"),
        ("huaxi_test", "huaxi"),
        ("shenzhen_train", "shenzhen"),
        ("shenzhen_test", "shenzhen"),
    ):
        parsed[name], cleaning_audit[name] = parse_source_rows(
            raw[name], cohort=cohort, source_name=paths[name].name
        )

    huaxi_seen_reconciliation = reconcile_huaxi_seen_test_cases(
        parsed["huaxi_train"], parsed["huaxi_test"]
    )

    train_cases, validation_cases, validation_ids = partition_training_cases(
        parsed["huaxi_train"],
        parsed["shenzhen_train"],
        huaxi_test_ids=huaxi_test_ids,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    valid_huaxi_test_ids = {str(case["case_id"]) for case in parsed["huaxi_test"]}
    huaxi_train_case_ids = {
        str(case["case_id"])
        for case in train_cases
        if case["cohort"] == "huaxi"
    }
    missing_valid_seen_test = valid_huaxi_test_ids - huaxi_train_case_ids
    if missing_valid_seen_test:
        raise AssertionError(
            "Valid Huaxi seen-test cases were not retained from raw train: "
            f"{len(missing_valid_seen_test)}"
        )

    # Important: Huaxi test rows are used only below for the seen-test eval file.
    # They are never appended to train_cases.
    output_paths = {
        "train": project_root / "data" / "temporal_views" / "train_views.jsonl",
        "internal_val": project_root
        / "data"
        / "internal_val"
        / "internal_val_canonical.jsonl",
        "huaxi_seen_test": project_root
        / "data"
        / "external_test"
        / "huaxi_seen_test_canonical.jsonl",
        "shenzhen_test": project_root
        / "data"
        / "external_test"
        / "shenzhen_test_canonical.jsonl",
    }
    outputs = {
        "train": write_views(
            output_paths["train"],
            training_views(
                train_cases,
                huaxi_test_ids=huaxi_test_ids,
                dropout_copies=args.dropout_copies,
                seed=args.seed,
                dropout_min=args.dropout_min,
                dropout_max=args.dropout_max,
            ),
        ),
        "internal_val": write_views(
            output_paths["internal_val"],
            canonical_views(
                validation_cases,
                split="internal_val",
                huaxi_test_ids=huaxi_test_ids,
            ),
        ),
        "huaxi_seen_test": write_views(
            output_paths["huaxi_seen_test"],
            canonical_views(
                parsed["huaxi_test"],
                split="huaxi_seen_test",
                huaxi_test_ids=huaxi_test_ids,
            ),
        ),
        "shenzhen_test": write_views(
            output_paths["shenzhen_test"],
            canonical_views(
                parsed["shenzhen_test"],
                split="shenzhen_internal_test",
                huaxi_test_ids=huaxi_test_ids,
            ),
        ),
    }

    train_keys = {(case["cohort"], str(case["case_id"])) for case in train_cases}
    val_keys = {(case["cohort"], str(case["case_id"])) for case in validation_cases}
    huaxi_val_test_overlap = validation_ids["huaxi_val_ids"] & huaxi_test_ids
    if train_keys & val_keys or huaxi_val_test_overlap:
        raise AssertionError("Final split audit failed")

    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "project_root": str(project_root),
        "source_data_dir": str(source_root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "canonical_boundaries": {
            stage.name: stage.cutoff_day for stage in CANONICAL_STAGES
        },
        "local_windows": [
            {
                "name": name,
                "stage": stage,
                "start_day": start,
                "end_day": end,
            }
            for name, stage, start, end in LOCAL_WINDOWS
        ],
        "dropout": {
            "copies_per_training_case": args.dropout_copies,
            "minimum_fraction": args.dropout_min,
            "maximum_fraction": args.dropout_max,
            "bundle_key": ["start_day", "end_day"],
        },
        "source": {
            name: {
                "path": str(paths[name]),
                "sha256": file_sha256(paths[name]),
                "raw_rows": len(raw[name]),
                "valid_cases": len(parsed[name]),
                "cleaning_audit": counter_dict(cleaning_audit[name]),
            }
            for name in paths
        },
        "huaxi_seen_test_reconciliation": {
            "policy": (
                "user-specified train-seen alignment: never append test rows; "
                "require identical cleaned inputs; fill only missing raw-train "
                "targets from valid seen-test targets; reject valid conflicts"
            ),
            "audit": counter_dict(huaxi_seen_reconciliation),
        },
        "data_quality_thresholds": {
            "purpose": "conservative entry/unit-error masking; not diagnostic thresholds",
            "umbilical_vein_diameter_mm": {
                "valid_rule": "0 < value <= 30",
                "auto_unit_conversion": False,
            },
            "umbilical_vein_flow_ml_min": {
                "valid_rule": "0 < value <= 2000",
                "auto_unit_conversion": False,
            },
        },
        "case_splits": {
            "training_cases": len(train_cases),
            "internal_validation_cases": len(validation_cases),
            "huaxi_internal_validation_ids": len(validation_ids["huaxi_val_ids"]),
            "shenzhen_internal_validation_ids": len(
                validation_ids["shenzhen_val_ids"]
            ),
            "huaxi_raw_train_test_id_overlap": len(
                huaxi_raw_train_ids & huaxi_test_ids
            ),
            "huaxi_valid_seen_test_ids_retained_in_train": len(
                valid_huaxi_test_ids & huaxi_train_case_ids
            ),
            "huaxi_test_rows_appended_to_training": 0,
            "train_internal_val_overlap": len(train_keys & val_keys),
            "huaxi_internal_val_seen_test_overlap": len(huaxi_val_test_overlap),
        },
        "outputs": outputs,
        "schema": {
            "model_input_column": "messages",
            "target_tags": [
                "delivery_days",
                "birth_weight_g",
                "birth_length_cm",
            ],
            "delivery_target_semantics": "absolute DaysFromLMP at delivery",
            "missing_numeric_target": "NA in assistant XML and null in targets",
        },
        "invariants": {
            "canonical_boundaries_exact": [97, 160, 202, 230, 258],
            "block_inclusion_rule": "start_day >= window_start_day and end_day <= cutoff_day",
            "risk_set_rule": "delivery_days > cutoff_day",
            "huaxi_seen_test_policy": (
                "retain one raw-train row, never append huaxi_test; require identical "
                "cleaned inputs and fill only missing train targets from valid seen-test labels"
            ),
            "internal_validation_excludes_huaxi_test_ids": True,
            "gt_hint_removed": True,
            "full_cycle_summary_removed": True,
            "classification_targets_excluded": True,
            "postdelivery_records_removed": True,
            "visit_dropout_is_label_free": True,
        },
    }
    summary_path = project_root / "logs" / "prepare_temporal_views_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
