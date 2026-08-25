#!/usr/bin/env python3
"""Distributed inference on the model-agnostic core robustness benchmark."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = (
    PROJECT_ROOT / "benchmarks/model_agnostic_robustness_v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/benchmark_run"
GENERATE_MODELS = frozenset({"hulumed", "qwen_base", "lingshu"})
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
TARGET_NAMES = ("delivery_days", "birth_weight_g", "birth_length_cm")
QUERY_TOKENS = (
    "<|delivery_outcome_query|>",
    "<|birth_weight_outcome_query|>",
    "<|birth_length_outcome_query|>",
)
QUERY_SUFFIX = "\n\n[内部结局回归查询]\n" + "\n".join(QUERY_TOKENS)
XML_PATTERNS = {
    name: re.compile(
        rf"<\s*{re.escape(name)}\s*>\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
        rf"<\s*/\s*{re.escape(name)}\s*>",
        flags=re.IGNORECASE,
    )
    for name in TARGET_NAMES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("ours", "hulumed", "qwen_base", "lingshu"),
        required=True,
    )
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ours-base", type=Path)
    parser.add_argument("--ours-checkpoint", type=Path)
    parser.add_argument("--hulumed-model", type=Path)
    parser.add_argument("--lingshu-model", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Optional single merged JSONL. When set, ignores --variants loop.",
    )
    parser.add_argument(
        "--predictions-name",
        type=str,
        default="merged_all_variants",
        help="Prediction shard basename when --input-jsonl is used.",
    )
    parser.add_argument("--split", choices=("core", "full"), default="core")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=24000)
    parser.add_argument("--tokenize-chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples-per-variant", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_model_paths(args: argparse.Namespace) -> None:
    required = {
        "ours": ("ours_base", "ours_checkpoint"),
        "qwen_base": ("ours_base",),
        "hulumed": ("hulumed_model",),
        "lingshu": ("lingshu_model",),
    }[args.model]
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"model={args.model} requires {', '.join(missing)}")
    for name in required:
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(path)


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_local_rows(
    path: Path, rank: int, world_size: int, limit: int | None
) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if index % world_size == rank:
                yield index, json.loads(line)


def chunks(iterator: Iterator[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def dynamic_batches(
    encoded: list[tuple[int, dict[str, Any], list[int]]],
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[list[tuple[int, dict[str, Any], list[int]]]]:
    encoded.sort(key=lambda item: len(item[2]))
    result: list[list[tuple[int, dict[str, Any], list[int]]]] = []
    current: list[tuple[int, dict[str, Any], list[int]]] = []
    current_max = 0
    for item in encoded:
        candidate_max = max(current_max, len(item[2]))
        candidate_size = len(current) + 1
        if current and (
            candidate_size > max_batch_size
            or candidate_max * candidate_size > max_batch_tokens
        ):
            result.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, len(item[2]))
    if current:
        result.append(current)
    return result


def common_record(row: dict[str, Any], source_index: int, input_length: int) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "benchmark_id": row["benchmark_id"],
        "pair_id": row["pair_id"],
        "case_id": row["case_id"],
        "cohort": row["cohort"],
        "variant_name": row["variant_name"],
        "cutoff_bin": row["cutoff_bin"],
        "continuous_slot": row["continuous_slot"],
        "cutoff_day": row["cutoff_day"],
        "window_start_day": row["window_start_day"],
        "visible_visit_count": row["visible_visit_count"],
        "visible_record_count": row["visible_record_count"],
        "input_length": input_length,
    }


def parse_xml(text: str) -> tuple[list[float | None], list[int]]:
    predictions: list[float | None] = []
    mask: list[int] = []
    for name in TARGET_NAMES:
        match = XML_PATTERNS[name].search(text)
        value = float(match.group(1)) if match else None
        if value is not None and not math.isfinite(value):
            value = None
        predictions.append(value)
        mask.append(int(value is not None))
    return predictions, mask


def prepare_ours(args: argparse.Namespace, local_rank: int) -> tuple[Any, Any]:
    os.environ.setdefault("OUTCOME_HEAD_ARCH", "linear")
    os.environ.setdefault("OUTCOME_LOSS_MODE", "regression_only")
    plugin = import_module(
        PROJECT_ROOT / "plugins/outcome_regression.py",
        "core_benchmark_outcome_plugin",
    )
    processor = AutoProcessor.from_pretrained(str(args.ours_base), trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": list(plugin.QUERY_TOKENS)})
    query_ids = tuple(tokenizer.convert_tokens_to_ids(list(plugin.QUERY_TOKENS)))
    if query_ids != tuple(plugin.DEFAULT_QUERY_TOKEN_IDS):
        raise ValueError(f"unexpected outcome query IDs: {query_ids}")
    tokenizer.padding_side = "right"
    model = plugin.Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.ours_base),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=f"cuda:{local_rank}",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model, str(args.ours_checkpoint), is_trainable=False
    )
    model.config.use_cache = False
    model.eval()
    return tokenizer, model


def prepare_hulumed(args: argparse.Namespace, local_rank: int) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.hulumed_model), trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(args.hulumed_model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=f"cuda:{local_rank}",
    )
    model.eval()
    return tokenizer, model


def prepare_qwen_base(args: argparse.Namespace, local_rank: int) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(str(args.ours_base), trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(args.ours_base),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=f"cuda:{local_rank}",
    )
    model.eval()
    return tokenizer, model


def prepare_lingshu(args: argparse.Namespace, local_rank: int) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.lingshu_model), trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 32768
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.lingshu_model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=f"cuda:{local_rank}",
    )
    model.config.use_cache = True
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.use_cache = True
    model.eval()
    return tokenizer, model


def encode_chunk(
    model_name: str,
    tokenizer: Any,
    chunk: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any], list[int]]]:
    conversations = []
    for _, row in chunk:
        messages = copy.deepcopy(row["messages"])
        if model_name == "ours":
            messages[-1]["content"] += QUERY_SUFFIX
        conversations.append(messages)
    if model_name == "ours":
        encoding = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids = encoding["input_ids"]
    elif model_name == "qwen_base":
        encoding = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = encoding["input_ids"]
    else:
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in conversations
        ]
        input_ids = tokenizer(
            texts, add_special_tokens=False, truncation=False
        )["input_ids"]
    return [
        (source_index, row, ids)
        for (source_index, row), ids in zip(chunk, input_ids, strict=True)
    ]


def infer_batch(
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch: list[tuple[int, dict[str, Any], list[int]]],
) -> list[dict[str, Any]]:
    max_length = max(len(item[2]) for item in batch)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids = torch.full(
        (len(batch), max_length), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for batch_index, (_, _, ids) in enumerate(batch):
        values = torch.tensor(ids, dtype=torch.long, device=device)
        if args.model in GENERATE_MODELS:
            input_ids[batch_index, -len(ids) :] = values
            attention_mask[batch_index, -len(ids) :] = 1
        else:
            input_ids[batch_index, : len(ids)] = values
            attention_mask[batch_index, : len(ids)] = 1

    with torch.inference_mode():
        if args.model == "ours":
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            numeric = output.outcome_predictions.detach().float().cpu().tolist()
            generated = [None] * len(batch)
            generated_token_counts = [0] * len(batch)
        else:
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = sequences[:, max_length:]
            generated = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            generated_token_counts = [
                int((suffix != pad_id).sum().item()) for suffix in new_tokens
            ]
            numeric = [parse_xml(text)[0] for text in generated]

    records = []
    for position, (item, prediction) in enumerate(zip(batch, numeric, strict=True)):
        source_index, row, ids = item
        record = common_record(row, source_index, len(ids))
        record["model"] = args.model
        record["predictions"] = prediction
        if args.model == "ours":
            record["parse_mask"] = [1, 1, 1]
        else:
            assert generated[position] is not None
            _, parse_mask = parse_xml(generated[position])
            record["parse_mask"] = parse_mask
            record["generated_tokens"] = generated_token_counts[position]
            record["raw_output"] = generated[position]
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    validate_model_paths(args)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_output = args.output_dir / "predictions" / args.model
    model_output.mkdir(parents=True, exist_ok=True)
    if args.model == "ours":
        tokenizer, model = prepare_ours(args, local_rank)
    elif args.model == "qwen_base":
        tokenizer, model = prepare_qwen_base(args, local_rank)
    elif args.model == "lingshu":
        tokenizer, model = prepare_lingshu(args, local_rank)
    else:
        tokenizer, model = prepare_hulumed(args, local_rank)

    if args.input_jsonl is not None:
        jobs = [(args.predictions_name, args.input_jsonl.resolve())]
    else:
        jobs = [
            (variant, args.benchmark_root / args.split / variant / "test.jsonl")
            for variant in args.variants
        ]

    for job_name, input_path in jobs:
        output_path = model_output / f"{job_name}.rank{rank}.jsonl"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] model={args.model} rank={rank} job={job_name}", flush=True)
            if world_size > 1:
                dist.barrier()
            continue
        temporary = output_path.with_suffix(".jsonl.tmp")
        started = time.monotonic()
        processed = 0
        with temporary.open("w", encoding="utf-8") as writer:
            local_rows = read_local_rows(
                input_path, rank, world_size, args.max_samples_per_variant
            )
            for chunk in chunks(local_rows, args.tokenize_chunk_size):
                encoded = encode_chunk(args.model, tokenizer, chunk)
                for batch in dynamic_batches(
                    encoded, args.max_batch_size, args.max_batch_tokens
                ):
                    for record in infer_batch(args, tokenizer, model, device, batch):
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += len(batch)
                if rank == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"[progress] model={args.model} job={job_name} "
                        f"rank0={processed} speed={processed / elapsed:.2f}/s",
                        flush=True,
                    )
        os.replace(temporary, output_path)
        print(
            f"[done] model={args.model} rank={rank} job={job_name} rows={processed}",
            flush=True,
        )
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        run_config = {
            "model": args.model,
            "world_size": world_size,
            "benchmark_root": str(args.benchmark_root.resolve()),
            "split": args.split,
            "input_jsonl": (
                str(args.input_jsonl.resolve()) if args.input_jsonl is not None else None
            ),
            "predictions_name": (
                args.predictions_name if args.input_jsonl is not None else None
            ),
            "variants": args.variants if args.input_jsonl is None else None,
            "max_samples_per_variant": args.max_samples_per_variant,
            "max_batch_size": args.max_batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "max_new_tokens": (
                args.max_new_tokens if args.model in GENERATE_MODELS else None
            ),
            "ours_base": (
                str(args.ours_base.resolve())
                if args.model in {"ours", "qwen_base"}
                else None
            ),
            "ours_checkpoint": (
                str(args.ours_checkpoint.resolve()) if args.model == "ours" else None
            ),
            "hulumed_model": (
                str(args.hulumed_model.resolve()) if args.model == "hulumed" else None
            ),
            "lingshu_model": (
                str(args.lingshu_model.resolve()) if args.model == "lingshu" else None
            ),
        }
        (model_output / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
