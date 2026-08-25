#!/usr/bin/env python3
"""Qwen3.5 outcome-query regression heads for ms-swift SFT.

The plugin patches Qwen3_5ForConditionalGeneration before model loading. It
adds one trainable module named ``outcome_regression`` so PEFT can preserve it
with ``--modules_to_save outcome_regression``.

Ground-truth XML remains after the query tokens in the teacher-forced sequence,
but causal attention prevents each query state from seeing those future tokens.
The plugin supports either regression-only training or ordinary next-token SFT
with a masked, standardized numeric regression loss added to the CE loss.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3_5ForConditionalGeneration

QUERY_TOKENS = (
    "<|delivery_outcome_query|>",
    "<|birth_weight_outcome_query|>",
    "<|birth_length_outcome_query|>",
)
DEFAULT_QUERY_TOKEN_IDS = (248077, 248078, 248079)
TARGET_NAMES = ("delivery_days", "birth_weight_g", "birth_length_cm")
DEFAULT_CENTERS = (275.0, 3250.0, 50.0)
DEFAULT_SCALES = (10.0, 500.0, 2.0)
PATCH_MARKER = "_outcome_regression_heads_v2_patched"
TEMPLATE_PATCH_MARKER = "_outcome_regression_fields_v2_patched"
OPTIMIZER_NAME = "outcome_differential"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


def _env_csv_ints(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(int(part.strip()) for part in raw.split(","))
    if len(values) != len(default):
        raise ValueError(f"{name} must contain {len(default)} comma-separated IDs")
    return values


class OutcomeOutputLinear(nn.Linear):
    """Marker class for the small-init scalar output layers."""


class OutcomeRegressionBlock(nn.Module):
    """Three trainable query embeddings and three independent numeric heads."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        head_arch = _env_choice(
            "OUTCOME_HEAD_ARCH",
            "mlp",
            {"linear", "mlp"},
        )
        self.query_embeddings = nn.Embedding(3, hidden_size)
        if head_arch == "linear":
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        OutcomeOutputLinear(hidden_size, 1),
                    )
                    for _ in TARGET_NAMES
                ]
            )
        else:
            head_hidden = _env_int("OUTCOME_HEAD_HIDDEN_SIZE", 1024)
            dropout = _env_float("OUTCOME_HEAD_DROPOUT", 0.10)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, head_hidden),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                        OutcomeOutputLinear(head_hidden, 1),
                    )
                    for _ in TARGET_NAMES
                ]
            )
        self.register_buffer(
            "target_centers",
            torch.tensor(DEFAULT_CENTERS, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "target_scales",
            torch.tensor(DEFAULT_SCALES, dtype=torch.float32),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_embeddings.weight, mean=0.0, std=0.02)
        for head in self.heads:
            final = head[-1]
            nn.init.normal_(
                final.weight,
                mean=0.0,
                std=_env_float("OUTCOME_OUTPUT_INIT_STD", 1e-3),
            )
            nn.init.zeros_(final.bias)

    @torch.no_grad()
    def restore_target_buffers(self) -> None:
        self.target_centers.copy_(
            torch.tensor(
                DEFAULT_CENTERS,
                dtype=self.target_centers.dtype,
                device=self.target_centers.device,
            )
        )
        self.target_scales.copy_(
            torch.tensor(
                DEFAULT_SCALES,
                dtype=self.target_scales.dtype,
                device=self.target_scales.device,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        operation: str,
        token_embeddings: torch.Tensor | None = None,
        query_token_ids: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if operation == "replace_embeddings":
            if token_embeddings is None or query_token_ids is None:
                raise ValueError("embedding replacement requires embeddings and IDs")
            query_index = torch.full_like(x, -1)
            for index, token_id in enumerate(query_token_ids):
                query_index = torch.where(
                    x == token_id,
                    torch.full_like(query_index, index),
                    query_index,
                )
            replacement_mask = query_index >= 0
            safe_index = query_index.clamp_min(0)
            query_vectors = self.query_embeddings(safe_index)
            return torch.where(
                replacement_mask.unsqueeze(-1),
                query_vectors.to(token_embeddings.dtype),
                token_embeddings,
            )

        if operation == "predict":
            if x.ndim != 3 or x.shape[1] != 3:
                raise ValueError(
                    f"query hidden states must have shape [batch, 3, hidden], got {x.shape}"
                )
            predictions = [
                head(x[:, index]).squeeze(-1)
                for index, head in enumerate(self.heads)
            ]
            return torch.stack(predictions, dim=-1)

        raise ValueError(f"unsupported operation: {operation}")


def _text_hidden_size(config: Any) -> int:
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("unable to resolve Qwen3.5 text hidden size")
    return int(hidden_size)


def _install_runtime_hooks(model: Qwen3_5ForConditionalGeneration) -> None:
    embedding = model.get_input_embeddings()
    if getattr(model, "_outcome_embedding_module", None) is not embedding:
        old_handle = getattr(model, "_outcome_embedding_hook_handle", None)
        if old_handle is not None:
            old_handle.remove()

        def replace_query_embeddings(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            del module
            input_ids = args[0]
            return model.outcome_regression(
                input_ids,
                operation="replace_embeddings",
                token_embeddings=output,
                query_token_ids=model._outcome_query_token_ids,
            )

        model._outcome_embedding_hook_handle = embedding.register_forward_hook(
            replace_query_embeddings
        )
        model._outcome_embedding_module = embedding

    final_norm = model.model.language_model.norm
    if getattr(model, "_outcome_final_norm_module", None) is not final_norm:
        old_handle = getattr(model, "_outcome_final_norm_hook_handle", None)
        if old_handle is not None:
            old_handle.remove()

        def capture_final_hidden(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            del module, args
            model._outcome_last_hidden = output
            return output

        model._outcome_final_norm_hook_handle = final_norm.register_forward_hook(
            capture_final_hidden
        )
        model._outcome_final_norm_module = final_norm


def _patch_qwen35_template() -> None:
    """Preserve regression fields through ms-swift's multimodal pre-hook.

    Qwen3.5's training template replaces ``input_ids`` with ``inputs_embeds``.
    The generic pre-hook also drops non-standard batch fields. Carry a private
    copy of the token IDs plus the targets/mask in the returned mapping so the
    model patch can still locate and replace the three query embeddings.
    """

    from swift.template.templates.qwen import Qwen3_5Template

    if getattr(Qwen3_5Template, TEMPLATE_PATCH_MARKER, False):
        return
    original_post_encode = Qwen3_5Template._post_encode

    def patched_post_encode(
        self: Any,
        model: nn.Module,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        encoded = original_post_encode(self, model, inputs)
        if "input_ids" in inputs:
            encoded["_outcome_input_ids"] = inputs["input_ids"]
        for key in ("outcome_targets", "outcome_mask"):
            if key in inputs:
                encoded[key] = inputs[key]
        return encoded

    Qwen3_5Template._post_encode = patched_post_encode
    setattr(Qwen3_5Template, TEMPLATE_PATCH_MARKER, True)


def _query_hidden_states(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    query_token_ids: tuple[int, int, int],
) -> torch.Tensor:
    gathered = []
    for token_id in query_token_ids:
        matches = input_ids == token_id
        counts = matches.sum(dim=-1)
        if not torch.all(counts == 1):
            raise ValueError(
                f"query token id {token_id} must occur exactly once per sample; "
                f"counts={counts.detach().cpu().tolist()}"
            )
        positions = matches.to(torch.int64).argmax(dim=-1)
        batch_index = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )
        gathered.append(hidden_states[batch_index, positions])
    return torch.stack(gathered, dim=1)


def _regression_loss(
    prediction_norm: torch.Tensor,
    outcome_targets: Any,
    outcome_mask: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.as_tensor(
        outcome_targets,
        dtype=torch.float32,
        device=prediction_norm.device,
    )
    mask = torch.as_tensor(
        outcome_mask,
        dtype=torch.bool,
        device=prediction_norm.device,
    )
    if targets.shape != prediction_norm.shape or mask.shape != prediction_norm.shape:
        raise ValueError(
            "outcome targets/mask must match prediction shape; "
            f"prediction={prediction_norm.shape}, targets={targets.shape}, mask={mask.shape}"
        )
    centers = torch.tensor(
        DEFAULT_CENTERS, dtype=torch.float32, device=prediction_norm.device
    )
    scales = torch.tensor(
        DEFAULT_SCALES, dtype=torch.float32, device=prediction_norm.device
    )
    target_norm = (targets - centers) / scales
    loss_type = _env_choice(
        "OUTCOME_LOSS_TYPE",
        "smooth_l1",
        {"smooth_l1", "mse_centered"},
    )
    if loss_type == "smooth_l1":
        element_loss = F.smooth_l1_loss(
            prediction_norm.float(),
            target_norm,
            reduction="none",
            beta=_env_float("OUTCOME_HUBER_BETA", 0.5),
        )
    else:
        element_loss = (prediction_norm.float() - target_norm).square()
    task_losses = []
    for index in range(3):
        task_mask = mask[:, index]
        if task_mask.any():
            task_loss = element_loss[task_mask, index].mean()
            if loss_type == "mse_centered" and int(task_mask.sum()) > 1:
                prediction_valid = prediction_norm.float()[task_mask, index]
                target_valid = target_norm[task_mask, index]
                centered_error = (
                    prediction_valid - prediction_valid.mean()
                ) - (
                    target_valid - target_valid.mean()
                )
                task_loss = task_loss + _env_float(
                    "OUTCOME_CENTERED_LOSS_WEIGHT", 0.5
                ) * centered_error.square().mean()
            task_losses.append(task_loss)
        else:
            task_losses.append(element_loss[:, index].sum() * 0.0)
    active = torch.tensor(
        [bool(mask[:, index].any()) for index in range(3)],
        dtype=torch.bool,
        device=prediction_norm.device,
    )
    stacked = torch.stack(task_losses)
    loss = stacked[active].mean()
    prediction_raw = prediction_norm.float() * scales + centers
    return loss, stacked, prediction_raw


def patch_qwen35() -> None:
    cls = Qwen3_5ForConditionalGeneration
    if getattr(cls, PATCH_MARKER, False):
        return

    original_init = cls.__init__
    original_init_weights = cls._init_weights
    original_forward = cls.forward

    def patched_init_weights(
        self: Qwen3_5ForConditionalGeneration,
        module: nn.Module,
    ) -> None:
        # Hugging Face initializes plugin parameters as missing base-model
        # weights. Handle the tagged scalar layers here, before DDP/DeepSpeed
        # creates optimizer master weights, instead of resetting at first use.
        if isinstance(module, OutcomeOutputLinear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=_env_float("OUTCOME_OUTPUT_INIT_STD", 1e-3),
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return
        if isinstance(module, OutcomeRegressionBlock):
            module.restore_target_buffers()
            return
        original_init_weights(self, module)

    def patched_init(self: Qwen3_5ForConditionalGeneration, config: Any) -> None:
        original_init(self, config)
        self.outcome_regression = OutcomeRegressionBlock(_text_hidden_size(config))
        self._outcome_query_token_ids = _env_csv_ints(
            "OUTCOME_QUERY_TOKEN_IDS",
            DEFAULT_QUERY_TOKEN_IDS,
        )
        self._outcome_last_hidden = None
        self._outcome_embedding_module = None
        self._outcome_final_norm_module = None
        self._outcome_loss_logged = False

    def patched_forward(
        self: Qwen3_5ForConditionalGeneration,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        outcome_targets = kwargs.pop("outcome_targets", None)
        outcome_mask = kwargs.pop("outcome_mask", None)
        kwargs.pop("num_items_in_batch", None)
        loss_mode = _env_choice(
            "OUTCOME_LOSS_MODE",
            "regression_only",
            {"ce_plus_regression", "regression_only"},
        )
        ce_labels = kwargs.pop("labels", None)
        ce_logits_to_keep = kwargs.get("logits_to_keep")
        # Compute CE below only at positions whose shifted label is valid.
        # This avoids materializing a full-vocabulary tensor for the padded
        # common suffix that ms-swift uses with variable-length batches.
        kwargs["logits_to_keep"] = 1
        kwargs["return_dict"] = True

        input_ids = kwargs.pop("_outcome_input_ids", None)
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise ValueError("outcome regression requires input_ids")

        _install_runtime_hooks(self)
        if kwargs.get("inputs_embeds") is not None:
            kwargs["inputs_embeds"] = self.outcome_regression(
                input_ids,
                operation="replace_embeddings",
                token_embeddings=kwargs["inputs_embeds"],
                query_token_ids=self._outcome_query_token_ids,
            )
        self._outcome_last_hidden = None
        outputs = original_forward(self, *args, **kwargs)
        hidden_states = self._outcome_last_hidden
        self._outcome_last_hidden = None
        if hidden_states is None:
            raise RuntimeError("final hidden-state hook did not run")

        query_counts = [
            (input_ids == token_id).sum(dim=-1)
            for token_id in self._outcome_query_token_ids
        ]
        queries_exact = all(torch.all(counts == 1) for counts in query_counts)
        if not queries_exact:
            # During cached autoregressive decoding, only the newest generated
            # token is passed back through forward, so the three query tokens
            # are intentionally absent. The language-model logits remain
            # valid; numeric predictions were already available at prefill.
            is_cached_decode = kwargs.get("past_key_values") is not None
            if (
                not is_cached_decode
                or outcome_targets is not None
                or outcome_mask is not None
                or ce_labels is not None
            ):
                counts = [item.detach().cpu().tolist() for item in query_counts]
                raise ValueError(
                    "outcome query tokens must occur exactly once outside cached "
                    f"generation; counts={counts}"
                )
            return outputs

        ce_loss = None
        ce_selected_tokens = 0
        if loss_mode == "ce_plus_regression" and ce_labels is not None:
            if isinstance(ce_logits_to_keep, torch.Tensor):
                ce_hidden = hidden_states[:, ce_logits_to_keep, :]
            else:
                keep = int(ce_labels.shape[-1])
                ce_hidden = hidden_states[:, -keep:, :]
            shift_hidden = ce_hidden[:, :-1, :]
            shift_labels = ce_labels[:, 1:]
            valid_ce = shift_labels != -100
            ce_selected_tokens = int(valid_ce.sum().item())
            if ce_selected_tokens == 0:
                raise RuntimeError("ce_plus_regression found no valid next-token labels")
            selected_hidden = shift_hidden[valid_ce]
            selected_labels = shift_labels[valid_ce]
            selected_logits = self.lm_head(selected_hidden)
            ce_loss = F.cross_entropy(
                selected_logits.float(),
                selected_labels,
                reduction="mean",
            )

        query_hidden = _query_hidden_states(
            input_ids,
            hidden_states,
            self._outcome_query_token_ids,
        )
        prediction_norm = self.outcome_regression(
            query_hidden,
            operation="predict",
        )
        centers = torch.tensor(
            DEFAULT_CENTERS, dtype=torch.float32, device=prediction_norm.device
        )
        scales = torch.tensor(
            DEFAULT_SCALES, dtype=torch.float32, device=prediction_norm.device
        )
        prediction_raw = prediction_norm.float() * scales + centers

        if outcome_targets is not None and outcome_mask is not None:
            regression_loss, task_losses, prediction_raw = _regression_loss(
                prediction_norm,
                outcome_targets,
                outcome_mask,
            )
            if loss_mode == "ce_plus_regression":
                if ce_loss is None:
                    raise RuntimeError(
                        "ce_plus_regression requires teacher-forced labels and CE loss"
                    )
                outputs.loss = (
                    _env_float("OUTCOME_CE_WEIGHT", 1.0) * ce_loss
                    + _env_float("OUTCOME_REG_WEIGHT", 1.0) * regression_loss
                )
                outputs["outcome_ce_loss"] = ce_loss.detach()
                if not self._outcome_loss_logged:
                    if int(os.getenv("RANK", "0")) == 0:
                        print(
                            "[outcome_joint_loss] "
                            f"ce={ce_loss.detach().float().item():.6f} "
                            f"reg={regression_loss.detach().float().item():.6f} "
                            f"total={outputs.loss.detach().float().item():.6f} "
                            f"input_shape={tuple(input_ids.shape)} "
                            f"labels_shape={tuple(ce_labels.shape)} "
                            f"framework_logits_to_keep={ce_logits_to_keep} "
                            f"selected_ce_tokens={ce_selected_tokens}",
                            flush=True,
                        )
                    self._outcome_loss_logged = True
            else:
                outputs.loss = regression_loss
            outputs["outcome_regression_loss"] = regression_loss.detach()
            outputs["outcome_task_losses"] = task_losses.detach()
        elif loss_mode == "regression_only":
            outputs.loss = None

        outputs["outcome_predictions"] = prediction_raw
        outputs["outcome_predictions_normalized"] = prediction_norm
        return outputs

    cls.__init__ = patched_init
    cls._init_weights = patched_init_weights
    cls.forward = patched_forward
    setattr(cls, PATCH_MARKER, True)
    print(
        "[outcome_regression_plugin:v7] patched Qwen3_5ForConditionalGeneration "
        f"query_ids={DEFAULT_QUERY_TOKEN_IDS} targets={','.join(TARGET_NAMES)}"
    )


_patch_qwen35_template()
patch_qwen35()


def register_outcome_optimizer() -> None:
    """Register head-only/joint parameter groups for two-stage training."""

    from swift.optimizers import OptimizerCallback, optimizers_map
    from swift.utils import get_logger
    from transformers import Trainer as HfTrainer

    if OPTIMIZER_NAME in optimizers_map:
        return
    logger = get_logger()

    class OutcomeDifferentialOptimizerCallback(OptimizerCallback):

        def create_optimizer(self, model: nn.Module | None = None):
            args = self.args
            model = model or self.trainer.model
            mode = os.getenv("OUTCOME_OPTIMIZER_MODE", "joint").strip().lower()
            if mode not in {"head_only", "joint"}:
                raise ValueError(
                    "OUTCOME_OPTIMIZER_MODE must be head_only or joint, "
                    f"got {mode!r}"
                )
            head_lr = _env_float("OUTCOME_HEAD_LR", float(args.learning_rate))
            lora_lr = _env_float("OUTCOME_LORA_LR", float(args.learning_rate))
            head_weight_decay = _env_float("OUTCOME_HEAD_WEIGHT_DECAY", 0.01)

            # PEFT keeps a frozen original copy beside modules_to_save.default.
            # Never reactivate that original copy; only the active adapter copy
            # belongs in the optimizer.
            for name, parameter in model.named_parameters():
                if "outcome_regression.original_module" in name:
                    parameter.requires_grad_(False)
                elif "outcome_regression" in name:
                    parameter.requires_grad_(True)
                elif "lora_" in name:
                    parameter.requires_grad_(mode == "joint")

            named_head = []
            named_lora = []
            unexpected = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if "outcome_regression" in name:
                    named_head.append((name, parameter))
                elif "lora_" in name:
                    named_lora.append((name, parameter))
                else:
                    unexpected.append(name)
            if unexpected:
                raise RuntimeError(
                    "unexpected trainable parameters outside outcome/LoRA groups: "
                    f"{unexpected[:20]}"
                )
            if not named_head:
                raise RuntimeError("outcome optimizer found no trainable head/query parameters")
            if mode == "joint" and not named_lora:
                raise RuntimeError("joint outcome optimizer found no trainable LoRA parameters")
            if mode == "head_only" and named_lora:
                raise RuntimeError("head-only outcome optimizer failed to freeze LoRA")

            decay_names = set(HfTrainer.get_decay_parameter_names(None, model))
            groups = []

            def append_groups(named_parameters, lr: float, decay: float, label: str):
                for use_decay in (True, False):
                    selected = [
                        parameter
                        for name, parameter in named_parameters
                        if (name in decay_names) == use_decay
                    ]
                    if selected:
                        suffix = "decay" if use_decay else "no_decay"
                        groups.append(
                            {
                                "params": selected,
                                "lr": lr,
                                "weight_decay": decay if use_decay else 0.0,
                                "outcome_group": f"{label}_{suffix}",
                            }
                        )

            append_groups(named_head, head_lr, head_weight_decay, "head")
            if mode == "joint":
                append_groups(named_lora, lora_lr, float(args.weight_decay), "lora")

            optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(
                args, model
            )
            if int(os.getenv("RANK", "0")) == 0:
                logger.info(
                    "outcome optimizer: mode=%s head_lr=%s lora_lr=%s "
                    "head_tensors=%s head_params=%s lora_tensors=%s lora_params=%s "
                    "groups=%s",
                    mode,
                    head_lr,
                    lora_lr,
                    len(named_head),
                    sum(parameter.numel() for _, parameter in named_head),
                    len(named_lora),
                    sum(parameter.numel() for _, parameter in named_lora),
                    [
                        {
                            "name": group["outcome_group"],
                            "lr": group["lr"],
                            "weight_decay": group["weight_decay"],
                            "tensors": len(group["params"]),
                        }
                        for group in groups
                    ],
                )
            return optimizer_cls(groups, **optimizer_kwargs)

    optimizers_map[OPTIMIZER_NAME] = OutcomeDifferentialOptimizerCallback
    print(f"[outcome_regression_plugin:v5] registered optimizer={OPTIMIZER_NAME}")


register_outcome_optimizer()
