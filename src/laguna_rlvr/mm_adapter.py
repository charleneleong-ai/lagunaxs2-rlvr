"""Config guardrails for a frozen-backbone generalized adapter experiment.

This module intentionally avoids torch/transformers imports. It is the cheap control plane for
deciding whether an experiment fits its declared GPU budget before any heavyweight training script
runs — the budget is read from the config (`[gpu_guardrails].max_vram_gb`), not hardcoded, so the
same check serves a 40GB-NVFP4 proof of concept and an 80GB-BF16 run honestly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Rough bytes/param by quantization (experts dominate the footprint). Substring match on the config's
# quantization string; default to BF16 (the conservative, unquantized assumption) when unspecified.
_BYTES_PER_PARAM = {"nvfp4": 0.5, "int4": 0.5, "4bit": 0.5, "fp8": 1.0, "8bit": 1.0, "bf16": 2.0, "fp16": 2.0}


def _bytes_per_param(quantization: str) -> float:
    q = quantization.lower()
    return next((b for key, b in _BYTES_PER_PARAM.items() if key in q), 2.0)


@dataclass(frozen=True, slots=True)
class AdapterPlan:
    name: str
    backbone_model: str
    backbone_quantization: str
    backbone_params_b: float
    freeze_backbone: bool
    max_sequence_length: int
    modality: str
    encoder_model: str
    encoder_role: str
    freeze_encoder: bool
    fallback_encoder_model: str
    adapter_kind: str
    objective: str
    train_projector: bool
    lora_enabled: bool
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_vram_gb: float
    reserve_vram_gb: float

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def backbone_vram_gb(self) -> float:
        """Estimated weight footprint of the (frozen) backbone at its quantization."""
        return self.backbone_params_b * _bytes_per_param(self.backbone_quantization)


def _table(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key!r} must be a table")
    return value


def plan_from_config(cfg: dict[str, Any]) -> AdapterPlan:
    """Normalize a TOML-loaded adapter config into one small dataclass."""
    backbone = _table(cfg, "backbone")
    modality = _table(cfg, "modality")
    adapter = _table(cfg, "adapter")
    training = _table(cfg, "training")
    guardrails = _table(cfg, "gpu_guardrails")

    return AdapterPlan(
        name=str(cfg.get("name", "laguna-generalized-adapter")),
        backbone_model=str(backbone.get("model_id", "")),
        backbone_quantization=str(backbone.get("quantization", "unknown")),
        backbone_params_b=float(backbone.get("params_b", 0)),
        freeze_backbone=bool(backbone.get("freeze", True)),
        max_sequence_length=int(backbone.get("max_sequence_length", 8192)),
        modality=str(modality.get("kind", "image")),
        encoder_model=str(modality.get("encoder_id", "")),
        encoder_role=str(modality.get("encoder_role", "generic_visual_encoder")),
        freeze_encoder=bool(modality.get("freeze_encoder", True)),
        fallback_encoder_model=str(modality.get("fallback_encoder_id", "")),
        adapter_kind=str(adapter.get("kind", "projector")),
        objective=str(adapter.get("objective", "visual_alignment")),
        train_projector=bool(adapter.get("train_projector", True)),
        lora_enabled=bool(training.get("lora_enabled", False)),
        micro_batch_size=int(training.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 16)),
        max_vram_gb=float(guardrails.get("max_vram_gb", 40)),
        reserve_vram_gb=float(guardrails.get("reserve_vram_gb", 4)),
    )


def validate_gpu_budget(plan: AdapterPlan) -> list[str]:
    """Return blocking issues for the projector-only stage against the config's own VRAM budget."""
    issues: list[str] = []
    if plan.backbone_params_b > 0:
        need = plan.backbone_vram_gb + plan.reserve_vram_gb
        if need > plan.max_vram_gb:
            issues.append(
                f"backbone ~{plan.backbone_vram_gb:.0f}GB ({plan.backbone_quantization}) + "
                f"{plan.reserve_vram_gb:g}GB reserved exceeds the {plan.max_vram_gb:g}GB budget — "
                f"use a smaller-precision backbone or a larger-VRAM config"
            )
    if not plan.freeze_backbone:
        issues.append("freeze the backbone; full-model training is out of scope for this stage")
    if not plan.freeze_encoder:
        issues.append("freeze the modality encoder for the first alignment stage")
    if not plan.train_projector:
        issues.append("train_projector must be true; otherwise there is no native adapter to learn")
    if plan.micro_batch_size != 1:
        issues.append("use micro_batch_size=1 and gradient accumulation for the single-GPU stage")
    if plan.max_sequence_length > 8192:
        issues.append("start with max_sequence_length<=8192; long-context multimodal RL can come later")
    if plan.lora_enabled:
        issues.append("disable LoRA for the first projector-only stage; add QLoRA after the projector learns")
    return issues


def render_plan(plan: AdapterPlan) -> str:
    """Human-readable summary for dry runs and PR review."""
    lines = [
        f"Experiment: {plan.name}",
        f"Backbone: {plan.backbone_model} ({plan.backbone_quantization}, frozen={plan.freeze_backbone}, "
        f"~{plan.backbone_vram_gb:.0f}GB)",
        f"Modality: {plan.modality} via {plan.encoder_model} as {plan.encoder_role} "
        f"(frozen={plan.freeze_encoder})",
        f"Adapter: {plan.adapter_kind}, objective={plan.objective}, train_projector={plan.train_projector} "
        f"(vision tokens = encoder patches ÷ pool)",
        f"Training: micro_batch={plan.micro_batch_size}, grad_accum={plan.gradient_accumulation_steps}, "
        f"effective_batch={plan.effective_batch_size}",
        f"Guardrail: {plan.max_vram_gb:g}GB VRAM with {plan.reserve_vram_gb:g}GB reserved",
    ]
    if plan.fallback_encoder_model:
        lines.append(f"Fallback encoder: {plan.fallback_encoder_model}")
    issues = validate_gpu_budget(plan)
    if issues:
        lines.append("Blocking issues:")
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("GPU guardrails: pass")
    return "\n".join(lines)
