"""Config guardrails for a frozen-backbone generalized adapter experiment.

This module intentionally avoids torch/transformers imports. It is the cheap control plane for
deciding whether an experiment is shaped like an A100-40GB proof of concept before any heavyweight
training script is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AdapterPlan:
    name: str
    backbone_model: str
    backbone_quantization: str
    freeze_backbone: bool
    max_sequence_length: int
    modality: str
    encoder_model: str
    encoder_role: str
    freeze_encoder: bool
    fallback_encoder_model: str
    adapter_kind: str
    objective: str
    output_tokens: int
    train_projector: bool
    lora_enabled: bool
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_vram_gb: float
    reserve_vram_gb: float
    max_output_tokens: int

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


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
    guardrails = _table(cfg, "a100_40gb_guardrails")

    return AdapterPlan(
        name=str(cfg.get("name", "laguna-generalized-adapter")),
        backbone_model=str(backbone.get("model_id", "")),
        backbone_quantization=str(backbone.get("quantization", "unknown")),
        freeze_backbone=bool(backbone.get("freeze", True)),
        max_sequence_length=int(backbone.get("max_sequence_length", 8192)),
        modality=str(modality.get("kind", "image")),
        encoder_model=str(modality.get("encoder_id", "")),
        encoder_role=str(modality.get("encoder_role", "generic_visual_encoder")),
        freeze_encoder=bool(modality.get("freeze_encoder", True)),
        fallback_encoder_model=str(modality.get("fallback_encoder_id", "")),
        adapter_kind=str(adapter.get("kind", "projector")),
        objective=str(adapter.get("objective", "visual_alignment")),
        output_tokens=int(adapter.get("output_tokens", 64)),
        train_projector=bool(adapter.get("train_projector", True)),
        lora_enabled=bool(training.get("lora_enabled", False)),
        micro_batch_size=int(training.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 16)),
        max_vram_gb=float(guardrails.get("max_vram_gb", 40)),
        reserve_vram_gb=float(guardrails.get("reserve_vram_gb", 4)),
        max_output_tokens=int(guardrails.get("max_output_tokens", 128)),
    )


def validate_a100_40gb(plan: AdapterPlan) -> list[str]:
    """Return blocking issues for the intended one-A100-40GB training shape."""
    issues: list[str] = []
    if plan.max_vram_gb > 40:
        issues.append("max_vram_gb must stay at or below 40 for the A100-40GB target")
    if not plan.freeze_backbone:
        issues.append("freeze the Laguna backbone; full-model training is not A100-40GB safe")
    if not plan.freeze_encoder:
        issues.append("freeze the modality encoder for the first alignment stage")
    if not plan.train_projector:
        issues.append("train_projector must be true; otherwise there is no native adapter to learn")
    if plan.output_tokens > plan.max_output_tokens:
        issues.append(f"adapter output_tokens={plan.output_tokens} exceeds guardrail {plan.max_output_tokens}")
    if plan.micro_batch_size != 1:
        issues.append("use micro_batch_size=1 and gradient accumulation on a single A100-40GB")
    if plan.max_sequence_length > 8192:
        issues.append("start with max_sequence_length<=8192; long-context multimodal RL can come later")
    if plan.lora_enabled:
        issues.append("disable LoRA for the first projector-only stage; add QLoRA after the projector learns")
    if "4" not in plan.backbone_quantization.lower() and "nvfp4" not in plan.backbone_model.lower():
        issues.append("use a 4-bit/NVFP4 backbone for a realistic A100-40GB proof of concept")
    return issues


def render_plan(plan: AdapterPlan) -> str:
    """Human-readable summary for dry runs and PR review."""
    lines = [
        f"Experiment: {plan.name}",
        f"Backbone: {plan.backbone_model} ({plan.backbone_quantization}, frozen={plan.freeze_backbone})",
        f"Modality: {plan.modality} via {plan.encoder_model} as {plan.encoder_role} "
        f"(frozen={plan.freeze_encoder})",
        f"Adapter: {plan.adapter_kind}, objective={plan.objective}, "
        f"{plan.output_tokens} learned tokens, train_projector={plan.train_projector}",
        f"Training: micro_batch={plan.micro_batch_size}, grad_accum={plan.gradient_accumulation_steps}, "
        f"effective_batch={plan.effective_batch_size}",
        f"Guardrail: {plan.max_vram_gb:g}GB VRAM with {plan.reserve_vram_gb:g}GB reserved",
    ]
    if plan.fallback_encoder_model:
        lines.append(f"Fallback encoder: {plan.fallback_encoder_model}")
    issues = validate_a100_40gb(plan)
    if issues:
        lines.append("Blocking issues:")
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("A100-40GB guardrails: pass")
    return "\n".join(lines)
