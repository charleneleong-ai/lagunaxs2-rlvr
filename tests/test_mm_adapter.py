from laguna_rlvr.mm_adapter import plan_from_config, render_plan, validate_a100_40gb


_CONFIG = {
    "name": "test-plan",
    "backbone": {
        "model_id": "poolside/Laguna-XS.2-NVFP4",
        "quantization": "4bit-nvfp4",
        "freeze": True,
        "max_sequence_length": 8192,
    },
    "modality": {
        "kind": "image",
        "encoder_id": "google/siglip-base-patch16-224",
        "freeze_encoder": True,
    },
    "adapter": {
        "kind": "resampler_projector",
        "output_tokens": 64,
        "train_projector": True,
    },
    "training": {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "lora_enabled": False,
    },
    "a100_40gb_guardrails": {
        "max_vram_gb": 40,
        "reserve_vram_gb": 4,
        "max_output_tokens": 128,
    },
}


def test_projector_only_plan_passes_a100_guardrails():
    plan = plan_from_config(_CONFIG)
    assert plan.effective_batch_size == 16
    assert validate_a100_40gb(plan) == []


def test_unfrozen_backbone_and_too_many_tokens_are_blocked():
    cfg = {
        **_CONFIG,
        "backbone": {**_CONFIG["backbone"], "freeze": False},
        "adapter": {**_CONFIG["adapter"], "output_tokens": 256},
    }
    issues = validate_a100_40gb(plan_from_config(cfg))
    assert any("freeze the Laguna backbone" in issue for issue in issues)
    assert any("output_tokens=256" in issue for issue in issues)


def test_render_plan_explains_guardrail_status():
    rendered = render_plan(plan_from_config(_CONFIG))
    assert "poolside/Laguna-XS.2-NVFP4" in rendered
    assert "A100-40GB guardrails: pass" in rendered
