from laguna_rlvr.mm_adapter import plan_from_config, render_plan, validate_gpu_budget

_CONFIG = {
    "name": "test-plan",
    "backbone": {
        "model_id": "poolside/Laguna-XS.2-NVFP4",
        "quantization": "4bit-nvfp4",
        "params_b": 33.4,
        "freeze": True,
        "max_sequence_length": 8192,
    },
    "modality": {
        "kind": "document_image",
        "encoder_id": "zai-org/GLM-OCR",
        "encoder_role": "ocr_context_encoder",
        "fallback_encoder_id": "google/siglip2-base-patch16-naflex",
        "freeze_encoder": True,
    },
    "adapter": {
        "kind": "ocr_context_projector",
        "objective": "optical_context_reconstruction",
        "train_projector": True,
    },
    "training": {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "lora_enabled": False,
    },
    "gpu_guardrails": {"max_vram_gb": 40, "reserve_vram_gb": 4},
}


def test_projector_only_plan_passes_gpu_budget():
    plan = plan_from_config(_CONFIG)
    assert plan.effective_batch_size == 16  # 1 x 16 grad-accum
    assert plan.backbone_vram_gb < 20  # 33.4B @ 4-bit ≈ 16.7GB
    assert validate_gpu_budget(plan) == []


def test_unfrozen_backbone_is_blocked():
    cfg = {**_CONFIG, "backbone": {**_CONFIG["backbone"], "freeze": False}}
    assert any("freeze the backbone" in i for i in validate_gpu_budget(plan_from_config(cfg)))


def test_bf16_backbone_does_not_fit_40gb_budget():
    # 33.4B @ bf16 ≈ 67GB — must be blocked on a 40GB budget (the honest NVFP4->BF16 trap).
    cfg = {**_CONFIG, "backbone": {**_CONFIG["backbone"], "quantization": "bf16"}}
    assert any("exceeds the 40GB budget" in i for i in validate_gpu_budget(plan_from_config(cfg)))


def test_bf16_backbone_fits_80gb_budget():
    cfg = {
        **_CONFIG,
        "backbone": {**_CONFIG["backbone"], "quantization": "bf16"},
        "gpu_guardrails": {"max_vram_gb": 80, "reserve_vram_gb": 6},
    }
    assert validate_gpu_budget(plan_from_config(cfg)) == []


def test_render_plan_reports_guardrail_status():
    rendered = render_plan(plan_from_config(_CONFIG))
    assert "poolside/Laguna-XS.2-NVFP4" in rendered
    assert "zai-org/GLM-OCR" in rendered
    assert "GPU guardrails: pass" in rendered
