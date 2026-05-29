from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from laguna_rlvr.visual.encoders import Encoder
from laguna_rlvr.visual.projector import Projector

_PROMPT = "Transcribe the text in the image:"


@dataclass
class Output:
    loss: torch.Tensor


def _assert_backbone_loaded(model: nn.Module, missing_keys: list[str], base_llm: str) -> None:
    """Raise if real pretrained weights were missing from the checkpoint and randomly initialized.

    Non-persistent buffers (e.g. rotary inv_freq) are recomputed at init and legitimately absent;
    anything else missing means the "frozen" backbone is partly noise — e.g. NVFP4-packed MoE
    experts that compressed-tensors can't map onto this modeling revision's fused expert params
    (caught on feat/mm-adapter, 2026-05). Use the model's own non-persistent-buffer registry rather
    than a name denylist so the check generalizes across architectures.
    """
    benign = {
        f"{mod}.{buf}" if mod else buf
        for mod, m in model.named_modules()
        for buf in m._non_persistent_buffers_set
    }
    real = [k for k in missing_keys if k not in benign]
    if real:
        keys = "\n  ".join(real[:8] + (["..."] if len(real) > 8 else []))
        raise RuntimeError(
            f"{base_llm}: {len(real)} backbone weights missing from the checkpoint and randomly "
            f"initialized — the frozen backbone would be partly noise. Use a checkpoint whose layout "
            f"matches this modeling revision (e.g. the unquantized base). Missing:\n  {keys}"
        )


class VisualAdapter(nn.Module):
    """Frozen encoder + trainable projector + frozen causal LLM. Trains the projector only."""

    def __init__(
        self,
        encoder: Encoder,
        base_llm: str,
        projector_kind: str = "linear",
        dtype: torch.dtype | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.encoder = encoder
        self.tok = AutoTokenizer.from_pretrained(base_llm, trust_remote_code=True)
        # Quantized checkpoints (e.g. Laguna NVFP4) carry their own dtype + placement: load via
        # device_map and don't override dtype or call .to(). Plain checkpoints take the explicit path.
        cfg = AutoConfig.from_pretrained(base_llm, trust_remote_code=True)
        load_kwargs = {"trust_remote_code": True, "output_loading_info": True}
        if getattr(cfg, "quantization_config", None) is not None:
            load_kwargs["device_map"] = self.device
        else:
            load_kwargs["dtype"] = dtype
        self.llm, info = AutoModelForCausalLM.from_pretrained(base_llm, **load_kwargs)
        if "dtype" in load_kwargs:  # plain checkpoint: move to device (quantized self-places)
            self.llm = self.llm.to(self.device)
        _assert_backbone_loaded(self.llm, info["missing_keys"], base_llm)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)
        # use_reentrant=False so grads still flow to the (trainable) projector even though
        # every LLM input embed is detached — reentrant checkpointing requires an input
        # that requires grad and would otherwise raise "none of the inputs require grad".
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.projector = Projector(encoder.d_enc, self.llm.config.hidden_size, projector_kind)
        self.projector.to(device=self.llm.device, dtype=self.llm.dtype)

    def trainable_parameters(self):
        return self.projector.parameters()

    def _embed(self, text: str) -> torch.Tensor:
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.llm.device)
        return self.llm.get_input_embeddings()(ids)  # (1, T, D)

    def forward(self, images: list, labels: list[str]) -> Output:
        feats = self.encoder.encode(images).to(device=self.llm.device, dtype=self.llm.dtype)
        vis = self.projector(feats)  # (B, Nv, D)
        losses = []
        for b, label in enumerate(labels):
            prompt_e = self._embed(_PROMPT)  # (1, Tp, D)
            label_ids = self.tok(label, return_tensors="pt").input_ids.to(self.llm.device)
            label_e = self.llm.get_input_embeddings()(label_ids)
            v = vis[b : b + 1]  # (1, Nv, D)
            inputs = torch.cat([v, prompt_e, label_e], dim=1)
            mask = torch.full(
                (1, v.shape[1] + prompt_e.shape[1]), -100, device=self.llm.device
            )
            tgt = torch.cat([mask, label_ids], dim=1)
            losses.append(self.llm(inputs_embeds=inputs, labels=tgt).loss)
        return Output(loss=torch.stack(losses).mean())

    @torch.no_grad()
    def transcribe(self, images: list, max_new_tokens: int = 48) -> list[str]:
        """Greedy-decode the LLM's reading of each image (vision tokens + prompt) for eval."""
        feats = self.encoder.encode(images).to(device=self.llm.device, dtype=self.llm.dtype)
        vis = self.projector(feats)
        out = []
        for b in range(len(images)):
            inputs = torch.cat([vis[b : b + 1], self._embed(_PROMPT)], dim=1)
            gen = self.llm.generate(inputs_embeds=inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out.append(self.tok.decode(gen[0], skip_special_tokens=True))
        return out
