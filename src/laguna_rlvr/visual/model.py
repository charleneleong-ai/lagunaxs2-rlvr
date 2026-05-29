from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from laguna_rlvr.visual.encoders import Encoder
from laguna_rlvr.visual.projector import Projector

_PROMPT = "Transcribe the text in the image:"


@dataclass
class Output:
    loss: torch.Tensor


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
        self.tok = AutoTokenizer.from_pretrained(base_llm)
        self.llm = AutoModelForCausalLM.from_pretrained(base_llm, dtype=dtype).to(self.device)
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
