from __future__ import annotations

from typing import Protocol

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from laguna_rlvr.mm.projector import Projector

# Default suits the OCR/vision path (it measurably aids the frozen LLM's overfit); other
# modalities pass their own, e.g. audio -> "Transcribe the speech:".
_PROMPT = "Transcribe the text in the image:"


class EncoderProtocol(Protocol):
    """Any frozen modality encoder (vision tower, Whisper encoder, ...) whose features the
    projector maps into the LLM. Implemented by laguna_rlvr.visual / laguna_rlvr.audio encoders."""

    d_enc: int

    def encode(self, batch: list) -> torch.Tensor: ...


class ModalityAdapter(nn.Module):
    """Frozen encoder + trainable projector + frozen causal LLM. Trains the projector only.

    Modality-agnostic: the encoder may be a vision tower (OCR/doc images) or a Whisper encoder
    (speech). `inputs` are whatever that encoder's `.encode()` accepts (PIL images, waveforms, ...)."""

    def __init__(
        self,
        encoder: EncoderProtocol,
        base_llm: str,
        projector_kind: str = "linear",
        prompt: str = _PROMPT,
        dtype: torch.dtype | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.prompt = prompt  # modality-specific instruction (e.g. "Transcribe the speech:")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.encoder = encoder
        self.tok = AutoTokenizer.from_pretrained(base_llm, trust_remote_code=True)
        # Quantized checkpoints (e.g. Laguna NVFP4) carry their own dtype + placement: load via
        # device_map and don't override dtype or call .to(). Plain checkpoints take the explicit path.
        cfg = AutoConfig.from_pretrained(base_llm, trust_remote_code=True)
        if getattr(cfg, "quantization_config", None) is not None:
            self.llm = AutoModelForCausalLM.from_pretrained(
                base_llm, trust_remote_code=True, device_map=self.device)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                base_llm, trust_remote_code=True, dtype=dtype).to(self.device)
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

    def forward(self, inputs: list, labels: list[str]) -> torch.Tensor:
        feats = self.encoder.encode(inputs).to(device=self.llm.device, dtype=self.llm.dtype)
        tokens = self.projector(feats)  # (B, N, D)
        prompt_e = self._embed(self.prompt)  # (1, Tp, D) — constant across the batch
        losses = []
        for b, label in enumerate(labels):
            label_ids = self.tok(label, return_tensors="pt").input_ids.to(self.llm.device)
            label_e = self.llm.get_input_embeddings()(label_ids)
            v = tokens[b : b + 1]  # (1, N, D)
            seq = torch.cat([v, prompt_e, label_e], dim=1)
            mask = torch.full((1, v.shape[1] + prompt_e.shape[1]), -100, device=self.llm.device)
            tgt = torch.cat([mask, label_ids], dim=1)
            losses.append(self.llm(inputs_embeds=seq, labels=tgt).loss)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def transcribe(self, inputs: list, max_new_tokens: int = 48) -> list[str]:
        """Greedy-decode the LLM's reading of each input (modality tokens + prompt) for eval."""
        feats = self.encoder.encode(inputs).to(device=self.llm.device, dtype=self.llm.dtype)
        tokens = self.projector(feats)
        prompt_e = self._embed(self.prompt)  # constant across the batch
        out = []
        for b in range(len(inputs)):
            seq = torch.cat([tokens[b : b + 1], prompt_e], dim=1)
            gen = self.llm.generate(inputs_embeds=seq, max_new_tokens=max_new_tokens, do_sample=False)
            out.append(self.tok.decode(gen[0], skip_special_tokens=True))
        return out
