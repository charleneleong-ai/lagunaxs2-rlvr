from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from laguna_rlvr.visual.encoders import Encoder
from laguna_rlvr.visual.projector import Projector

IMAGE_TOKEN = "<image>"
_PROMPT = f"{IMAGE_TOKEN}\nTranscribe the text in the image:"
# Cap label length: the full-vocab (100K) LM-head logits scale with sequence length, so long
# WebSight/WebCode2M HTML targets OOM the 33B backbone at the loss (caught 2026-05). 512 tokens is
# ample for projector alignment; the full screenshot->code objective belongs to a later stage.
_MAX_LABEL_TOKENS = 512


@dataclass
class Output:
    loss: torch.Tensor


@dataclass
class Turn:
    """One user turn of a multimodal conversation: text (with `<image>` markers) + its images."""

    text: str
    images: list = field(default_factory=list)


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
        self._add_image_token()  # <image> marker + subtoken-avg init, before the freeze below
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
        # frozen backbone => its native-embedding norm scale is constant; cache the median once so the
        # base-preservation gauge (embedding_norm_ratio) doesn't re-reduce the full vocab table per call.
        self._emb_norm_median = self.llm.get_input_embeddings().weight.float().norm(dim=-1).median()

    def trainable_parameters(self):
        return self.projector.parameters()

    def _add_image_token(self) -> None:
        """Add an <image> placeholder token, embedding-initialized as the mean of its subtoken
        embeddings — Laguna's new-token recipe (§4.1.1: subtoken-averaged init + frozen warmup).

        We keep it frozen (the projector carries the learning). The token marks where projected
        vision tokens are spliced into the sequence (see `_embed_with_vision`), which is what lets
        vision arrive as a tool observation anywhere in the chat template, not only as a prefix.
        """
        if IMAGE_TOKEN in self.tok.get_vocab():
            self.image_token_id = self.tok.convert_tokens_to_ids(IMAGE_TOKEN)
            return
        sub_ids = self.tok(IMAGE_TOKEN, add_special_tokens=False).input_ids  # before it's atomic
        self.tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        self.image_token_id = self.tok.convert_tokens_to_ids(IMAGE_TOKEN)
        self.llm.resize_token_embeddings(len(self.tok))
        emb = self.llm.get_input_embeddings().weight.data
        emb[self.image_token_id] = emb[sub_ids].mean(dim=0)

    def _embed_multi(self, text: str, vis_list: list[torch.Tensor]) -> torch.Tensor:
        """Embed `text` and splice one vision-token group per `<image>` marker, in order.

        vis_list: list of (1, Nv, D), one per marker. Supports 0 markers (text-only turn) up to N,
        which is what lets vision arrive across multiple conversation turns, not only as a prefix.
        """
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.llm.device)
        embeds = self.llm.get_input_embeddings()(ids)  # (1, T, D)
        pos = (ids[0] == self.image_token_id).nonzero(as_tuple=True)[0].tolist()
        if len(pos) != len(vis_list):
            raise ValueError(f"{len(pos)} {IMAGE_TOKEN} markers but {len(vis_list)} image groups")
        if not pos:
            return embeds
        parts, prev = [], 0
        for p, vis in zip(pos, vis_list):
            parts += [embeds[:, prev:p], vis.to(embeds.dtype)]
            prev = p + 1
        parts.append(embeds[:, prev:])
        return torch.cat(parts, dim=1)

    def _embed_with_vision(self, text: str, vis: torch.Tensor) -> torch.Tensor:
        """Single-image splice — `text` must contain exactly one <image>. See `_embed_multi`."""
        return self._embed_multi(text, [vis])

    def _anchor(self, vis: torch.Tensor) -> torch.Tensor:
        """Soft scalar anchor: scale all vision tokens by one factor so their mean L2 norm matches the
        frozen embedding median. Keeps the spliced sequence in the backbone's input distribution while
        preserving inter-token magnitude ratios (salience) — the one trainable bridge can't quietly
        drift the input scale off base. float() for a stable bf16 reduction; the scalar broadcasts back.
        """
        mean_norm = vis.flatten(0, 1).float().norm(dim=-1).mean().clamp_min(1e-6)
        return vis * (self._emb_norm_median / mean_norm).to(vis.dtype)

    def _project(self, images: list, anchor: bool = True) -> torch.Tensor:
        feats = self.encoder.encode(images).to(device=self.llm.device, dtype=self.llm.dtype)
        vis = self.projector(feats)  # (B, Nv, D)
        return self._anchor(vis) if anchor else vis

    @torch.no_grad()
    def embedding_norm_ratio(self, images: list) -> float | None:
        """Mean L2 norm of the *raw* (pre-anchor) projected vision tokens / median embedding norm.

        Base-preservation gauge measured behind the anchor: ~1.0 = the projector naturally emits
        base-scale tokens (anchor is a no-op); far from 1 = the projector is drifting and the anchor is
        doing the work to keep the frozen LLM's inputs in-distribution. Equals 1/(anchor scale factor).
        None if there are no images.
        """
        if not images:
            return None
        vis = self._project(images, anchor=False)  # raw — what the projector wants, pre-correction
        return (vis.flatten(0, 1).float().norm(dim=-1).mean() / self._emb_norm_median).item()

    def forward(self, images: list, labels: list[str]) -> Output:
        vis = self._project(images)  # (B, Nv, D)
        losses = []
        for b, label in enumerate(labels):
            prompt_e = self._embed_with_vision(_PROMPT, vis[b : b + 1])  # vision spliced at <image>
            label_ids = self.tok(label, return_tensors="pt", truncation=True,
                                  max_length=_MAX_LABEL_TOKENS).input_ids.to(self.llm.device)
            label_e = self.llm.get_input_embeddings()(label_ids)
            inputs = torch.cat([prompt_e, label_e], dim=1)
            mask = torch.full((1, prompt_e.shape[1]), -100, device=self.llm.device)
            tgt = torch.cat([mask, label_ids], dim=1)
            losses.append(self.llm(inputs_embeds=inputs, labels=tgt).loss)
        return Output(loss=torch.stack(losses).mean())

    @torch.no_grad()
    def transcribe(self, images: list, max_new_tokens: int = 48) -> list[str]:
        """Greedy-decode the LLM's reading of each image (vision tokens + prompt) for eval."""
        vis = self._project(images)
        out = []
        for b in range(len(images)):
            inputs = self._embed_with_vision(_PROMPT, vis[b : b + 1])
            gen = self.llm.generate(inputs_embeds=inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out.append(self.tok.decode(gen[0], skip_special_tokens=True))
        return out

    @torch.no_grad()
    def chat(self, turns: list[Turn], max_new_tokens: int = 48) -> list[str]:
        """Multi-turn multimodal QA: generate an assistant reply per turn, conditioned on all prior
        turns + their images. Each turn's `<image>` markers are filled by that turn's images (0..N),
        so vision arrives across turns — the agentic, tool-observation-style use, not a fixed prefix.
        """
        self.llm.gradient_checkpointing_disable()  # generation only — no backward to checkpoint for
        try:
            ctx, replies = None, []
            for turn in turns:
                vis_list = []
                if turn.images:
                    proj = self._project(turn.images)
                    vis_list = [proj[b : b + 1] for b in range(len(turn.images))]
                turn_e = self._embed_multi(turn.text, vis_list)
                ctx = turn_e if ctx is None else torch.cat([ctx, turn_e], dim=1)
                reply_ids = self.llm.generate(
                    inputs_embeds=ctx, max_new_tokens=max_new_tokens, do_sample=False)
                replies.append(self.tok.decode(reply_ids[0], skip_special_tokens=True))
                ctx = torch.cat([ctx, self.llm.get_input_embeddings()(reply_ids)], dim=1)
            return replies
        finally:
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
