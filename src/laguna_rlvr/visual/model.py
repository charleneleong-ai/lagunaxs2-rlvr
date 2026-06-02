from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from laguna_rlvr.visual.corpora import CORPUS_KIND, read_question
from laguna_rlvr.visual.encoders import Encoder
from laguna_rlvr.visual.projector import Projector

IMAGE_TOKEN = "<image>"
_PROMPT = f"{IMAGE_TOKEN}\nTranscribe the text in the image:"
# Cap label length: the full-vocab (100K) LM-head logits scale with sequence length, so long
# WebSight/WebCode2M HTML targets OOM the 33B backbone at the loss (caught 2026-05). 512 tokens is
# ample for projector alignment; the full screenshot->code objective belongs to a later stage.
_MAX_LABEL_TOKENS = 384   # label budget (was 512) — trimmed to keep the loss sequence under the OOM line
_MAX_SEQ_TOKENS = 896     # skip (vision + label) sequences above this — their fp32 logits OOM the 80GB loss


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


def load_causal_lm(base_llm: str, device: str, dtype: torch.dtype) -> tuple[nn.Module, AutoTokenizer]:
    """Load a frozen base causal LM + its tokenizer, quantization-aware (the single load path shared
    by the adapter and the Stage-0 baselines, so the NVFP4 placement logic can't drift).

    Quantized checkpoints (e.g. Laguna NVFP4) carry their own dtype + placement: load via device_map
    and don't override dtype or call .to(). Plain checkpoints take the explicit dtype/.to() path.
    Raises if real pretrained weights were missing (would leave the "frozen" backbone partly noise).
    """
    tok = AutoTokenizer.from_pretrained(base_llm, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(base_llm, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True, "output_loading_info": True}
    if getattr(cfg, "quantization_config", None) is not None:
        load_kwargs["device_map"] = device
    else:
        load_kwargs["dtype"] = dtype
    llm, info = AutoModelForCausalLM.from_pretrained(base_llm, **load_kwargs)
    if "dtype" in load_kwargs:  # plain checkpoint: move to device (quantized self-places)
        llm = llm.to(device)
    _assert_backbone_loaded(llm, info["missing_keys"], base_llm)
    return llm, tok


class VisualAdapter(nn.Module):
    """Frozen encoder + trainable projector + frozen causal LLM. Trains the projector only."""

    def __init__(
        self,
        encoder: Encoder,
        base_llm: str,
        projector_kind: str = "linear",
        dtype: torch.dtype | None = None,
        device: str | None = None,
        unfreeze: str = "",
        use_anchor: bool = True,
        norm_penalty: float = 0.0,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.encoder = encoder
        self.unfreeze = unfreeze
        self.use_anchor = use_anchor  # soft-scalar norm match; off lets the projector keep per-token scale
        self.norm_penalty = norm_penalty  # soft cap on projected-token scale (the --no-anchor ballooning)
        self.llm, self.tok = load_causal_lm(base_llm, self.device, dtype)
        self._add_image_token()  # <image> marker + subtoken-avg init, before the freeze below
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)
        if unfreeze == "lora":  # let the frozen decoder learn to *read* the projector tokens
            self._apply_lora()
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

    def _apply_lora(self) -> None:
        """Wrap the frozen LLM with attention-only LoRA (q/k/v/o). The MoE expert MLPs stay frozen —
        adapting attention is the minimal lever for the decoder to attend to the projected vision
        tokens, and keeps trainable params (+AdamW state) tiny enough to co-reside with the 33B base."""
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.llm = get_peft_model(self.llm, cfg)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]  # projector (+ LoRA if unfrozen)

    def adapter_state_dict(self) -> dict:
        """The trainable deliverable: projector always, plus LoRA deltas when unfrozen. The LoRA
        tensors are saved as raw named_parameters — peft's get/set_peft_model_state_dict route
        through a transformers weight-conversion that is version-fragile (WeightConverter signature
        drift). Only LoRA params require grad (base frozen), so the filter excludes the frozen
        embedding for free — no save_embedding_layers bloat."""
        sd = {"projector": self.projector.state_dict()}
        if self.unfreeze == "lora":
            sd["lora"] = {k: v.detach().cpu() for k, v in self.llm.named_parameters() if v.requires_grad}
        return sd

    def load_adapter_state_dict(self, sd: dict) -> None:
        sd = sd if "projector" in sd else {"projector": sd}  # accept a legacy raw projector state_dict too
        self.projector.load_state_dict(sd["projector"])
        if "lora" in sd:
            lora, own = sd["lora"], dict(self.llm.named_parameters())
            if lora and next(iter(lora)) not in own:  # legacy peft-stripped keys -> re-insert ".default"
                lora = {k.replace(".lora_A.weight", ".lora_A.default.weight")
                         .replace(".lora_B.weight", ".lora_B.default.weight"): v for k, v in lora.items()}
            res = self.llm.load_state_dict(lora, strict=False)
            assert not res.unexpected_keys, f"unmatched LoRA keys: {res.unexpected_keys[:3]}"

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

    def _scale_penalty(self, vis: torch.Tensor) -> torch.Tensor:
        """Soft cap on the projected-token scale — penalize the mean token norm above the embedding
        median. Under --no-anchor the resampler's output scale balloons unbounded (~30x), which tracks
        the qa-accuracy decay (W&B). Scalar loss term; 0 when within base scale or norm_penalty == 0."""
        if self.norm_penalty <= 0:
            return vis.sum() * 0.0
        ratio = vis.flatten(0, 1).float().norm(dim=-1).mean() / self._emb_norm_median
        return self.norm_penalty * (ratio - 1.0).clamp(min=0) ** 2

    def _project(self, images: list, anchor: bool | None = None) -> torch.Tensor:
        # Encode + resample each image individually, then stack the fixed-size (Nv, D) outputs. NaFlex
        # emits a variable patch count per image, so raw encoder features can't be stacked into one batch
        # pre-resampler (crashes on the heterogeneous Stage-2 mix via transcribe/_log_predictions). The
        # resampler folds N -> fixed Nv, so the per-image outputs ARE uniform and stack cleanly. AnyRes
        # (fixed N) works identically; _project runs at micro_batch=1 in training, batched only at eval.
        dev, dt = self.llm.device, self.llm.dtype
        vis = torch.cat([self.projector(self.encoder.encode([img]).to(device=dev, dtype=dt))
                         for img in images], dim=0)  # (B, Nv, D)
        return self._anchor(vis) if (self.use_anchor if anchor is None else anchor) else vis

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
        # one image at a time then concat the per-token norms — variable-resolution images yield
        # different patch counts, which `encoder.encode` can't stack into one batch (see its docstring).
        norms = [self._project([img], anchor=False).flatten(0, 1).float().norm(dim=-1) for img in images]
        return (torch.cat(norms).mean() / self._emb_norm_median).item()

    def forward(self, images: list, labels: list[str]) -> Output:
        vis = self._project(images)  # (B, Nv, D)
        losses = []
        for b, label in enumerate(labels):
            prompt_e = self._embed_with_vision(_PROMPT, vis[b : b + 1])  # vision spliced at <image>
            label_ids = self.tok(label, return_tensors="pt", truncation=True,
                                  max_length=_MAX_LABEL_TOKENS).input_ids.to(self.llm.device)
            # The full-vocab fp32 logits scale with sequence length; a long (vision + label) sequence
            # OOMs the loss on an 80GB A100 (tall page screenshots, caught 2026-05). Skip those rather
            # than crash — they're a small fraction, so the projector still learns on the rest.
            if prompt_e.shape[1] + label_ids.shape[1] > _MAX_SEQ_TOKENS:
                continue
            label_e = self.llm.get_input_embeddings()(label_ids)
            inputs = torch.cat([prompt_e, label_e], dim=1)
            mask = torch.full((1, prompt_e.shape[1]), -100, device=self.llm.device)
            tgt = torch.cat([mask, label_ids], dim=1)
            losses.append(self.llm(inputs_embeds=inputs, labels=tgt).loss)
        if not losses:  # whole micro-batch skipped — 0 loss tied to the projector so backward is a no-op
            return Output(loss=vis.sum() * 0.0)
        return Output(loss=torch.stack(losses).mean() + self._scale_penalty(vis))

    def forward_qa(self, images: list, answers: list[str], corpora: list,
                   questions: list | None = None) -> Output:
        """QA-SFT loss: masked CE on the answer to (image, per-kind question). The answer is a
        label-derived needle (chart/page title) that is NOT in the question, so the loss can only drop
        by USING the image — forcing the projector to convey vision. (Reconstruction lets the frozen
        LLM shortcut via text-LM and ignore the projector; this objective can't be shortcut.)
        """
        vis = self._project(images)
        dev = self.llm.device
        seqs, labels = [], []
        for b, (answer, corpus) in enumerate(zip(answers, corpora)):
            q = (questions[b] if questions and questions[b] else read_question(CORPUS_KIND.get(corpus)))
            prompt_e = self._embed_with_vision(f"{IMAGE_TOKEN}\n{q}\nAnswer:", vis[b : b + 1])
            ans_ids = self.tok(" " + answer, return_tensors="pt", add_special_tokens=False,
                               truncation=True, max_length=_MAX_LABEL_TOKENS).input_ids.to(dev)
            if prompt_e.shape[1] + ans_ids.shape[1] > _MAX_SEQ_TOKENS:
                continue
            ans_e = self.llm.get_input_embeddings()(ans_ids)
            seqs.append(torch.cat([prompt_e, ans_e], dim=1)[0])  # (Li, D)
            labels.append(torch.cat([torch.full((prompt_e.shape[1],), -100, dtype=torch.long, device=dev),
                                     ans_ids[0]]))  # (Li,) prompt masked, answer supervised
        if not seqs:
            return Output(loss=vis.sum() * 0.0)
        return Output(loss=self._batched_lm_loss(seqs, labels, self.llm) + self._scale_penalty(vis))

    @staticmethod
    def _batched_lm_loss(seqs: list[torch.Tensor], labels: list[torch.Tensor], llm) -> torch.Tensor:
        """Right-pad a batch of (inputs_embeds (Li, D), labels (Li,)) and run `llm` once over the
        [B, L] batch, returning the EXAMPLE-weighted mean CE — each example's mean over its own answer
        tokens, then averaged across the batch. This matches the per-example loop it replaces (so loss
        semantics are unchanged) but in a single batched forward, which actually fills the GPU at
        micro_batch>1. Padding gets attention_mask=0 + label -100, so it neither attends nor scores;
        right-padding leaves the real tokens at positions 0..Li-1, so default position ids stay correct."""
        pad = torch.nn.utils.rnn.pad_sequence
        emb = pad(seqs, batch_first=True)                              # (B, L, D), 0-padded
        lab = pad(labels, batch_first=True, padding_value=-100)        # (B, L)
        lengths = torch.tensor([s.shape[0] for s in seqs], device=emb.device)
        attn = (torch.arange(emb.shape[1], device=emb.device)[None] < lengths[:, None]).long()
        logits = llm(inputs_embeds=emb, attention_mask=attn).logits[:, :-1]  # (B, L-1, V)
        tgt = lab[:, 1:]                                                      # next-token targets
        ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                         ignore_index=-100, reduction="none").view(tgt.shape)
        per_example = ce.sum(1) / (tgt != -100).sum(1).clamp(min=1)          # mean over each row's answer
        return per_example.mean()

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
