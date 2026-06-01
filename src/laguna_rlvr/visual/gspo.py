"""GSPO / GRPO (RLVR) for the visual adapter — optimize a **verifiable reward** directly (the metric
SFT can't: exact-match reading / edit-applied / box-IoU), via group-relative policy optimization.
Defaults to **GSPO** (length-normalized *sequence*-level importance ratio): the base is a 33B MoE, where
GRPO's per-token ratios go unstable (expert routing differs old-vs-new), and our rewards are
sequence-level anyway. GRPO (token-level) kept as a switch (`mode="grpo"`).

Hand-rolled rather than TRL: TRL drives generation + the logprob forward through `input_ids` only, but
our adapter generates from `inputs_embeds` with vision spliced at `<image>` (see the design spec). The
loop is small: per prompt, sample G completions, score them with the reward, group-normalize to
advantages, and take a clipped-ratio policy-gradient step. Only the projector (+ LoRA) is trained; the
base + encoder stay frozen. The reward is a plug-in — the eval scorers (read F1 / edit-applied / IoU)
double as rewards, so the metric bank IS the objective bank.
"""
from __future__ import annotations

import random

import torch
import typer

from laguna_rlvr.visual.corpora import (CORPUS_KIND, DEFAULT_VQA, QASFTDataset, build_corpus,
                                        load_vqa, read_question)
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import IMAGE_TOKEN, VisualAdapter
from laguna_rlvr.visual.multiturn_qa import _match, _norm, dataset_qa_accuracy


def read_reward(needle: str, completion: str) -> float:
    """SHAPED verifiable read reward: 1.0 for a real read (exact/substantial substring), else partial
    credit (token-overlap x 0.5, capped below the exact reward). A binary 0/1 reward is too sparse to
    bootstrap GSPO — from a ~10% reader, sampled groups come out all-zeros, so the group-relative
    advantage is zero and there's no gradient (caught: reward 0.000 at G=4). Partial credit gives reward
    *variance* within the group (warmer completions score higher) -> a gradient toward better reads
    before exact hits, while exact reads (1.0) stay preferred over plausible-but-wrong overlap (<=0.5)."""
    if _match(needle, completion):
        return 1.0
    nt, rt = set(_norm(needle).split()), set(_norm(completion).split())
    return 0.5 * (2 * len(nt & rt) / (len(nt) + len(rt))) if nt and rt else 0.0


REWARDS = {"read": read_reward}  # edit / ground rewards plug in here (edit_eval, grounding.box_iou)


class GSPOTrainer:
    """Group-Relative Policy Optimization over a frozen-base + trainable-projector(+LoRA) adapter."""

    def __init__(self, adapter: VisualAdapter, reward_fn=read_reward, *, group_size: int = 8,
                 lr: float = 1e-6, clip: float = 0.2, max_new_tokens: int = 24, temperature: float = 1.0,
                 mode: str = "gspo"):
        self.a = adapter
        self.reward_fn = reward_fn
        self.G, self.clip, self.max_new_tokens, self.temperature = group_size, clip, max_new_tokens, temperature
        # gspo (default): length-normalized SEQUENCE-level importance ratio — stable on the 33B MoE base
        # (GRPO's per-token ratios explode when expert routing differs old-vs-new) and matches our
        # sequence-level verifiable rewards. grpo: token-level ratio (kept for comparison).
        self.mode = mode
        self.opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)
        self.pad = adapter.tok.pad_token_id or adapter.tok.eos_token_id

    def _prompt_embeds(self, image, question: str) -> torch.Tensor:  # (1, P, D) — vision spliced
        return self.a._embed_with_vision(f"{IMAGE_TOKEN}\n{question}\nAnswer:", self.a._project([image])[0:1])

    @torch.no_grad()
    def _sample_group(self, prompt_e: torch.Tensor):
        """G sampled completions + their generation (old-policy) per-token logprobs."""
        out = self.a.llm.generate(
            inputs_embeds=prompt_e.expand(self.G, -1, -1),
            attention_mask=torch.ones(self.G, prompt_e.shape[1], dtype=torch.long, device=prompt_e.device),
            max_new_tokens=self.max_new_tokens, do_sample=True, temperature=self.temperature, top_p=0.95,
            pad_token_id=self.pad, return_dict_in_generate=True, output_scores=True)
        seqs = out.sequences  # (G, gen_len) — inputs_embeds carry no ids, so this is generated-only
        logp = torch.log_softmax(torch.stack(out.scores, dim=1).float(), dim=-1)  # (G, gen_len, V)
        return seqs, logp.gather(-1, seqs.unsqueeze(-1)).squeeze(-1)  # (G, gen_len)

    def _logprobs(self, prompt_e: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
        """Per-token logprob of `seqs` under the CURRENT policy (with grad through projector/LoRA)."""
        comp_e = self.a.llm.get_input_embeddings()(seqs)  # (G, gen_len, D)
        inp = torch.cat([prompt_e.expand(self.G, -1, -1), comp_e], dim=1)  # (G, P+gen_len, D)
        logits = self.a.llm(inputs_embeds=inp).logits  # (G, P+gen_len, V)
        comp_logits = logits[:, prompt_e.shape[1] - 1:-1, :]  # positions predicting the gen tokens
        return torch.log_softmax(comp_logits.float(), dim=-1).gather(-1, seqs.unsqueeze(-1)).squeeze(-1)

    def step(self, items: list) -> dict[str, float]:
        """`items`: (image, question, needle). Accumulate a GRPO loss over the batch; one optimizer step."""
        self.opt.zero_grad()
        tot_loss = tot_rew = 0.0
        for image, question, needle in items:
            prompt_e = self._prompt_embeds(image, question)
            seqs, old_logp = self._sample_group(prompt_e)  # no grad
            texts = [self.a.tok.decode(s, skip_special_tokens=True) for s in seqs]
            rew = torch.tensor([self.reward_fn(needle, t) for t in texts], device=seqs.device)
            adv = (rew - rew.mean()) / (rew.std() + 1e-4)  # group-relative advantage (G,)
            cur_logp = self._logprobs(prompt_e, seqs)  # (G, gen_len) WITH grad
            mask = (seqs != self.pad).float()
            if self.mode == "gspo":  # length-normalized SEQUENCE ratio, clipped per sequence (MoE-stable)
                length = mask.sum(1).clamp_min(1.0)
                s = torch.exp(((cur_logp - old_logp) * mask).sum(1) / length)  # (G,)
                pg = torch.min(s * adv, s.clamp(1 - self.clip, 1 + self.clip) * adv)
                loss = -pg.mean() / len(items)
            else:  # grpo: per-token importance ratio, clipped per token
                ratio = torch.exp(cur_logp - old_logp)
                pg = torch.min(ratio * adv[:, None], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[:, None])
                loss = -(pg * mask).sum() / mask.sum().clamp_min(1.0) / len(items)
            loss.backward()
            tot_loss += loss.item() * len(items)
            tot_rew += rew.mean().item()
        self.opt.step()
        return {"gspo/loss": tot_loss / len(items), "gspo/reward": tot_rew / len(items)}


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    init_ckpt: str = typer.Option(..., help="warm-start checkpoint (projector + LoRA) from SFT"),
    encoder: str = "siglip", base: str = "poolside/Laguna-XS.2", projector: str = "resampler",
    unfreeze: str = "lora", reward: str = "read", mode: str = "gspo",
    steps: int = 1500, group_size: int = 8, batch: int = 2, lr: float = 1e-6,
    n_train: int = 512, eval_every: int = 100, seed: int = 0, out: str = "results/visual",
    name_suffix: str = "gspo",
) -> None:
    """RLVR fine-tune the warm-started adapter on a verifiable reward via GSPO (the metric IS the loss)."""
    from pathlib import Path

    torch.manual_seed(seed)
    random.seed(seed)
    a = VisualAdapter(load_encoder(encoder, pool=(4 if "qwen" in encoder else 1)), base,
                      projector_kind=projector, use_anchor=False, unfreeze=unfreeze)
    a.load_adapter_state_dict(torch.load(init_ckpt, map_location=a.llm.device))
    print(f"warm-started from {init_ckpt}", flush=True)

    full = QASFTDataset(build_corpus("mix", n_train), vqa_sources=load_vqa(DEFAULT_VQA, n_train))
    items = [(full[i][0], full[i][3] or read_question(CORPUS_KIND.get(full[i][2])), full[i][1])
             for i in range(len(full))]  # (image, question, needle)
    eval_items = [full[i] for i in range(min(40, len(full)))]
    print(f"GSPO over {len(items)} items | reward={reward} mode={mode} G={group_size} batch={batch}", flush=True)

    trainer = GSPOTrainer(a, reward_fn=REWARDS[reward], group_size=group_size, lr=lr, mode=mode)
    out_dir = Path(out) / f"{encoder}__{Path(base).name}__{name_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for step in range(1, steps + 1):
        m = trainer.step(random.sample(items, batch))
        if step % 20 == 0:
            print(f"step {step}/{steps}  loss {m['gspo/loss']:.4f}  reward {m['gspo/reward']:.3f}", flush=True)
        if step % eval_every == 0:
            torch.cuda.empty_cache()
            qa = dataset_qa_accuracy(a, eval_items)["qa/metrics/accuracy"]
            print(f"  [eval] step {step}: qa_acc {qa:.3f}  (best {max(best, qa):.3f})", flush=True)
            if qa > best:
                best = qa
                torch.save(a.adapter_state_dict(), out_dir / "best.pt")
    print(f"done. best qa_acc {best:.3f} -> {out_dir}/best.pt", flush=True)


if __name__ == "__main__":
    app()
