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

import torch

from laguna_rlvr.visual.model import IMAGE_TOKEN, VisualAdapter
from laguna_rlvr.visual.multiturn_qa import _match


def read_reward(needle: str, completion: str) -> float:
    """Verifiable read reward: 1.0 if the completion contains the answer needle (token-F1 match)."""
    return float(_match(needle, completion))


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
        return {"grpo/loss": tot_loss / len(items), "grpo/reward": tot_rew / len(items)}
