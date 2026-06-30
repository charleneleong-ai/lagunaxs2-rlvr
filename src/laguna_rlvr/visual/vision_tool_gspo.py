"""Tool-loop GSPO — RLVR on the VisionToolEnv *episode* reward (the agentic encoder+decoder+tool loop).

`gspo.py` optimizes single-shot `read_reward` on `{IMAGE}\nQ\nAnswer:` — it never calls a tool, so it
can't climb the agentic headroom or fix the ocr-call trust ratio (the warm-started adapter under-calls
ocr, 40 vs 69 trained). This trains the TOOL-LOOP outcome instead: per item, sample G stochastic
VisionToolEnv episodes (ocr?/answer across turns), reward each by whether it solved, group-normalize to
advantages, and take a GSPO sequence-ratio step over the MODEL-generated tokens only (the spliced image
and injected ocr observations are context, masked out). Calling ocr when it helps is the trajectory that
solves -> higher advantage -> the policy learns *when* to read, which is exactly the trust-ratio fix.

Rollout reuses VisionToolEnv's `episode_prompt` / `_interpret` / `ocr_observation` verbatim, so the RL
distribution is the same loop `vision_tool_eval probe` measures. GSPO (sequence-level ratio) only — the
reward is per-episode (sequence-level), which is what GSPO is for; per-token GRPO is left to gspo.py.

    uv run python -m laguna_rlvr.visual.vision_tool_gspo \
        --init-ckpt results/visual/glm_ocr__tool_sft/best.pt --steps 800
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import typer
import wandb

from laguna_rlvr.visual.model import VisualAdapter
from laguna_rlvr.visual.multiturn_qa import _match
from laguna_rlvr.visual.ocr_backend_eval import _glyph_corpora
from laguna_rlvr.visual.tool_eval import load_items
from laguna_rlvr.visual.vision_tool_eval import (_NO_OCR, _glyph_transcripts, _interpret, _load_adapter,
                                                 episode_prompt, run_episode)

Turn = tuple[torch.Tensor, str | None]  # (generated token ids, observation injected after — None ends it)
_OCR_OBS = "[ocr of "  # ocr_observation()'s prefix — marks turns where the policy actually called ocr


def episode_reward(solved: bool, turns: list[Turn], gold: str, final_reply: str, transcript: str,
                   *, max_turns: int, tool_bonus: float = 0.1) -> float:
    """Per-episode reward, dominated by the DISCRETE solve so optimizing it moves the greedy reader (the
    earlier char-similarity proxy let reward climb without ever crossing the solve boundary -> greedy eval
    flat at the 0.125 floor while sampled reward rose). The two shaping terms are each CAUSALLY upstream of
    solving, so they don't pull off-objective; they exist to give within-group spread on unanimous-miss /
    unanimous-solve groups (zero spread -> zero GSPO advantage -> no gradient):
      tool   — +`tool_bonus` when the ocr-call decision matched need (call ocr iff transcript carries gold):
               correct tool use is a PREREQUISITE for solving the read, and this is the trust-ratio fix. It's
               also the only pro-ocr signal on a transcription item that never solves — raising it (or
               `_advantages(corpus_norm=...)`) is the ocrvqa-collapse fix; see `_advantages`.
      effic  — +<=0.05 for solving in fewer turns: breaks ties inside the all-solve group
    Scale is irrelevant to GSPO (advantage normalizes); the terms only differentiate near-tied rollouts."""
    answer = 1.0 if solved else 0.0
    called_ocr = any(obs is not None and obs.startswith(_OCR_OBS) for _, obs in turns)
    tool = tool_bonus if called_ocr == _match(gold, transcript) else 0.0
    effic = 0.05 * (1 - (len(turns) - 1) / max_turns) if solved else 0.0
    return answer + tool + effic


def _advantages(rew_groups: list[torch.Tensor], corpora: list[str], *, corpus_norm: bool,
                baselines: dict[str, float] | None = None, std_floor: float = 1e-4) -> list[torch.Tensor]:
    """Group-relative GSPO advantages, scaled by the (stable) global batch std either way.

    The centre is the knob. Default (cross-batch) subtracts the global batch mean — but then on a
    transcription item that never solves, both its ocr-call (+tool) and direct-answer (0) rollouts sit far
    BELOW a mean inflated by solves elsewhere, so both get negative advantage and the only positive
    gradient in the batch comes from the direct-answer solves on non-transcription corpora -> the policy
    drops ocr globally. `corpus_norm` instead centres each item on a RUNNING per-corpus baseline (`baselines`,
    an EMA the trainer carries across steps), so an unsolved ocr-call (reward = tool_bonus) that beats its
    corpus's typical reward gets positive advantage — restoring the pro-ocr gradient where the read needs it
    without touching the good drift on other corpora. The baseline is a cross-step EMA, NOT the within-batch
    corpus mean, so a batch holding one item per corpus doesn't collapse corpus_norm back to per-item
    normalization (which would re-starve unanimous groups — the very thing cross-batch centring fixed)."""
    flat = torch.cat(rew_groups)
    std = flat.std().clamp_min(std_floor)
    mean = flat.mean()
    if not corpus_norm:
        return [(g - mean) / std for g in rew_groups]
    base, gmean = baselines or {}, float(mean)  # gmean: one GPU->CPU sync, cold-start for an unseen corpus
    return [(g - base.get(c, gmean)) / std for g, c in zip(rew_groups, corpora)]


def _kl_to_ref(cur_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimator (Schulman) of KL(current || reference): unbiased, non-negative, low-variance.
    Penalizing it leashes the policy to the SFT reference, the textbook fix for the degenerate mode collapse
    pure-clip GSPO drifts into (ocrvqa -> a memorized constant). The leash is uniform, so it also damps the
    *beneficial* drift on reasoning corpora — keep the coefficient small and A/B it against the targeted knobs."""
    delta = ref_logp - cur_logp
    return torch.exp(delta) - delta - 1.0


def _gen_positions(prompt_len: int, segments: list[tuple[int, int]]) -> list[int]:
    """Embedding positions of the generated tokens in a rebuilt trajectory. `segments` is (gen_len,
    obs_len) per turn (obs_len 0 when the turn ended). Pulled out pure — this off-by-one-prone index map
    decides which tokens get credited, so it's unit-tested without loading the model."""
    pos, gen = prompt_len, []
    for gen_len, obs_len in segments:
        gen += range(pos, pos + gen_len)
        pos += gen_len + obs_len
    return gen


def load_tool_items(n: int) -> list[tuple]:
    """(image, image_id, question, transcript, gold) per glyph item — the same corpora + cached Qwen3-VL
    transcripts VisionToolEnv probes, so RL trains on the exact eval distribution."""
    transcripts = _glyph_transcripts(n)
    seen: dict[str, int] = {}
    items = []
    for corpus, img, q, gold in load_items(_glyph_corpora(), n):
        i = seen.get(corpus, 0)
        seen[corpus] = i + 1
        tr = transcripts.get((corpus, i), _NO_OCR)
        items.append((img, f"{corpus}.png", q, tr, str(gold)))
    return items


class ToolLoopGSPO:
    """GSPO over VisionToolEnv episodes: sample G tool-loop trajectories per item, score each with the
    solve-dominated `episode_reward` (small tool/efficiency terms give within-group spread so groups aren't
    unanimous), and take a clipped sequence-ratio step over the raw-rescored old policy (ratio == 1 at the
    on-policy step). Only the projector (+ LoRA) trains; base + encoder stay frozen."""

    def __init__(self, adapter: VisualAdapter, *, group_size: int = 6, lr: float = 3e-6, clip: float = 0.2,
                 fmt: str = "poolside", max_turns: int = 4, max_new_tokens: int = 24, temperature: float = 0.8,
                 max_grad_norm: float = 1.0, max_logratio: float = 10.0, tool_bonus: float = 0.1,
                 corpus_norm: bool = False, kl_coef: float = 0.0):
        self.a = adapter
        self.G, self.clip, self.fmt = group_size, clip, fmt
        self.max_turns, self.max_new_tokens, self.temperature = max_turns, max_new_tokens, temperature
        self.max_grad_norm, self.max_logratio = max_grad_norm, max_logratio
        self.tool_bonus, self.corpus_norm, self.kl_coef = tool_bonus, corpus_norm, kl_coef
        self.corpus_baseline: dict[str, float] = {}  # running EMA of per-corpus mean reward (corpus_norm)
        # KL-to-SFT reference = the warm-start adapter snapshot (the trainer is built right after init_ckpt
        # loads, before any opt step). Shared frozen base, so the reference is just these trainable deltas.
        self.ref_state = self._snapshot() if kl_coef > 0 else None
        self.opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)
        self.pad = adapter.tok.pad_token_id or adapter.tok.eos_token_id
        self.last_rollouts: list = []  # (image, question, replies, rewards) for W&B sample logging
        self.last_solved: list[float] = []  # per-item solve fraction this step — feeds the difficulty sampler

    def _snapshot(self) -> dict:
        """A fully-cloned trainable-param snapshot. adapter_state_dict() cpu-clones the LLM deltas but
        projector.state_dict() returns LIVE references — and load_adapter_state_dict copies in-place, so an
        un-cloned projector snapshot would corrupt on the first swap. Clone the projector explicitly."""
        sd = self.a.adapter_state_dict()
        sd["projector"] = {k: v.detach().clone() for k, v in sd["projector"].items()}
        return sd

    def _ref_logps(self, items: list, all_eps: list) -> list[list[torch.Tensor]]:
        """Per-rollout reference logprobs: swap the SFT snapshot in, score every trajectory under no-grad,
        swap the current policy back. The base is frozen and shared, so only the trainable deltas move."""
        cur = self._snapshot()
        self.a.load_adapter_state_dict(self.ref_state)
        try:
            with torch.no_grad():
                return [[self._cur_logp(im, iid, q, ts) for ts, *_ in eps]
                        for (im, iid, q, _, _), eps in zip(items, all_eps, strict=True)]
        finally:
            self.a.load_adapter_state_dict(cur)  # restore even if a forward raises

    def _trim(self, seq: torch.Tensor) -> torch.Tensor:
        """A batched generate right-pads short samples past their first eos; keep through the first eos
        (inclusive). pad may == eos, so match the first pad occurrence — everything after it is padding."""
        hit = (seq == self.pad).nonzero(as_tuple=True)[0]
        end = hit[0].item() + 1 if hit.numel() else seq.shape[0]
        return seq[:end]

    @torch.no_grad()
    def _sample(self, ctx: torch.Tensor, n: int) -> list[torch.Tensor]:
        """n sampled token sequences (generated-only, trimmed) from a shared context. n>1 runs as ONE
        batched generate (num_return_sequences) — the GPU win on the G-wide turn 0. Logprobs are NOT read
        from generation: `out.scores` are temperature/top_p-WARPED (top_p sets non-nucleus logits to -inf),
        so they'd never match `_cur_logp`'s raw re-score -> ratio != 1 even on-policy. The step recomputes
        old logprobs from raw logits instead (see `_rollout_group`)."""
        seqs = self.a.llm.generate(inputs_embeds=ctx, num_return_sequences=n, max_new_tokens=self.max_new_tokens,
                                   do_sample=True, temperature=self.temperature, top_p=0.95, pad_token_id=self.pad)
        return [self._trim(seqs[i]) for i in range(n)]

    def _step_turn(self, seq: torch.Tensor, ctx: torch.Tensor, turns: list[Turn], gold: str,
                   transcript: str) -> tuple[torch.Tensor, str, bool, bool]:
        """Record one generated turn, interpret it, and extend ctx with the reply and — if it called ocr —
        the injected observation. Returns (ctx, reply, done, solved)."""
        reply = self.a.tok.decode(seq, skip_special_tokens=True).strip()
        ctx = torch.cat([ctx, self.a.llm.get_input_embeddings()(seq[None])], dim=1)
        kind, payload = _interpret(reply, self.fmt, gold, transcript)
        if kind == "done":
            turns.append((seq, None))
            return ctx, reply, True, bool(payload)
        turns.append((seq, payload))
        return torch.cat([ctx, self.a._embed_multi(f"\n{payload}\n", [])], dim=1), reply, False, False

    @torch.no_grad()
    def _rollout_group(self, image, image_id: str, question: str, transcript: str,
                       gold: str) -> list[tuple[list[Turn], torch.Tensor, float, str, bool]]:
        """G sampled episodes for one item. Turn 0 shares the prompt across the group, so it's ONE batched
        generate (num_return_sequences=G) — the bulk of the work, where batch=1 left the GPU at ~27%. Turns
        1+ diverge (only rollouts that called ocr continue, on their own observation), so they fall back to
        per-sequence generation. Old-policy logprobs are the RAW re-score of the assembled trajectory (same
        path as `_cur_logp`, detached here), so the on-policy ratio is exactly 1 at the step."""
        vis = self.a._project([image])[0:1]
        base = self.a._embed_multi(episode_prompt(image_id, question, self.fmt), [vis])
        episodes = []
        for seq in self._sample(base, self.G):
            turns: list[Turn] = []
            ctx, reply, done, solved = self._step_turn(seq, base, turns, gold, transcript)
            for _ in range(self.max_turns - 1):
                if done:
                    break
                seq, = self._sample(ctx, 1)
                ctx, reply, done, solved = self._step_turn(seq, ctx, turns, gold, transcript)
            old = self._cur_logp(image, image_id, question, turns)  # raw old-policy logprob (no grad here)
            reward = episode_reward(solved, turns, gold, reply, transcript, max_turns=self.max_turns,
                                    tool_bonus=self.tool_bonus)
            episodes.append((turns, old, reward, reply, solved))
        return episodes

    def _cur_logp(self, image, image_id: str, question: str, turns: list[Turn]) -> torch.Tensor:
        """Per-token logprob of every generated token under the CURRENT policy (grad through projector +
        LoRA). Rebuilds the trajectory deterministically — prompt (vision spliced, with grad), generated
        spans, injected observations — matching the rollout exactly, so old vs current are comparable."""
        vis = self.a._project([image])[0:1]  # grad flows here
        parts = [self.a._embed_multi(episode_prompt(image_id, question, self.fmt), [vis])]
        emb = self.a.llm.get_input_embeddings()
        seg, gen_ids = [], []
        for seq, obs in turns:
            parts.append(emb(seq[None]))
            gen_ids.append(seq)
            if obs is None:
                seg.append((seq.shape[0], 0))
                continue
            o = self.a._embed_multi(f"\n{obs}\n", [])
            seg.append((seq.shape[0], o.shape[1]))
            parts.append(o)
        logits = self.a.llm(inputs_embeds=torch.cat(parts, dim=1)).logits[0]  # (L, V)
        gp = torch.tensor(_gen_positions(parts[0].shape[1], seg), device=logits.device) - 1  # p-1 predicts p
        lp = torch.log_softmax(logits[gp].float(), dim=-1)
        return lp.gather(-1, torch.cat(gen_ids)[:, None]).squeeze(-1)  # (T_gen,)

    def step(self, items: list) -> dict[str, float]:
        self.opt.zero_grad()
        self.last_rollouts, self.last_solved = [], []

        # Pass 1: collect all rollouts (no grad; _rollout_group is @torch.no_grad()).
        self.a.llm.gradient_checkpointing_disable()
        all_eps = [self._rollout_group(img, iid, q, tr, gold) for img, iid, q, tr, gold in items]

        # Advantage normalization pools all batch*G rewards (see _advantages): cross-batch centring keeps
        # signal when each per-item G-group is unanimous but the batch mixes easy/hard items; corpus_norm
        # centres per-corpus so an unsolved ocr-call stays positive within its corpus (the ocrvqa fix).
        dev = self.a.llm.device
        rew_groups = [torch.tensor([r for _, _, r, _, _ in eps], device=dev) for eps in all_eps]
        corpora = [iid.rsplit(".", 1)[0] for _, iid, _, _, _ in items]
        adv_groups = _advantages(rew_groups, corpora, corpus_norm=self.corpus_norm,
                                 baselines=self.corpus_baseline)  # centre on the PRIOR EMA, then update below
        if self.corpus_norm:
            for c, g in zip(corpora, rew_groups):
                gm = float(g.mean())  # one sync/group; seeds the EMA cold-start and the update
                self.corpus_baseline[c] = 0.9 * self.corpus_baseline.get(c, gm) + 0.1 * gm

        # Reference logprobs for the optional KL-to-SFT leash — scored before pass 2 flips on checkpointing.
        # None when the leash is off; [None]*G per item otherwise so the inner zip yields a ref per rollout.
        ref_logps = self._ref_logps(items, all_eps) if self.kl_coef > 0 else [None] * len(items)

        # Pass 2: backward with grad; checkpointing remains enabled through every _cur_logp forward.
        tot_loss = tot_rew = tot_solved = tot_kl = 0.0
        self.a.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for (image, image_id, question, transcript, gold), eps, rew, adv, refs in zip(
                items, all_eps, rew_groups, adv_groups, ref_logps, strict=True):
            turns_list, old_logps, _, replies, solved_flags = zip(*eps)
            self.last_rollouts.append((image, question, list(replies),
                                       [round(x, 3) for x in rew.tolist()]))
            refs = refs if refs is not None else [None] * len(turns_list)
            logr, kls = [], []
            for ts, old, ref in zip(turns_list, old_logps, refs, strict=True):
                cur = self._cur_logp(image, image_id, question, ts)
                logr.append(((cur - old).sum() / max(cur.shape[0], 1)).clamp(-self.max_logratio, self.max_logratio))
                if ref is not None:
                    kls.append(_kl_to_ref(cur, ref).mean())
            s = torch.exp(torch.stack(logr))  # (G,) length-normalized sequence ratio
            pg = torch.min(s * adv, s.clamp(1 - self.clip, 1 + self.clip) * adv)
            loss = -pg.mean() / len(items)
            if kls:  # KL leash off -> no ref tensor, no sync (keeps the other variants' hot path clean)
                kl = torch.stack(kls).mean()
                loss = loss + self.kl_coef * kl / len(items)
                tot_kl += float(kl)
            if torch.isfinite(loss):
                loss.backward()
                tot_loss += loss.item() * len(items)
            tot_rew += rew.mean().item()
            item_solved = sum(solved_flags) / self.G  # true solve-rate, not inferred from the shaped reward
            self.last_solved.append(item_solved)
            tot_solved += item_solved
        self.a.llm.gradient_checkpointing_disable()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.a.trainable_parameters(), self.max_grad_norm)
        if torch.isfinite(grad_norm):
            self.opt.step()  # else drop the batch — a non-finite grad would write NaN into the weights
        return {"tlg/loss": tot_loss / len(items), "tlg/reward": tot_rew / len(items),
                "tlg/solved": tot_solved / len(items), "tlg/grad_norm": float(grad_norm),
                "tlg/kl": tot_kl / len(items)}


class DifficultySampler:
    """Weighted item sampler that biases batches toward the learnable boundary band — items whose sampled
    solve-rate sits near 0.5, where the G rollouts split (mixed solve/miss => nonzero within-group spread =>
    nonzero GSPO advantage). Items that always solve or always miss give zero spread and zero gradient, so
    GSPO wastes those batches; under uniform sampling at solve-rate ~0.18 almost every batch was saturated
    (grad_norm 0.00). The weight is the Bernoulli variance p*(1-p) of the per-item solve EMA — exactly the
    scale of the squared group-relative advantage that item contributes — plus a floor so no item is ever
    zeroed (it stays explorable, and its estimate stays fresh as the policy drifts). The estimate is online
    over the rollouts the trainer already runs (no separate probe pass); unseen items start optimistic
    (p=0.5 => max weight), and the first observation replaces that prior outright so a truly always-miss item
    drops out of the boundary band immediately instead of lingering at the blended 0.35."""

    def __init__(self, n: int, *, floor: float = 0.05, alpha: float = 0.3):
        self.p = [0.5] * n  # solve-rate EMA, optimistic init -> every item drawn early
        self.seen = [False] * n
        self.floor, self.alpha = floor, alpha

    def weights(self) -> torch.Tensor:
        return torch.tensor([self.floor + p * (1 - p) for p in self.p])

    def sample(self, batch: int) -> list[int]:
        return torch.multinomial(self.weights(), min(batch, len(self.p)), replacement=False).tolist()

    def update(self, idx: int, solve_frac: float) -> None:
        a = self.alpha if self.seen[idx] else 1.0  # first obs replaces the optimistic prior outright
        self.p[idx] = (1 - a) * self.p[idx] + a * solve_frac
        self.seen[idx] = True


@torch.no_grad()
def _eval_solved(adapter: VisualAdapter, items: list, *, fmt: str, max_turns: int, temperature: float,
                 k: int = 4) -> tuple[float, float]:
    """Held-out VisionToolEnv solve-rate, returned as (sampled, greedy).

    `sampled` is the mean solve fraction over `k` draws/item under the *training* decode (do_sample, no
    repetition_penalty) — the expected-solve quantity GSPO actually maximizes, so it's the primary metric.
    `greedy` is the bake-off's deterministic + repetition_penalty=1.3 decode — a secondary deploy number,
    a different distribution the gradient doesn't directly optimize (it's why greedy sat at the warm-start
    floor while sampled solve climbed)."""
    adapter.llm.gradient_checkpointing_disable()
    n = max(len(items), 1)

    def solve_rate(draws: int, **decode) -> float:
        return sum(run_episode(adapter, img, iid, q, tr, gold, fmt=fmt, max_turns=max_turns, **decode)[0]
                   for img, iid, q, tr, gold in items for _ in range(draws)) / (n * draws)

    greedy = solve_rate(1)  # default decode: deterministic + repetition_penalty=1.3
    sampled = solve_rate(k, do_sample=True, temperature=temperature, top_p=0.95, repetition_penalty=1.0)
    return sampled, greedy


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    init_ckpt: str = typer.Option(..., help="warm-started adapter (projector + LoRA) from vision_tool_sft"),
    base: str = typer.Option("poolside/Laguna-XS.2"),
    fmt: str = typer.Option("poolside", help="tool-call scaffold the loop runs in"),
    steps: int = typer.Option(800), group_size: int = typer.Option(8), batch: int = typer.Option(2),
    lr: float = typer.Option(3e-6, help="3e-6 — the prior 1e-6 moved the greedy mode too little over 400 "
                             "steps; the aligned discrete-solve reward tolerates a larger step"),
    max_turns: int = typer.Option(4),
    temperature: float = typer.Option(0.8, help="rollout sampling temperature — high enough that the G "
                                      "samples per item DIFFER (mixed solve/miss within a group = nonzero "
                                      "GSPO advantage), low enough to stay near the greedy reader; 1.0 drifts "
                                      "too far off-policy -> all-zero rewards -> zero gradient"),
    n_train: int = typer.Option(40, help="items per glyph corpus (40 reuses the cached Qwen3-VL transcripts)"),
    eval_n: int = typer.Option(40, help="held-out items for the greedy solve-rate eval"),
    eval_every: int = typer.Option(50), seed: int = typer.Option(0),
    difficulty_sampling: bool = typer.Option(True, "--difficulty/--uniform",
                                             help="bias batches toward the learnable boundary band (items "
                                             "whose solve-rate splits the G group) instead of uniform — the "
                                             "fix for saturated all-solve/all-miss batches starving the grad"),
    sampler_floor: float = typer.Option(0.05, help="DifficultySampler weight floor — raise it so a corpus "
                                        "whose solve-EMA collapses (e.g. ocrvqa) keeps getting sampled "
                                        "instead of being abandoned out of the boundary band"),
    tool_bonus: float = typer.Option(0.1, help="reward for the correct ocr-call decision; raise to give a "
                                     "standalone pro-ocr gradient on transcription items that never solve"),
    corpus_norm: bool = typer.Option(False, "--corpus-norm/--cross-batch-norm",
                                     help="centre advantage per-corpus (not global batch mean) so an unsolved "
                                     "ocr-call stays positive within its corpus — the targeted ocrvqa fix"),
    kl_coef: float = typer.Option(0.0, help="KL-to-SFT leash coefficient (0 = off). Prevents the degenerate "
                                  "mode collapse, but uniformly — too large also undoes the good drift. "
                                  "Doubles the per-step forward cost (a reference pass per rollout)"),
    out: str = typer.Option("results/visual"), name_suffix: str = typer.Option("tool_gspo"),
    wandb_tracking: bool = typer.Option(True, "--wandb/--no-wandb"),
) -> None:
    """RLVR the warm-started adapter on the VisionToolEnv episode reward via tool-loop GSPO."""
    torch.manual_seed(seed)
    random.seed(seed)
    adapter = _load_adapter(init_ckpt, base)  # exact glm_ocr config the warm-start was trained with
    adapter.train()
    print(f"warm-started from {init_ckpt}", flush=True)

    all_items = load_tool_items(n_train)
    held = set(range(0, len(all_items), max(len(all_items) // eval_n, 1)))  # strided -> spans all corpora
    eval_items = [all_items[i] for i in sorted(held)][:eval_n]
    items = [it for i, it in enumerate(all_items) if i not in held]  # disjoint train split
    print(f"tool-loop GSPO over {len(items)} train / {len(eval_items)} held-out items | "
          f"G={group_size} batch={batch} fmt={fmt} | "
          f"sampling={'difficulty' if difficulty_sampling else 'uniform'} floor={sampler_floor} | "
          f"tool_bonus={tool_bonus} norm={'corpus' if corpus_norm else 'cross-batch'} kl={kl_coef}", flush=True)

    run_name = f"glm_ocr__{Path(base).name}__{name_suffix}"
    run = wandb.init(project="laguna-mm-adapter", name=run_name,
                     config={"group_size": group_size, "batch": batch, "lr": lr, "steps": steps,
                             "temperature": temperature, "max_turns": max_turns, "init_ckpt": init_ckpt,
                             "reward": "episode_reward", "tool_bonus": tool_bonus, "corpus_norm": corpus_norm,
                             "sampler_floor": sampler_floor, "kl_coef": kl_coef}) if wandb_tracking else None
    trainer = ToolLoopGSPO(adapter, group_size=group_size, lr=lr, fmt=fmt, max_turns=max_turns,
                           temperature=temperature, tool_bonus=tool_bonus, corpus_norm=corpus_norm,
                           kl_coef=kl_coef)
    out_dir = Path(out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = DifficultySampler(len(items), floor=sampler_floor) if difficulty_sampling else None
    best = -1.0
    for step in range(1, steps + 1):
        idxs = sampler.sample(batch) if sampler else random.sample(range(len(items)), batch)
        m = trainer.step([items[i] for i in idxs])
        if sampler:
            for i, sr in zip(idxs, trainer.last_solved, strict=True):  # loud if order/len ever diverge
                sampler.update(i, sr)
        if run is not None:
            run.log(m, step=step)
        if step % 10 == 0:
            print(f"step {step}/{steps}  loss {m['tlg/loss']:.4f}  reward {m['tlg/reward']:.3f}  "
                  f"solved {m['tlg/solved']:.3f}  grad_norm {m['tlg/grad_norm']:.2f}", flush=True)
        if step % eval_every == 0:
            torch.cuda.empty_cache()
            acc, greedy = _eval_solved(adapter, eval_items, fmt=fmt, max_turns=max_turns, temperature=temperature)
            best = max(best, acc)  # checkpoint on the sampled solve-rate — the distribution GSPO optimizes
            print(f"  [eval] step {step}: sampled {acc:.3f}  greedy {greedy:.3f}  (best {best:.3f})", flush=True)
            if run is not None:
                run.log({"vision_tool/solve_rate": acc, "vision_tool/greedy": greedy,
                         "vision_tool/best": best}, step=step)
                if trainer.last_rollouts:
                    tbl = wandb.Table(columns=["image", "question", "rollout", "reward"])
                    for image, question, replies, rews in trainer.last_rollouts:
                        for t, rw in zip(replies, rews):
                            tbl.add_data(wandb.Image(image), question[:200], t[:200], rw)
                    run.log({"vision_tool/rollouts": tbl}, step=step)
            if acc >= best:
                torch.save(adapter.adapter_state_dict(), out_dir / "best.pt")
    print(f"done. best sampled solve_rate {best:.3f} -> {out_dir}/best.pt", flush=True)
    if run is not None:
        run.summary.update({"best_solve_rate": best})
        run.finish()


if __name__ == "__main__":
    app()
