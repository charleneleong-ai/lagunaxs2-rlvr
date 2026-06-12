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
                   *, max_turns: int) -> float:
    """Per-episode reward, dominated by the DISCRETE solve so optimizing it moves the greedy reader (the
    earlier char-similarity proxy let reward climb without ever crossing the solve boundary -> greedy eval
    flat at the 0.125 floor while sampled reward rose). The two shaping terms are each CAUSALLY upstream of
    solving, so they don't pull off-objective; they exist to give within-group spread on unanimous-miss /
    unanimous-solve groups (zero spread -> zero GSPO advantage -> no gradient):
      tool   — +0.1 when the ocr-call decision matched need (call ocr iff the transcript carries gold):
               correct tool use is a PREREQUISITE for solving the read, and this is the trust-ratio fix
      effic  — +<=0.05 for solving in fewer turns: breaks ties inside the all-solve group
    Scale is irrelevant to GSPO (advantage normalizes); the terms only differentiate near-tied rollouts."""
    answer = 1.0 if solved else 0.0
    called_ocr = any(obs is not None and obs.startswith(_OCR_OBS) for _, obs in turns)
    tool = 0.1 if called_ocr == _match(gold, transcript) else 0.0
    effic = 0.05 * (1 - (len(turns) - 1) / max_turns) if solved else 0.0
    return answer + tool + effic


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
                 max_grad_norm: float = 1.0, max_logratio: float = 10.0):
        self.a = adapter
        self.G, self.clip, self.fmt = group_size, clip, fmt
        self.max_turns, self.max_new_tokens, self.temperature = max_turns, max_new_tokens, temperature
        self.max_grad_norm, self.max_logratio = max_grad_norm, max_logratio
        self.opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr)
        self.pad = adapter.tok.pad_token_id or adapter.tok.eos_token_id
        self.last_rollouts: list = []  # (image, question, replies, rewards) for W&B sample logging

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
            reward = episode_reward(solved, turns, gold, reply, transcript, max_turns=self.max_turns)
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
        tot_loss = tot_rew = tot_solved = 0.0
        self.last_rollouts = []
        self.a.llm.gradient_checkpointing_disable()  # rollouts are generation-only (no grad to checkpoint)
        for image, image_id, question, transcript, gold in items:
            eps = self._rollout_group(image, image_id, question, transcript, gold)
            turns, old_logps, rewards, replies, solved = zip(*eps)
            rew = torch.tensor(rewards, device=self.a.llm.device)
            self.last_rollouts.append((image, question, list(replies),
                                       [round(x, 3) for x in rew.tolist()]))
            adv = (rew - rew.mean()) / (rew.std() + 1e-4)  # group-relative advantage (G,)
            self.a.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logr = []
            for ts, old in zip(turns, old_logps):
                cur = self._cur_logp(image, image_id, question, ts)
                logr.append(((cur - old).sum() / max(cur.shape[0], 1)).clamp(-self.max_logratio, self.max_logratio))
            self.a.llm.gradient_checkpointing_disable()
            s = torch.exp(torch.stack(logr))  # (G,) length-normalized sequence ratio
            pg = torch.min(s * adv, s.clamp(1 - self.clip, 1 + self.clip) * adv)
            loss = -pg.mean() / len(items)
            if torch.isfinite(loss):
                loss.backward()
                tot_loss += loss.item() * len(items)
            tot_rew += rew.mean().item()
            tot_solved += sum(solved) / self.G  # true solve-rate, not inferred from the shaped reward
        grad_norm = torch.nn.utils.clip_grad_norm_(self.a.trainable_parameters(), self.max_grad_norm)
        if torch.isfinite(grad_norm):
            self.opt.step()  # else drop the batch — a non-finite grad would write NaN into the weights
        return {"tlg/loss": tot_loss / len(items), "tlg/reward": tot_rew / len(items),
                "tlg/solved": tot_solved / len(items), "tlg/grad_norm": float(grad_norm)}


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
          f"G={group_size} batch={batch} fmt={fmt}", flush=True)

    run_name = f"glm_ocr__{Path(base).name}__{name_suffix}"
    run = wandb.init(project="laguna-mm-adapter", name=run_name,
                     config={"group_size": group_size, "batch": batch, "lr": lr, "steps": steps,
                             "temperature": temperature, "max_turns": max_turns, "init_ckpt": init_ckpt,
                             "reward": "episode_reward"}) if wandb_tracking else None
    trainer = ToolLoopGSPO(adapter, group_size=group_size, lr=lr, fmt=fmt, max_turns=max_turns,
                           temperature=temperature)
    out_dir = Path(out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for step in range(1, steps + 1):
        m = trainer.step(random.sample(items, batch))
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
