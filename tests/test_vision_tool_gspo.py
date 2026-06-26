from types import SimpleNamespace

import pytest
import torch

from laguna_rlvr.visual.vision_tool_gspo import (DifficultySampler, _advantages, _eval_solved,
                                                 _gen_positions, _kl_to_ref, episode_reward)


class TestEpisodeReward:
    """Reward is dominated by the discrete solve so optimizing it moves the greedy reader; the small tool /
    efficiency terms only break ties (zero within-group spread -> zero GSPO advantage -> no gradient)."""

    def _done(self, obs=None):
        return [(None, obs), (None, None)] if obs else [(None, None)]

    def test_solve_outscores_any_miss(self):
        solved = episode_reward(True, self._done(), "42.50", "42.50", "", max_turns=4)
        near = episode_reward(False, self._done(), "42.50", "42.5", "", max_turns=4)
        assert solved > near >= 0

    def test_no_miss_outscores_a_solve(self):
        # the redesign's core invariant: the shaping terms (<=0.1 tool + <=0.05 effic) can never let even a
        # perfectly-shaped miss out-reward a solve. The earlier char-similarity proxy violated this — a close
        # miss could out-reward a solve — which is why reward rose while the greedy mode never moved.
        solve = episode_reward(True, self._done(), "a", "a", "carries a", max_turns=4)        # wrong tool, solved
        miss = episode_reward(False, self._done("[ocr of x]\na"), "a", "x", "carries a", max_turns=4)  # right tool
        assert solve > miss

    def test_correct_ocr_decision_earns_the_tool_bonus(self):
        # gold IS in the transcript -> calling ocr was the right call -> +0.1 over not calling it
        called = episode_reward(False, self._done("[ocr of d.png]\nTotal 42.50"), "42.50", "x", "Total 42.50", max_turns=4)
        skipped = episode_reward(False, self._done(), "42.50", "x", "Total 42.50", max_turns=4)
        assert called == pytest.approx(skipped + 0.1)

    def test_tool_bonus_scales_the_ocr_signal(self):
        # the ocrvqa knob: a larger tool_bonus widens the pro-ocr gap on a transcription item that never solves
        called = episode_reward(False, self._done("[ocr of d.png]\nTotal 42.50"), "42.50", "x", "Total 42.50",
                                max_turns=4, tool_bonus=0.4)
        skipped = episode_reward(False, self._done(), "42.50", "x", "Total 42.50", max_turns=4, tool_bonus=0.4)
        assert called == pytest.approx(skipped + 0.4)

    def test_fewer_turns_scores_higher_when_solved(self):
        # both never call ocr (same tool decision), so the gap is purely the efficiency tie-breaker
        fast = episode_reward(True, [(None, None)], "a", "a", "", max_turns=4)                         # 1 turn
        slow = episode_reward(True, [(None, "No valid tool call"), (None, None)], "a", "a", "", max_turns=4)  # 2
        assert fast > slow


class TestEvalSolved:
    """(sampled, greedy) split: sampled averages k draws/item under the training decode (do_sample, no
    rep-penalty — the distribution GSPO optimizes); greedy is one deterministic rep-penalty draw/item."""

    def test_sampled_averages_k_draws_greedy_is_one_per_item(self, monkeypatch):
        calls = []

        def stub(adapter, image, image_id, question, transcript, gold, *, fmt, max_turns,
                 do_sample=False, temperature=1.0, top_p=1.0, repetition_penalty=1.3):
            calls.append(do_sample)
            solved = (transcript == "solve") if do_sample else (gold == "yes")  # decode-dependent outcome
            return solved, 1, "r"

        monkeypatch.setattr("laguna_rlvr.visual.vision_tool_gspo.run_episode", stub)
        items = [("img", "a.png", "q", "solve", "yes"), ("img", "b.png", "q", "miss", "no")]
        adapter = SimpleNamespace(llm=SimpleNamespace(gradient_checkpointing_disable=lambda: None))

        sampled, greedy = _eval_solved(adapter, items, fmt="poolside", max_turns=4, temperature=0.8, k=3)

        assert greedy == 0.5  # 1 of 2 items has gold "yes"
        assert sampled == 0.5  # 1 of 2 items has transcript "solve", averaged over k draws each
        assert calls.count(False) == 2 and calls.count(True) == 2 * 3  # 1 greedy + k sampled per item


class TestDifficultySampler:
    """Weights are the Bernoulli variance p*(1-p) of the per-item solve EMA (+floor) — boundary items
    (p~0.5) outweigh saturated ones (p~0/1), the first observation replaces the optimistic prior outright,
    and the floor keeps every item drawable."""

    def test_boundary_item_outweighs_saturated(self):
        s = DifficultySampler(3, floor=0.05)
        s.update(0, 0.0)   # always-miss -> variance 0
        s.update(1, 0.5)   # boundary   -> variance 0.25
        s.update(2, 1.0)   # always-solve -> variance 0
        w = s.weights().tolist()
        assert w[1] > w[0] and w[1] > w[2]
        assert w[0] == pytest.approx(0.05) and w[2] == pytest.approx(0.05)  # floor keeps them nonzero

    def test_first_obs_replaces_prior_then_blends(self):
        s = DifficultySampler(1, alpha=0.3)
        s.update(0, 0.0)                       # first obs: replace 0.5 prior outright
        assert s.p[0] == pytest.approx(0.0)
        s.update(0, 1.0)                       # second obs: EMA blend
        assert s.p[0] == pytest.approx(0.3)

    def test_sample_draws_distinct_indices(self):
        s = DifficultySampler(5)
        idxs = s.sample(3)
        assert len(idxs) == len(set(idxs)) == 3

    def test_unseen_items_start_at_max_weight(self):
        s = DifficultySampler(2, floor=0.05)
        assert s.weights().tolist() == pytest.approx([0.3, 0.3])  # floor + 0.5*0.5, optimistic

    def test_higher_floor_keeps_a_collapsed_corpus_drawable(self):
        # the ocrvqa knob: once a corpus's solve-EMA collapses to 0 its variance is 0, so the floor is its
        # entire sampling weight — raising the floor stops it being abandoned out of the boundary band
        lo, hi = DifficultySampler(1, floor=0.05), DifficultySampler(1, floor=0.30)
        lo.update(0, 0.0)
        hi.update(0, 0.0)
        assert lo.weights()[0] == pytest.approx(0.05)
        assert hi.weights()[0] == pytest.approx(0.30)  # ~6x more likely to keep getting gradient


class TestGenPositions:
    """The index map that decides which trajectory tokens get credited under the policy gradient — an
    off-by-one here silently trains on the wrong tokens, so it's pinned exactly."""

    def test_direct_answer_is_contiguous_after_the_prompt(self):
        # one turn, no observation: the 3 generated tokens sit right after the prompt.
        assert _gen_positions(5, [(3, 0)]) == [5, 6, 7]

    def test_observation_shifts_the_next_turn_past_the_injected_text(self):
        # ocr-then-answer: turn-0 gen [5,6], a 4-token ocr observation, then turn-1 gen — which must skip
        # the observation's positions (they're context, not generated): 5+2+4 = 11.
        assert _gen_positions(5, [(2, 4), (3, 0)]) == [5, 6, 11, 12, 13]

    @pytest.mark.parametrize("segments", [[], [(0, 0)]])
    def test_no_generated_tokens(self, segments):
        assert _gen_positions(5, segments) == []


class TestCrossBatchAdvantageNormalization:
    """Cross-batch normalization gives nonzero advantages for the unanimous-group pattern that
    killed ~47% of training steps: item A all-solves, item B all-misses. Per-item normalization
    (the prior code) returned zero advantages for both because each G-group had zero internal
    variance. Cross-batch normalization uses the full batch*G reward distribution as the baseline,
    so item A's rollouts get positive advantages and item B's get negative ones."""

    def _cross_batch_adv(self, rewards_per_item: list[list[float]]) -> list[list[float]]:
        """The real cross-batch path of `_advantages` (distinct corpus labels are irrelevant when
        corpus_norm is off — centring is the global batch mean)."""
        groups = [torch.tensor(rs) for rs in rewards_per_item]
        corpora = [str(i) for i in range(len(groups))]
        return [a.tolist() for a in _advantages(groups, corpora, corpus_norm=False)]

    def _per_item_adv(self, rewards: list[float]) -> list[float]:
        """Old per-item normalization for contrast."""
        rew = torch.tensor(rewards)
        return ((rew - rew.mean()) / (rew.std() + 1e-4)).tolist()

    def test_allsolve_allmiss_batch_gives_nonzero_advantages(self):
        """Smoking-gun case: steps 120/140 had reward=0.575, solved=0.500, grad_norm=0.00.
        Item A all-solves (reward ~1.15 each), item B all-misses (reward 0.0 each).
        Cross-batch normalization must yield strictly positive adv for item A and strictly
        negative adv for item B."""
        item_a = [1.15] * 8  # all-solve group
        item_b = [0.0] * 8   # all-miss group
        adv_a, adv_b = self._cross_batch_adv([item_a, item_b])
        assert all(a > 0 for a in adv_a), "all-solve item should get positive cross-batch advantages"
        assert all(a < 0 for a in adv_b), "all-miss item should get negative cross-batch advantages"

    def test_per_item_normalization_gives_zero_adv_for_same_input(self):
        """Document the old failure mode: per-item normalization returns zero advantages for both
        items when each group is internally unanimous, even if the batch is perfectly contrastive."""
        item_a = [1.15] * 8
        item_b = [0.0] * 8
        adv_a = self._per_item_adv(item_a)
        adv_b = self._per_item_adv(item_b)
        assert all(abs(a) < 1e-3 for a in adv_a), "all-same group should have near-zero per-item advantages"
        assert all(abs(a) < 1e-3 for a in adv_b)

    def test_mixed_group_produces_same_sign_structure_under_both_normalizations(self):
        """When a group has genuine within-item variance (mixed solve/miss), cross-batch and
        per-item normalization should agree on sign: solved rollouts get positive adv, missed get
        negative (up to the cross-item baseline shift)."""
        # item A: 4 solve, 4 miss; item B: all miss (so cross-batch mean is low, both sign-correct)
        item_a = [1.15, 1.15, 1.15, 1.15, 0.0, 0.0, 0.0, 0.0]
        item_b = [0.0] * 8
        adv_a, _ = self._cross_batch_adv([item_a, item_b])
        # solved rollouts of item A (indices 0-3) must have higher adv than missed (4-7)
        assert all(adv_a[i] > adv_a[j] for i in range(4) for j in range(4, 8))


class TestCorpusNormAdvantage:
    """corpus_norm centres each item on a RUNNING per-corpus baseline instead of the global batch mean —
    the targeted ocrvqa fix. The invariant: an unsolved ocr-call on a transcription corpus, batched with a
    higher-reward reasoning corpus, gets NEGATIVE advantage under cross-batch (swamped) but POSITIVE under
    corpus_norm (it beats its corpus's own typical reward)."""

    def test_unsolved_ocr_call_flips_positive_under_corpus_norm(self):
        chart = torch.tensor([1.1, 1.1, 1.1, 0.0])          # reasoning corpus, mostly solves
        ocr = torch.tensor([0.4, 0.0, 0.0, 0.0])            # ocrvqa: all miss; idx 0 called ocr (tool_bonus)
        corpora = ["chartqa", "ocrvqa"]
        baselines = {"chartqa": 0.8, "ocrvqa": 0.1}         # warmed-up running EMAs
        x_adv = _advantages([chart, ocr], corpora, corpus_norm=False)[1]
        c_adv = _advantages([chart, ocr], corpora, corpus_norm=True, baselines=baselines)[1]
        assert x_adv[0] < 0, "cross-batch swamps the unsolved ocr-call below the solve-inflated mean"
        assert c_adv[0] > 0 > c_adv[1], "corpus_norm lifts the ocr-call above its corpus baseline; direct-answer stays below"

    def test_missing_baseline_falls_back_to_batch_mean(self):
        # an unseen corpus (cold EMA) must not crash — it centres on the global mean, == cross-batch for it
        groups = [torch.tensor([1.0, 0.0]), torch.tensor([0.5, 0.5])]
        cn = _advantages(groups, ["a", "b"], corpus_norm=True, baselines={})
        xb = _advantages(groups, ["a", "b"], corpus_norm=False)
        assert cn[0].tolist() == pytest.approx(xb[0].tolist())


class TestKLToRef:
    """k3 estimator of KL(current || SFT-reference) — the anti-collapse leash. Non-negative, zero only when
    the policies match, and growing with the gap (so a policy drifting toward a degenerate constant pays)."""

    def test_zero_when_policies_match(self):
        lp = torch.tensor([-0.5, -1.2, -0.1])
        assert _kl_to_ref(lp, lp).tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    def test_nonnegative_and_grows_with_divergence(self):
        cur = torch.tensor([-0.5, -0.5])
        near, far = torch.tensor([-0.6, -0.6]), torch.tensor([-3.0, -3.0])
        assert (_kl_to_ref(cur, far) >= 0).all() and (_kl_to_ref(cur, near) >= 0).all()
        assert _kl_to_ref(cur, far).mean() > _kl_to_ref(cur, near).mean()
