from types import SimpleNamespace

import pytest

from laguna_rlvr.visual.vision_tool_gspo import _eval_solved, _gen_positions, episode_reward


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
