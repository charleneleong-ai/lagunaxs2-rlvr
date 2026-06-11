import pytest

from laguna_rlvr.visual.vision_tool_gspo import _gen_positions, episode_reward


class TestEpisodeReward:
    """The granular reward must give within-group spread — a flat 0/1 makes unanimous groups (zero GSPO
    advantage -> no gradient), which is the stall this reward fixes."""

    def _done(self, obs=None):
        return [(None, obs), (None, None)] if obs else [(None, None)]

    def test_solve_outscores_any_miss(self):
        solved = episode_reward(True, self._done(), "42.50", "42.50", "", max_turns=4)
        near = episode_reward(False, self._done(), "42.50", "42.5", "", max_turns=4)
        assert solved > near >= 0

    def test_closer_miss_scores_higher_than_far_miss(self):
        # the granularity property: char-similarity separates near from far, where word-jaccard tied them at ~0
        near = episode_reward(False, self._done(), "soil", "soi", "", max_turns=4)
        far = episode_reward(False, self._done(), "soil", "zzzz", "", max_turns=4)
        assert near > far

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
