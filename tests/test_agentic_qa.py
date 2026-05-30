import pytest
import torch

from laguna_rlvr.seed import seed_everything
from laguna_rlvr.visual.data import render_text
from laguna_rlvr.visual.encoders import load_encoder
from laguna_rlvr.visual.model import IMAGE_TOKEN, Turn, VisualAdapter

BASE = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def adapter():
    seed_everything(42)
    return VisualAdapter(encoder=load_encoder("glm_ocr", pool=4), base_llm=BASE)


class TestMultiTurnMultimodalQA:
    """Agentic validation: vision must work across multiple QA turns, not only as a prefix."""

    def test_two_turn_qa_yields_a_reply_per_turn(self, adapter):
        turns = [
            Turn(f"{IMAGE_TOKEN}\nWhat text is shown?", [render_text("hello 42", seed=1)]),
            Turn(f"And this one? {IMAGE_TOKEN}", [render_text("total 7", seed=2)]),
        ]
        replies = adapter.chat(turns, max_new_tokens=8)
        assert len(replies) == 2
        assert all(isinstance(r, str) and r.strip() for r in replies)

    def test_text_only_follow_up_turn(self, adapter):
        # a turn with no <image> (a follow-up question) must still generate, conditioned on prior vision
        turns = [
            Turn(f"{IMAGE_TOKEN}\nRead the image.", [render_text("invoice 9", seed=3)]),
            Turn("Now summarize what you saw."),
        ]
        replies = adapter.chat(turns, max_new_tokens=8)
        assert len(replies) == 2 and all(r.strip() for r in replies)

    def test_multi_image_single_prompt_splices_in_order(self, adapter):
        d = adapter.llm.config.hidden_size
        vis = [torch.zeros(1, 3, d, device=adapter.llm.device, dtype=adapter.llm.dtype) for _ in range(2)]
        text = f"a {IMAGE_TOKEN} b {IMAGE_TOKEN} c"
        n_text = adapter.tok(text, return_tensors="pt").input_ids.shape[1]
        out = adapter._embed_multi(text, vis)
        assert out.shape == (1, n_text - 2 + 6, d)  # 2 markers -> 2x3 vision tokens

    def test_marker_image_count_mismatch_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter._embed_multi(f"one {IMAGE_TOKEN} marker", [])
