from types import SimpleNamespace

import pytest
import torch

from laguna_rlvr.visual.baseline import (
    TASK_PROMPT,
    assemble_prompt,
    glm_ocr_transcribe,
    text_chat,
    text_generate,
)


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeProc:  # threads the image into input_ids so we can assert per-item mapping, order + echo-strip
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_dict=True, return_tensors=None):
        img = messages[0]["content"][0]["image"]  # the image is the last prompt token
        return _FakeBatch(input_ids=torch.tensor([[1, ord(img)]]))  # [instruction, image]

    def batch_decode(self, gen, skip_special_tokens=True):
        return ["".join(chr(int(i)) for i in row) for row in gen]


class _EchoOcrModel:  # emits the last input token (the image) as the "transcript"
    device = "cpu"

    def generate(self, input_ids=None, max_new_tokens=None, do_sample=False, **_):
        return torch.cat([input_ids, input_ids[:, -1:]], dim=1)


def test_glm_ocr_transcribe_one_string_per_item_in_order():
    items = [("A", "ref-a"), ("B", "ref-b")]  # (image, label); only the image is read
    # transcript = the image token decoded back, with the echoed prompt sliced off
    assert glm_ocr_transcribe(items, model=_EchoOcrModel(), proc=_FakeProc()) == ["A", "B"]


class _CharTok:  # char-level fake tokenizer: each char <-> its codepoint as one token id
    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]]))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


class _CannedModel:  # echoes input_ids then appends a fixed continuation (to assert the echo is stripped)
    device = "cpu"

    def __init__(self, reply: str = "OK"):
        self._reply = reply

    def generate(self, input_ids=None, max_new_tokens=None, do_sample=False):
        cont = torch.tensor([[ord(c) for c in self._reply]])
        return torch.cat([input_ids, cont], dim=1)


class _CtxLenModel:  # reply = one token = current context length, so we can prove turns accumulate
    device = "cpu"

    def generate(self, input_ids=None, max_new_tokens=None, do_sample=False):
        cont = torch.tensor([[input_ids.shape[1]]])
        return torch.cat([input_ids, cont], dim=1)


@pytest.mark.parametrize("kind", ["html", "python", None])
def test_task_prompt_nonempty_per_kind(kind):
    assert TASK_PROMPT[kind].strip()


def test_assemble_prompt_blind_is_task_only():
    assert assemble_prompt("do X", None) == "do X"  # no visual signal


def test_assemble_prompt_ocr_prepends_transcript():
    p = assemble_prompt("do X", "the transcript")
    assert "the transcript" in p and p.rstrip().endswith("do X")


def test_text_generate_strips_prompt_echo():
    preds = text_generate(_CannedModel("OK"), _CharTok(), ["abc", "de"])
    assert preds == ["OK", "OK"]  # one per prompt, echoed prompt removed


def test_text_chat_accumulates_context_across_turns():
    replies = text_chat(_CtxLenModel(), _CharTok(), ["ab", "cd", "ef"], max_new_tokens=1)
    assert [ord(r) for r in replies] == [2, 5, 8]  # ctx grows 2 -> 3+2 -> 6+2 as each reply feeds back
