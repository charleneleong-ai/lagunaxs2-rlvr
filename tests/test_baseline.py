from laguna_rlvr.visual.baseline import glm_ocr_transcribe


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeProc:  # echoes the image through the batch so we can assert per-item mapping + order
    def __call__(self, images, return_tensors=None):
        return _FakeBatch(pixel=images[0])

    def batch_decode(self, gen, skip_special_tokens=True):
        return [gen]


class _FakeModel:
    device = "cpu"

    def generate(self, pixel=None, **_):
        return pixel


def test_glm_ocr_transcribe_one_string_per_item_in_order():
    items = [("A", "ref-a"), ("B", "ref-b")]  # (image, label); only the image is read
    assert glm_ocr_transcribe(items, model=_FakeModel(), proc=_FakeProc()) == ["A", "B"]
