import pytest
from PIL import Image

from laguna_rlvr.visual.corpora import CAULDRON_VQA, CHOICES, REGISTRY, _resolve_vqa, load_text_image
from laguna_rlvr.visual.data import SyntheticOCR
from laguna_rlvr.visual.hf_image_text import parse_cauldron_vqa


def _row(user, assistant, with_image=True):
    img = Image.new("RGB", (4, 4)) if with_image else None
    return {"images": [img] if with_image else [], "texts": [{"user": user, "assistant": assistant}]}


def test_parse_cauldron_vqa_extracts_first_turn():
    r = parse_cauldron_vqa(_row("<image>\nWhat color is the car?", "red"))
    assert r["question"] == "What color is the car?" and r["answer"] == "red"
    assert r["image"].size == (4, 4)


def test_parse_cauldron_vqa_skips_incomplete():
    assert parse_cauldron_vqa(_row("", "red")) is None          # no question
    assert parse_cauldron_vqa(_row("Q?", "")) is None           # no answer
    assert parse_cauldron_vqa(_row("Q?", "a", with_image=False)) is None  # no image
    assert parse_cauldron_vqa({"images": [], "texts": []}) is None


def test_load_text_image_dispatches_synthetic():  # offline — no network/model
    ds = load_text_image("synthetic", 8)
    assert isinstance(ds, SyntheticOCR) and len(ds) == 8


def test_align_mix_is_registered_and_reading_biased():
    from laguna_rlvr.visual.corpora import CHOICES, _ALIGN_MIX
    assert "align" in CHOICES
    total = sum(w for _, w in _ALIGN_MIX)
    # reading sources (synthetic visible-text + cauldron_* transcription) must dominate so Stage-1
    # anchors readout; code-target corpora (websight=HTML) stay a minority or readout erodes (2026-05).
    reading = sum(w for n, w in _ALIGN_MIX if n == "synthetic" or n.startswith("cauldron"))
    assert reading / total >= 0.7
    assert max(_ALIGN_MIX, key=lambda kv: kv[1])[0] == "synthetic"  # synthetic the largest single anchor


def test_load_text_image_align_dispatches_to_mixture():  # offline — override to synthetic-only
    ds = load_text_image("align", 8, mixture=[("synthetic", 1.0)])
    assert len(ds) == 8 and ds[0][2] == "synthetic"


def test_cauldron_dataset_extracts_image_and_transcription(tmp_path, monkeypatch):  # offline — mocked
    from PIL import Image

    import laguna_rlvr.visual.hf_image_text as hit

    monkeypatch.setattr(hit, "_CACHE_DIR", tmp_path / "cache")
    rows = [{"images": [Image.new("RGB", (8, 8))],
             "texts": [{"user": "Type out the text.", "assistant": f"line {i}", "source": "x"}]}
            for i in range(2)]
    monkeypatch.setattr(hit, "load_dataset", lambda *a, **k: iter(rows))
    ds = hit.CauldronDataset("rendered_text", n=2)
    assert len(ds) == 2
    img, txt = ds[0]
    assert txt == "line 0"  # first turn's assistant becomes the recon transcription target


def test_load_text_image_unknown_raises():
    with pytest.raises(ValueError):
        load_text_image("nope", 4)


def test_parse_mixture():
    from laguna_rlvr.visual.corpora import parse_mixture

    assert parse_mixture("websight=0.6, webcode2m=0.4") == [("websight", 0.6), ("webcode2m", 0.4)]


def test_mixture_blends_corpora_by_weight():  # offline — synthetic only
    from laguna_rlvr.visual.corpora import _Mixture

    mix = _Mixture([("synthetic", 0.75), ("synthetic", 0.25)], n=8)
    assert len(mix) == 8  # round(8*.75)=6 + round(8*.25)=2
    img, txt, corpus = mix[0]
    assert txt and corpus == "synthetic"  # yields (image, text, corpus-tag)


def test_hf_image_text_caches_to_disk(tmp_path, monkeypatch):  # offline — load_dataset mocked
    from PIL import Image

    import laguna_rlvr.visual.hf_image_text as hit

    monkeypatch.setattr(hit, "_CACHE_DIR", tmp_path / "cache")
    rows = [{"image": Image.new("RGB", (8, 8), c), "text": f"<p>{c}</p>"} for c in ("red", "blue")]
    monkeypatch.setattr(hit, "load_dataset", lambda *a, **k: iter(rows))
    first = hit.HFImageTextDataset("fake/repo", n=2)
    assert len(first) == 2 and first[0][1] == "<p>red</p>"

    # second build must hit the disk cache — load_dataset now raises if the network is touched
    def _boom(*a, **k):
        raise AssertionError("re-streamed despite cache")

    monkeypatch.setattr(hit, "load_dataset", _boom)
    second = hit.HFImageTextDataset("fake/repo", n=2)
    assert [second[i][1] for i in range(len(second))] == ["<p>red</p>", "<p>blue</p>"]


def test_ocr_probe_falls_back_to_synthetic(monkeypatch):  # detached run must survive a fetch failure
    from laguna_rlvr.visual import train

    def _boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(train, "HFImageTextDataset", _boom)
    probe = train._ocr_probe(seed=1)
    assert len(probe) == 16  # SyntheticOCR floor, not the remote set
    _img, text = probe[0]
    assert text  # (image, text) transcription pairs


@pytest.mark.parametrize("label,kind,expected", [
    ('ax.set_title("Quarterly Revenue")', "python", "Quarterly Revenue"),
    ('plt.title( "Sales" )', "python", "Sales"),
    ("<title>Pricing</title>", "html", "Pricing"),
    ("<body><h1>Welcome <span>back</span></h1></body>", "html", "Welcome back"),  # nested tags stripped
])
def test_extract_needle_pulls_title(label, kind, expected):
    from laguna_rlvr.visual.corpora import extract_needle
    assert extract_needle(label, kind) == expected


@pytest.mark.parametrize("label,kind", [
    ("plt.plot([1, 2, 3])", "python"),  # no title call
    ("<p>no title</p>", "html"),         # no title/h1
    ("anything", None),                  # non-code corpus
])
def test_extract_needle_none_when_absent(label, kind):
    from laguna_rlvr.visual.corpora import extract_needle
    assert extract_needle(label, kind) is None


def test_caption_configs_registered():
    for name in ("cauldron_localized_narratives", "cauldron_screen2words"):
        assert name in REGISTRY and name in CHOICES


def test_resolve_vqa_dispatch():
    assert _resolve_vqa("textvqa") == "spec"
    assert _resolve_vqa("vqav2") == "cauldron" and "vqav2" in CAULDRON_VQA
    with pytest.raises(ValueError):
        _resolve_vqa("nope")


def test_qasft_dataset_extracts_needle_triples():
    from laguna_rlvr.visual.corpora import QASFTDataset

    class _Base:  # (image, label, corpus) rows like the mixture
        rows = [("imgA", 'ax.set_title("Sales")', "chartmimic"),    # python -> needle "Sales"
                ("imgB", "<title>Home</title>", "design2code"),      # html -> needle "Home"
                ("imgC", "plt.plot([1,2])", "chartmimic"),           # no title -> dropped
                ("imgD", "prose", "swebench_mm")]                    # kind None -> dropped
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    ds = QASFTDataset(_Base())
    assert len(ds) == 2  # only needle-bearing rows
    # (image, needle, corpus, question) — question is "" for needle-extracted rows (set only for VQA)
    assert ds[0] == ("imgA", "Sales", "chartmimic", "")
    assert ds[1] == ("imgB", "Home", "design2code", "")


def test_qasft_design_codegen_uses_full_code_and_prompt():
    from laguna_rlvr.visual.corpora import QASFTDataset, TASK_PROMPT

    class _Base:  # html/python rows + a non-code row
        rows = [("imgA", "<html><h1>Hi</h1></html>", "design2code"),  # html -> code-gen
                ("imgB", 'ax.set_title("Sales")', "chartmimic"),       # python -> code-gen
                ("imgC", "prose", "swebench_mm")]                       # kind None -> dropped
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    ds = QASFTDataset(_Base(), design_codegen=True)
    assert len(ds) == 2  # the two code rows; swebench prose has no kind -> dropped
    # full code is the answer, asked via the kind's generation prompt (not the title-needle)
    assert ds[0] == ("imgA", "<html><h1>Hi</h1></html>", "design2code", TASK_PROMPT["html"])
    assert ds[1] == ("imgB", 'ax.set_title("Sales")', "chartmimic", TASK_PROMPT["python"])
