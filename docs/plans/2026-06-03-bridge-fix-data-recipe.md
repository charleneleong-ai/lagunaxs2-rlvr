# Bridge Fix via Data Recipe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground the vision adapter by adopting the reference's data recipe — general image-caption Stage-1 + diverse image-dependent VQA Stage-2 — using `the_cauldron` configs through existing loaders, with plain LM loss.

**Architecture:** Caption Stage-1 reuses the existing `CauldronDataset` (already reads the assistant turn) via two new `REGISTRY` entries — no new loader. Stage-2 adds a `CauldronVQADataset` (reads the user/assistant turn as question/answer) wired into `load_vqa`. Both runs use existing flags (`--dataset align/mix`, `--mixture`, `--vqa`, `--init-projector`, `--lora-rank`).

**Tech Stack:** PyTorch, HuggingFace `datasets` (streaming + Arrow cache), `the_cauldron`, pytest.

---

## File structure

- `src/laguna_rlvr/visual/hf_image_text.py` — add `parse_cauldron_vqa` (pure row parser), `_cauldron_vqa_rows` (generator), `CauldronVQADataset` (class). Mirrors the existing `_cauldron_rows`/`CauldronDataset` and `_vqa_rows`/`VQADataset` pairs.
- `src/laguna_rlvr/visual/corpora.py` — add caption configs to `REGISTRY`; add `CAULDRON_VQA` list; make `load_vqa` dispatch by source type.
- `tests/test_corpora.py` — parser unit tests, registry membership, `load_vqa` dispatch (all network-free).
- `logs/run_stage1_caption.sh`, `logs/run_stage2_instruct.sh` — launch scripts (gitignored).

Note on the recon prompt: the spec mentioned a caption-specific prompt. Deferred (YAGNI) — Stage-1 grounding is prompt-agnostic (the projector learns image→text regardless of the fixed `_PROMPT` prefix) and Stage-2 retrains with its own `qa` prompts. Revisit only if alignment underperforms. No change to `model.py`.

---

## Task 1: the_cauldron VQA row parser (pure, testable)

**Files:**
- Modify: `src/laguna_rlvr/visual/hf_image_text.py` (add after `_cauldron_rows`, ~line 106)
- Test: `tests/test_corpora.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_corpora.py` (create the file if absent, with `from PIL import Image` and `from laguna_rlvr.visual.hf_image_text import parse_cauldron_vqa` at top):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run pytest tests/test_corpora.py -k parse_cauldron_vqa -v`
Expected: FAIL — `ImportError: cannot import name 'parse_cauldron_vqa'`

- [ ] **Step 3: Write the parser**

In `src/laguna_rlvr/visual/hf_image_text.py`, immediately after `_cauldron_rows` (line ~106):

```python
def parse_cauldron_vqa(row: dict) -> dict | None:
    """First (user, assistant) turn of a the_cauldron row -> {image, question, answer}, or None if any
    field is missing. Strips the '<image>' placeholder the cauldron prepends to the user question."""
    images, texts = row.get("images"), row.get("texts")
    if not images or not texts:
        return None
    turn = texts[0]
    q = (turn.get("user") or "").replace("<image>", "").strip()
    a = (turn.get("assistant") or "").strip()
    if images[0] is None or not q or not a:
        return None
    return {"image": images[0].convert("RGB"), "question": q, "answer": a}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run pytest tests/test_corpora.py -k parse_cauldron_vqa -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/laguna_rlvr/visual/hf_image_text.py tests/test_corpora.py
git commit -m "feat(data): parse_cauldron_vqa — extract Q/A from a the_cauldron turn"
```

---

## Task 2: CauldronVQADataset (generator + class)

**Files:**
- Modify: `src/laguna_rlvr/visual/hf_image_text.py` (generator after `parse_cauldron_vqa`; class after `CauldronDataset`, ~line 195)

- [ ] **Step 1: Write the generator**

After `parse_cauldron_vqa`:

```python
def _cauldron_vqa_rows(*, shard_idx, num_shards, per_shard, offset, config, split):
    """Yield {image, question, answer} rows from a the_cauldron VQA config — for from_generator."""
    for i in shard_idx:
        base = load_dataset("HuggingFaceM4/the_cauldron", config, split=split, streaming=True)
        rows = islice(base.shard(num_shards, i), per_shard) if num_shards > 1 else \
            islice(base, offset, offset + per_shard)
        for row in track(rows, total=per_shard, description=f"cauldron-vqa/{config} shard {i}"):
            if (r := parse_cauldron_vqa(row)) is not None:
                yield r
```

- [ ] **Step 2: Write the class**

After `CauldronDataset` (~line 195), mirroring `VQADataset`:

```python
class CauldronVQADataset(Dataset):
    """(image, question, answer) triples from a the_cauldron VQA config (vqav2/okvqa/visual7w) — the
    first user/assistant turn. General, image-dependent VQA for Stage-2 instruction (reference recipe).
    Same streaming + disk cache as VQADataset."""

    def __init__(self, config: str, *, split: str = "train", n: int = 2000, offset: int = 0):
        key = "cauldronvqa__" + "__".join(str(p) for p in (config, split, n, offset))
        self._ds = _cached_or_stream(key, lambda: self._stream(config, split, n, offset))

    @staticmethod
    def _stream(config, split, n, offset) -> HFDataset:
        return _materialize(_cauldron_vqa_rows, _VQA_FEATURES, n=n, offset=offset,
                            n_files=lambda: _n_shards("HuggingFaceM4/the_cauldron", config, split),
                            error=f"no usable rows from the_cauldron/{config} (vqa)",
                            config=config, split=split)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        row = self._ds[i]
        return row["image"], row["question"], row["answer"]
```

- [ ] **Step 3: Verify import + shape**

Run: `mise exec -- uv run python -c "from laguna_rlvr.visual.hf_image_text import CauldronVQADataset, _cauldron_vqa_rows; print('ok')"`
Expected: prints `ok` (no import error; confirms `_VQA_FEATURES`, `_materialize`, `_n_shards`, `HFDataset` are all in scope — they are, used by the sibling classes).

- [ ] **Step 4: Commit**

```bash
git add src/laguna_rlvr/visual/hf_image_text.py
git commit -m "feat(data): CauldronVQADataset — the_cauldron VQA configs as (image,q,a)"
```

---

## Task 3: Wire caption corpora + VQA dispatch into corpora.py

**Files:**
- Modify: `src/laguna_rlvr/visual/corpora.py` (`REGISTRY` ~line 131; `load_vqa` ~line 244)
- Test: `tests/test_corpora.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_corpora.py`:

```python
from laguna_rlvr.visual.corpora import REGISTRY, CHOICES, CAULDRON_VQA, _resolve_vqa

def test_caption_configs_registered():
    for name in ("cauldron_localized_narratives", "cauldron_screen2words"):
        assert name in REGISTRY and name in CHOICES

def test_resolve_vqa_dispatch():
    # lmms-lab suite -> VQADataset spec; cauldron configs -> CauldronVQADataset; unknown -> error
    assert _resolve_vqa("textvqa") == "spec"
    assert _resolve_vqa("vqav2") == "cauldron" and "vqav2" in CAULDRON_VQA
    import pytest
    with pytest.raises(ValueError):
        _resolve_vqa("nope")
```

- [ ] **Step 2: Run to verify it fails**

Run: `mise exec -- uv run pytest tests/test_corpora.py -k "caption_configs or resolve_vqa" -v`
Expected: FAIL — `ImportError: cannot import name 'CAULDRON_VQA'` / `_resolve_vqa`

- [ ] **Step 3: Add the caption REGISTRY entries**

In `corpora.py` `REGISTRY` (after `"cauldron_iam": ...`, ~line 141):

```python
    "cauldron_localized_narratives": _cauldron("localized_narratives"),  # general dense captions (grounding)
    "cauldron_screen2words": _cauldron("screen2words"),                  # UI screenshot -> summary
```

- [ ] **Step 4: Add CAULDRON_VQA + dispatch helper + load_vqa**

Replace the existing `load_vqa` (lines ~244-248) with:

```python
CAULDRON_VQA = ["vqav2", "okvqa", "visual7w"]  # the_cauldron general image-dependent VQA (Stage-2)


def _resolve_vqa(name: str) -> str:
    """Which loader backs a VQA name: 'spec' (lmms-lab via VQA_SPECS) or 'cauldron' (the_cauldron)."""
    if name in VQA_SPECS:
        return "spec"
    if name in CAULDRON_VQA:
        return "cauldron"
    raise ValueError(f"unknown VQA set {name!r}; choices: {list(VQA_SPECS) + CAULDRON_VQA}")


def load_vqa(names: list[str], n: int) -> list[tuple]:
    from laguna_rlvr.visual.hf_image_text import CauldronVQADataset, VQADataset

    def build(name):
        src = _resolve_vqa(name)
        ds = VQADataset(n=n, **VQA_SPECS[name]) if src == "spec" else CauldronVQADataset(name, n=n)
        return ds, name

    # Each VQA set is an independent I/O-bound materialization; load concurrently to overlap startup.
    with ThreadPoolExecutor(max_workers=max(1, len(names))) as ex:
        return list(ex.map(build, names))
```

- [ ] **Step 5: Run to verify it passes**

Run: `mise exec -- uv run pytest tests/test_corpora.py -v`
Expected: PASS (all tasks 1+3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/laguna_rlvr/visual/corpora.py tests/test_corpora.py
git commit -m "feat(data): register caption configs + dispatch load_vqa to the_cauldron VQA"
```

---

## Task 4: Stage-1 caption-alignment run + bridge validation

**Files:**
- Create: `logs/run_stage1_caption.sh` (gitignored)

- [ ] **Step 1: Write the launch script**

```bash
cat > logs/run_stage1_caption.sh <<'SH'
#!/bin/bash
# Stage-1 CAPTION alignment (reference recipe): general image->caption grounds the projector via the
# plain recon LM loss (captioning is inherently image-dependent). the_cauldron caption configs.
cd /home/ubuntu/lagunaxs2-rlvr-stage0 || exit 1
set -a; . /home/ubuntu/lagunaxs2-rlvr/.env 2>/dev/null; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp/mpl-laguna HF_HOME="$HOME/.cache/huggingface"
MIX="cauldron_localized_narratives=0.5,cauldron_textcaps=0.2,cauldron_screen2words=0.1,synthetic=0.2"
for a in 1 2 3 4 5 6; do
  echo "=== attempt $a $(date -u +%H:%M:%SZ) ==="
  /home/ubuntu/.local/bin/mise exec -- uv run python -u -m laguna_rlvr.visual.train \
    --config configs/mm_adapter/a100-80gb-laguna-bf16-microbatch2.toml --dataset align --base poolside/Laguna-XS.2 \
    --encoder siglip_naflex --projector resampler --objective recon \
    --mixture "$MIX" --name-suffix stage1caption --steps 3000 --n-train 50000 --resume \
    --description "Stage-1 caption alignment (reference recipe): general image->caption grounds the projector via plain recon LM loss. the_cauldron localized_narratives/textcaps/screen2words + synthetic."
  c=$?; echo "=== exit code=$c attempt=$a $(date -u +%H:%M:%SZ) ==="; [ "$c" -eq 0 ] && break; sleep 15
done
SH
chmod +x logs/run_stage1_caption.sh
```

- [ ] **Step 2: Launch detached (PPID=1) + verify alive**

```bash
cd /home/ubuntu/lagunaxs2-rlvr-stage0
ts=$(date -u +%Y%m%dT%H%M%SZ); log="logs/stage1caption_${ts}.log"
setsid nohup bash logs/run_stage1_caption.sh </dev/null >>"$log" 2>&1 & disown
sleep 25; ps -eo pid,ppid,cmd | grep run_stage1_caption | grep -v grep | head -1
```
Expected: a `bash logs/run_stage1_caption.sh` line with PPID `1`. Confirm `grep -aE "guardrails|step 1/" "$log"` shows training starts.

- [ ] **Step 3: After it finishes, validate the bridge on the new checkpoint**

The Stage-1 checkpoint lands at `results/visual/siglip_naflex__Laguna-XS.2__align__stage1caption/best.pt`. Point the existing diagnostics at it:

```bash
# loss image-dependence (was ~7%): edit logs/diag_projector_grad.py CKPT-load is fresh-adapter, so instead
# run the base-decoder sanity on the new ckpt:
sed 's#siglip__Laguna-XS.2__mix__stage2_anyres_lora64#siglip_naflex__Laguna-XS.2__align__stage1caption#; s#load_encoder("siglip", pool=2)#load_encoder("siglip_naflex", pool=4)#; s#unfreeze="lora", use_anchor=True, lora_rank=64#unfreeze="", use_anchor=True#' \
  logs/diag_base_decoder.py > logs/diag_caption_ckpt.py
mise exec -- uv run python -u logs/diag_caption_ckpt.py 2>&1 | grep -E "PASS|fail|READ|miss|->"
```
Expected signal (success): the VISION huge-font / in-distribution probes start landing reads (> 0/6) instead of emitting reasoning chatter — grounding has formed. If still 0/6, grounding did not form from captioning; stop and reassess (see plan tail).

- [ ] **Step 4: Commit the validation note**

No code change; record the outcome in the sweep writeup (see Task 5 commit). No commit here.

---

## Task 5: Stage-2 instruction run + per-task eval

**Files:**
- Create: `logs/run_stage2_instruct.sh` (gitignored)

- [ ] **Step 1: Write the launch script**

```bash
cat > logs/run_stage2_instruct.sh <<'SH'
#!/bin/bash
# Stage-2 INSTRUCT (reference recipe): diverse image-dependent VQA + design corpora, projector+LoRA,
# warm-started from the caption-aligned Stage-1. Fixed word-boundary qa metric, qa-eval-n 160.
cd /home/ubuntu/lagunaxs2-rlvr-stage0 || exit 1
set -a; . /home/ubuntu/lagunaxs2-rlvr/.env 2>/dev/null; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp/mpl-laguna HF_HOME="$HOME/.cache/huggingface"
STAGE1="results/visual/siglip_naflex__Laguna-XS.2__align__stage1caption/best.pt"
MIX="websight=0.4,webcode2m=0.3,synthetic=0.3"             # design corpora as the needle base
VQA="textvqa,docvqa,chartqa,ocrvqa,vqav2,okvqa,visual7w"   # reading suite + the_cauldron general VQA
for a in 1 2 3 4 5 6; do
  echo "=== attempt $a $(date -u +%H:%M:%SZ) ==="
  /home/ubuntu/.local/bin/mise exec -- uv run python -u -m laguna_rlvr.visual.train \
    --config configs/mm_adapter/a100-80gb-laguna-bf16-microbatch2.toml --dataset mix --base poolside/Laguna-XS.2 \
    --encoder siglip_naflex --projector resampler --objective qa --anchor --unfreeze lora --lr 2e-5 \
    --mixture "$MIX" --vqa "$VQA" --lora-rank 64 \
    --init-projector "$STAGE1" --name-suffix stage2instruct --steps 3000 --n-train 16000 \
    --qa-eval-n 160 --resume \
    --description "Stage-2 instruct (reference recipe): the_cauldron general VQA (vqav2/okvqa/visual7w) + reading suite + design corpora, projector+LoRA r=64, warm-started from caption-aligned Stage-1, fixed metric."
  c=$?; echo "=== exit code=$c attempt=$a $(date -u +%H:%M:%SZ) ==="; [ "$c" -eq 0 ] && break; sleep 15
done
SH
chmod +x logs/run_stage2_instruct.sh
```

- [ ] **Step 2: Launch detached after Stage-1 finishes (PPID=1)**

```bash
cd /home/ubuntu/lagunaxs2-rlvr-stage0
ts=$(date -u +%Y%m%dT%H%M%SZ); log="logs/stage2instruct_${ts}.log"
setsid nohup bash logs/run_stage2_instruct.sh </dev/null >>"$log" 2>&1 & disown
sleep 25; ps -eo pid,ppid,cmd | grep run_stage2_instruct | grep -v grep | head -1
```
Expected: PPID `1`; `grep -aE "guardrails|step 1/" "$log"` shows training.

- [ ] **Step 3: Read the per-task eval when it lands**

```bash
log=$(ls -t logs/stage2instruct_*.log | head -1)
grep -aE "val [0-9].*qa_acc" "$log" | sed -E 's/  +/ /g; s/wer.*embed_norm [0-9.]+ //'
grep -aoE "best qa [0-9.-]+" "$log" | sort -t' ' -k3 -n | tail -1
```
Expected (success): `synthetic` and the general-VQA corpora leave 0.00 and `best qa` rises clearly above the prior 0.044 ceiling — grounding transferred to reading. Design metrics (`code_valid`/CodeBLEU) appear in the W&B run.

- [ ] **Step 4: Commit a sweep writeup**

```bash
git add docs/experiments/  # writeup per the ML-sweep convention (or docs/specs/ note)
git commit -m "docs: bridge-fix data-recipe run results (stage1 caption + stage2 instruct)"
```

---

## If grounding still does not form (Task 4 Step 3 stays 0/6)

Captioning failed to ground → the deferred contrastive grounding loss becomes the next lever (separate spec). Do NOT keep tuning data. Record the negative in memory and stop.

## Self-review

- **Spec coverage:** caption Stage-1 (Task 4) ✓; diverse VQA Stage-2 (Tasks 1-3, 5) ✓; the_cauldron loaders via existing infra (Tasks 1-3) ✓; plain LM loss (no loss change) ✓; success = bridge diagnostics + per-task (Task 4 Step 3, Task 5 Step 3) ✓; tests (Tasks 1, 3) ✓. Caption-prompt change deferred with rationale (documented in File structure note) — intentional deviation.
- **Placeholder scan:** none — every step has exact code/commands.
- **Type consistency:** `parse_cauldron_vqa` (dict→dict|None) used by `_cauldron_vqa_rows`; `CauldronVQADataset.__getitem__`→(image,question,answer) matches the `vqa_sources` shape `QASFTDataset` consumes; `_resolve_vqa`/`CAULDRON_VQA` names match between corpora.py and the tests.
