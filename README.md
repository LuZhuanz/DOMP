# MJGPT Converter v0

This project converts gzip-compressed `.mjson` riichi mahjong logs from `data-draft/`
into readable GPT-style decision samples.

## Commands

```bash
uv run --with-editable . mjgpt-convert inspect data-draft
uv run --with-editable . mjgpt-convert validate data-draft --report out/v0/report.json
uv run --with-editable . mjgpt-convert convert data-draft \
  --out out/v0/decisions.long.jsonl \
  --report out/v0/report.json
uv run --with-editable . python -m unittest discover -s tests
```

Nested dataset directories are supported. Use gzip output for large corpora:

```bash
uv run --with-editable . mjgpt-convert validate dataset \
  --limit-files 10 \
  --report out/dataset-trial/report-10.json
uv run --with-editable . mjgpt-convert convert dataset \
  --limit-files 10 \
  --out out/dataset-trial/decisions-10.long.jsonl.gz \
  --report out/dataset-trial/report-convert-10.json
```

For full datasets, prefer year shards to avoid one huge output file:

```bash
uv run --with-editable . mjgpt-convert convert dataset/2018 \
  --out out/dataset/2018.long.jsonl.gz \
  --report out/dataset/2018.report.json
```

The JSONL output contains one record per decision with:

- `state_text`: long readable encoding ending in `<CHOICE>`, `<EXECUTE>`.
- `legal_actions`: numbered local action table.
- `choice_id`: chosen local action id.
- `execute`: executed action text.
- `validation_flags`: non-empty if the real action had to be appended for audit.

## Architecture

- [Mahjong GPT-like Policy Network 架构设计](docs/model_architecture.md)
- [训练程序实现设计](docs/training_program_design.md)

## Training

### 1. Smoke Test (Quick Validation)

Start with the lightest configuration to verify the training pipeline.

**Option A: Offline JSONL (ready-to-use data)**

`out/v0/decisions.long.jsonl` has already been converted from `data-draft`:

```bash
# ① Build vocab
uv run --with-editable . mjgpt-train build-vocab out/v0/decisions.long.jsonl \
  --data-format jsonl \
  --out out/train/smoke/vocab.json

# ② Train (debug model, 20 steps)
uv run --with-editable . mjgpt-train train out/v0/decisions.long.jsonl \
  --data-format jsonl \
  --vocab out/train/smoke/vocab.json \
  --output-dir out/train/smoke \
  --model-size debug \
  --batch-size 4 \
  --max-steps 20 \
  --device auto
```

**Option B: Streaming mjson (no intermediate JSONL)**

Read `.mjson` files directly in streaming mode:

```bash
# ① Build vocab
uv run --with-editable . mjgpt-train build-vocab data-draft \
  --data-format mjson \
  --out out/train/smoke/vocab.json

# ② Train
uv run --with-editable . mjgpt-train train data-draft \
  --data-format mjson \
  --vocab out/train/smoke/vocab.json \
  --output-dir out/train/smoke \
  --model-size debug \
  --batch-size 4 \
  --max-steps 20 \
  --device auto
```

If both commands finish successfully and print loss values, the training pipeline is fully functional.

### 2. Formal Training (tiny model example)

For a more serious run, switch to `tiny` or `small` and increase steps:

```bash
uv run --with-editable . mjgpt-train train out/v0/decisions.long.jsonl \
  --data-format jsonl \
  --vocab out/train/smoke/vocab.json \
  --output-dir out/train/tiny-run1 \
  --model-size tiny \
  --batch-size 8 \
  --max-steps 5000 \
  --lr 3e-4 \
  --device auto
```

Supported sizes: `debug` → `tiny` → `small` → `base`. Use `tiny` if GPU memory is limited.

### 3. Training on Full `dataset/`

For year-sharded full data (e.g. `dataset/2018`), streaming mjson is recommended to avoid huge intermediate JSONL files:

```bash
# Build vocab from 2018 data
uv run --with-editable . mjgpt-train build-vocab dataset/2018 \
  --data-format mjson \
  --out out/train/2018/vocab.json

# Streaming training
uv run --with-editable . mjgpt-train train dataset/2018 \
  --data-format mjson \
  --vocab out/train/2018/vocab.json \
  --output-dir out/train/2018 \
  --model-size small \
  --batch-size 8 \
  --max-steps 100000 \
  --num-workers 4 \
  --device auto
```

### 4. Output

After training, `--output-dir` contains:

- `vocab.json` — vocabulary (explicitly passing `--vocab` is recommended)
- `checkpoint-last/` — latest checkpoint (`model.pt`, `optimizer.pt`, etc.)
- Training logs are printed to stdout; future versions may write `train.log.jsonl`

### 5. Quick Summary

| Stage | Command | Note |
|-------|---------|------|
| Data prep | Use existing `out/v0/decisions.long.jsonl` or raw `dataset/**/*.mjson` | JSONL mode requires `mjgpt-convert convert` first |
| Build vocab | `mjgpt-train build-vocab ... --out vocab.json` | Must run before training |
| Train | `mjgpt-train train ... --vocab vocab.json --output-dir ...` | Supports both `jsonl` and `mjson` |

PyTorch is pinned to `torch==2.7.1+cu126` through the PyTorch CUDA 12.6 index.

## v0 Notes

- The converter uses visible information only.
- Seat labels are relative to the acting player.
- Honor tiles are normalized to `1z..7z`; red fives are preserved.
- Agari legality is shape-based in v0; yaku and furiten are not fully enforced yet.
- The CLI processes files one by one and clears rule caches between files to keep memory bounded.
