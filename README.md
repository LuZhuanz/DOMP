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

## Training Prototype

Build a word-level vocab from streaming mjson:

```bash
uv run --with-editable . mjgpt-train build-vocab data-draft \
  --data-format mjson \
  --out out/train/smoke/vocab.json
```

Run a minimal streaming training smoke:

```bash
uv run --with-editable . mjgpt-train train data-draft \
  --data-format mjson \
  --vocab out/train/smoke/vocab.json \
  --output-dir out/train/smoke \
  --model-size debug \
  --batch-size 4 \
  --max-steps 20 \
  --device auto
```

PyTorch is pinned to `torch==2.7.1+cu126` through the PyTorch CUDA 12.6 index.

## v0 Notes

- The converter uses visible information only.
- Seat labels are relative to the acting player.
- Honor tiles are normalized to `1z..7z`; red fives are preserved.
- Agari legality is shape-based in v0; yaku and furiten are not fully enforced yet.
- The CLI processes files one by one and clears rule caches between files to keep memory bounded.
