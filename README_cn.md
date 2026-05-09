# MJGPT Converter v0

本项目将 `data-draft/` 中的 gzip 压缩 `.mjson` 立直麻将日志转换为可读的 GPT 风格决策样本。

## 命令

```bash
uv run --with-editable . mjgpt-convert inspect data-draft
uv run --with-editable . mjgpt-convert validate data-draft --report out/v0/report.json
uv run --with-editable . mjgpt-convert convert data-draft \
  --out out/v0/decisions.long.jsonl \
  --report out/v0/report.json
uv run --with-editable . python -m unittest discover -s tests
```

支持嵌套数据集目录。大型语料库建议使用 gzip 输出：

```bash
uv run --with-editable . mjgpt-convert validate dataset \
  --limit-files 10 \
  --report out/dataset-trial/report-10.json
uv run --with-editable . mjgpt-convert convert dataset \
  --limit-files 10 \
  --out out/dataset-trial/decisions-10.long.jsonl.gz \
  --report out/dataset-trial/report-convert-10.json
```

处理全量数据时，建议按年份分片以避免生成单个巨大文件：

```bash
uv run --with-editable . mjgpt-convert convert dataset/2018 \
  --out out/dataset/2018.long.jsonl.gz \
  --report out/dataset/2018.report.json
```

JSONL 输出每条记录包含：

- `state_text`：以 `<CHOICE>`、`<EXECUTE>` 结尾的长可读编码。
- `legal_actions`：编号后的局部动作表。
- `choice_id`：选中的局部动作 id。
- `execute`：实际执行的动作文本。
- `validation_flags`：如果真实动作被追加用于审计，则非空。

## 架构

- [Mahjong GPT-like Policy Network 架构设计](docs/model_architecture.md)
- [训练程序实现设计](docs/training_program_design.md)

## 训练

### 1. Smoke Test（快速验证）

建议先用最轻量的配置跑几十步，确认训练链路能走通。

**方式 A：离线 JSONL（已有现成数据）**

仓库里 `out/v0/decisions.long.jsonl` 已经由 `data-draft` 转换好了，可以直接用：

```bash
# ① 构建词表
uv run --with-editable . mjgpt-train build-vocab out/v0/decisions.long.jsonl \
  --data-format jsonl \
  --out out/train/smoke/vocab.json

# ② 训练（debug 模型，20 步）
uv run --with-editable . mjgpt-train train out/v0/decisions.long.jsonl \
  --data-format jsonl \
  --vocab out/train/smoke/vocab.json \
  --output-dir out/train/smoke \
  --model-size debug \
  --batch-size 4 \
  --max-steps 20 \
  --device auto
```

**方式 B：流式 mjson（不生成中间 JSONL）**

直接从原始 `.mjson` 文件流式读取：

```bash
# ① 构建词表
uv run --with-editable . mjgpt-train build-vocab data-draft \
  --data-format mjson \
  --out out/train/smoke/vocab.json

# ② 训练
uv run --with-editable . mjgpt-train train data-draft \
  --data-format mjson \
  --vocab out/train/smoke/vocab.json \
  --output-dir out/train/smoke \
  --model-size debug \
  --batch-size 4 \
  --max-steps 20 \
  --device auto
```

如果上面两条命令都能正常结束并输出 loss，说明训练链路完全可用。

### 2. 正式训练（以 tiny 模型为例）

如果你想跑一个稍正式的训练，可以换成 `tiny` 或 `small` 模型，增加步数：

```bash
# 以离线 JSONL 为例
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

支持的模型规模：`debug` → `tiny` → `small` → `base`。显存有限时建议先用 `tiny`。

### 3. 使用全量 `dataset/` 数据

如果你要训练 `dataset/` 目录下按年份存放的全量数据（如 `dataset/2018`），推荐**流式 mjson** 模式，不需要预先转换出巨大的 JSONL：

```bash
# 从 2018 年数据构建词表
uv run --with-editable . mjgpt-train build-vocab dataset/2018 \
  --data-format mjson \
  --out out/train/2018/vocab.json

# 流式训练
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

### 4. 输出说明

训练完成后，`--output-dir` 下会生成：

- `vocab.json` — 词表（建议显式指定）
- `checkpoint-last/` — 最新 checkpoint（包含 `model.pt`、`optimizer.pt` 等）
- 训练日志会打印在终端，后续版本可能会写入 `train.log.jsonl`

### 5. 简要总结

| 阶段 | 命令 | 说明 |
|------|------|------|
| 数据准备 | 已有 `out/v0/decisions.long.jsonl` 或直接用 `dataset/**/*.mjson` | JSONL 模式需先 `mjgpt-convert convert` |
| 构建词表 | `mjgpt-train build-vocab ... --out vocab.json` | 必须先执行 |
| 训练 | `mjgpt-train train ... --vocab vocab.json --output-dir ...` | 支持 `jsonl` / `mjson` 两种格式 |

PyTorch 固定为 `torch==2.7.1+cu126`，通过 PyTorch CUDA 12.6 索引安装。

## v0 说明

- 转换器仅使用可见信息。
- 座位标签相对于当前行动玩家。
- 字牌归一化为 `1z..7z`；红宝牌保留。
- v0 中荣和合法性基于牌型判断；役种和振听尚未完全强制执行。
- CLI 逐个处理文件，并在文件之间清除规则缓存，以保持内存占用稳定。
