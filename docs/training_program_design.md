# 训练程序实现设计

本文档描述下一阶段需要编写的程序。它不是模型理论文档，而是工程落地规格：哪些模块要新增、每个模块负责什么、对外接口是什么、如何测试，以及第一版实现顺序。

目标是把当前 `mjgpt_converter` 转换器扩展为一个可训练的 Mahjong GPT-like policy baseline，同时保持转换器、数据管线、模型和训练入口之间的边界清晰。

## 1. 包结构

建议新增一个独立训练包，避免把模型训练逻辑塞进转换器模块：

```text
src/
  mjgpt_converter/
    ... existing trusted converter v0 ...
  mjgpt_training/
    __init__.py
    tokenizer.py
    samples.py
    dataset.py
    collator.py
    model.py
    config.py
    metrics.py
    checkpoint.py
    train.py
    eval.py
    cli.py
```

命令行入口建议新增：

```text
mjgpt-train
```

转换器仍保留：

```text
mjgpt-convert
```

这样两个阶段可以独立测试，也方便以后替换编码器或模型。

## 2. 数据流

训练数据流分为两种模式。

### 2.1 离线 JSONL 模式

用于调试、回归测试和小规模实验：

```text
out/v0/decisions.long.jsonl[.gz]
  -> read records
  -> build training sample
  -> tokenize
  -> collate
  -> train/eval
```

优点：

- 可复现。
- 易检查单条样本。
- 不依赖转换器实时速度。

缺点：

- 全量数据会产生很大的中间文件。
- 如果编码格式变化，需要重新生成。

### 2.2 流式 mjson 模式

用于大规模训练：

```text
dataset/**/*.mjson[.gz]
  -> trusted converter
  -> decision record
  -> build training sample
  -> tokenize
  -> collate
  -> train/eval
```

优点：

- 不需要保存全量 JSONL。
- 内存随 batch 大小稳定。
- 可按年份、目录、文件分片训练。

缺点：

- 当前瓶颈可能是 Python 转换速度。
- 需要更严格的错误处理和进度报告。

第一版应同时支持两种模式：先用离线 JSONL 让模型和训练循环稳定，再用流式模式扩展到全量数据。

## 3. `samples.py`

职责：把转换器输出的 decision record 变成不会泄漏标签的训练样本。

### 3.1 数据结构

```python
@dataclass(frozen=True)
class PolicySample:
    input_text: str
    label: int
    legal_action_count: int
    decision_type: str
    source: str | None = None
```

字段含义：

- `input_text`：模型可见文本，必须到 `<CHOICE>` 结束。
- `label`：真实动作局部下标，对应原始 `choice_id`。
- `legal_action_count`：合法动作数量。
- `decision_type`：真实动作类型，用于分项指标。
- `source`：可选来源信息，例如文件名、局号、样本序号。

### 3.2 核心函数

```python
def build_policy_sample(record: Mapping[str, Any], *, source: str | None = None) -> PolicySample:
    ...
```

要求：

- 从 `state_text` 中移除 `<EXECUTE>` 及之后的文本。
- 保证 `input_text` 包含且只包含一个 `<CHOICE>`。
- 保证 `input_text` 包含完整 `<LEGAL_ACTIONS> ... </LEGAL_ACTIONS>`。
- 从 `legal_actions` 或文本中得到合法动作数量。
- 验证 `0 <= choice_id < legal_action_count`。
- 推断 `decision_type`，优先从 `execute` 或选中动作文本读取。

失败时抛出明确异常，例如：

```python
class SampleFormatError(ValueError):
    pass
```

### 3.3 测试点

- 输入含 `<EXECUTE>` 时能正确截断。
- 输入不含 `<CHOICE>` 时报错。
- `choice_id` 越界时报错。
- `legal_actions` 与文本动作数量不一致时报错或给出审计标记。
- 输出 `input_text` 不包含真实动作答案。

## 4. `tokenizer.py`

职责：实现长文本 word-level tokenizer 和词表管理。

### 4.1 数据结构

```python
@dataclass
class MahjongVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    choice_token: str = "<CHOICE>"
```

```python
@dataclass
class TokenizedSample:
    input_ids: list[int]
    choice_position: int
    action_positions: list[int]
    label: int
    decision_type: str
```

### 4.2 核心函数

```python
def tokenize_text(text: str) -> list[str]:
    ...

def build_vocab(samples: Iterable[PolicySample], *, min_freq: int = 1, max_size: int | None = None) -> MahjongVocab:
    ...

def encode_sample(sample: PolicySample, vocab: MahjongVocab) -> TokenizedSample:
    ...
```

### 4.3 分词规则

第一版使用简单、可审计的规则：

- 先按空白切分。
- 对动作边界、结构 token、牌 token 不做拆分。
- 不做大小写归一化。
- 不做数值 bucket。
- 未登录 token 映射为 `<UNK>`。

如果现有转换文本中有紧贴标点的 token，优先调整 converter 输出或样本构造，让关键结构 token 独立出现，而不是在 tokenizer 里写复杂规则。

### 4.4 位置提取

`encode_sample` 必须提取：

- `<CHOICE>` 的 token 下标。
- 所有 `</A\d+>` 的 token 下标。

动作结束 token 数量必须等于 `legal_action_count`。

### 4.5 持久化

词表保存为 JSON：

```json
{
  "version": 1,
  "token_to_id": {"<PAD>": 0, "<UNK>": 1},
  "special_tokens": {
    "pad": "<PAD>",
    "unk": "<UNK>",
    "bos": "<BOS>",
    "choice": "<CHOICE>"
  }
}
```

### 4.6 测试点

- 特殊 token id 稳定。
- encode/decode 基本可逆。
- `5mr` 等红五 token 不被拆分。
- `</A0>`、`</A12>` 可被识别为动作结束。
- 缺少 `<CHOICE>` 或动作结束 token 时明确报错。

## 5. `dataset.py`

职责：提供训练样本迭代器。

### 5.1 离线 Dataset

```python
class JsonlPolicyDataset(IterableDataset):
    def __init__(self, paths: Sequence[Path], *, vocab: MahjongVocab, strict: bool = True):
        ...
```

行为：

- 支持 `.jsonl` 和 `.jsonl.gz`。
- 一行一个 JSON record。
- 逐行读取，逐条构造 `PolicySample`，再编码为 `TokenizedSample`。
- `strict=True` 时遇到坏样本立即报错。
- `strict=False` 时跳过坏样本，并累计错误计数。

### 5.2 流式 Dataset

```python
class MjsonStreamingPolicyDataset(IterableDataset):
    def __init__(
        self,
        roots: Sequence[Path],
        *,
        vocab: MahjongVocab,
        strict: bool = True,
        shuffle_files: bool = False,
        shuffle_buffer_size: int = 0,
    ):
        ...
```

行为：

- 递归读取 `.mjson` / `.mjson.gz`。
- 复用 `mjgpt_converter` 的文件迭代和转换逻辑。
- DataLoader 多 worker 时按 worker id 对文件列表分片。
- 可选文件级 shuffle。
- 可选有限 shuffle buffer。
- 每个样本输出 `TokenizedSample`。

### 5.3 文件分片

多 worker 下推荐按文件分片：

```text
worker_files = all_files[worker_id::num_workers]
```

这样同一个 mjson 文件只会被一个 worker 读取，避免重复样本。

### 5.4 测试点

- JSONL gzip/plain 都能读。
- 不存在路径时报错。
- 多 worker 分片不重复、不漏文件。
- `strict=False` 能跳过坏样本。
- shuffle buffer 不改变样本字段合法性。

## 6. `collator.py`

职责：把变长 tokenized samples 组装成 batch tensor。

### 6.1 数据结构

```python
@dataclass
class PolicyBatch:
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    choice_positions: torch.LongTensor
    action_positions: torch.LongTensor
    action_mask: torch.BoolTensor
    labels: torch.LongTensor
    decision_types: list[str]
```

### 6.2 核心类

```python
class PolicyCollator:
    def __init__(self, *, pad_id: int, max_length: int = 512, truncation: str = "left_history"):
        ...

    def __call__(self, samples: Sequence[TokenizedSample]) -> PolicyBatch:
        ...
```

### 6.3 截断策略

第一版可先实现保守策略：

- 如果 `len(input_ids) <= max_length`，正常 padding。
- 如果超过 `max_length`，先报错或跳过。

在确认真实超长比例后，再实现结构化截断。结构化截断必须满足：

- 不截断 `<LEGAL_ACTIONS>`。
- 不截断 `<CHOICE>`。
- 不破坏动作边界。
- `choice_positions` 和 `action_positions` 随截断偏移修正。

### 6.4 测试点

- 不同长度样本可正确 padding。
- 不同合法动作数可正确 padding。
- padding action 的 `action_mask` 为 false。
- label 必须落在有效 action 范围。
- 超长样本行为符合配置。

## 7. `model.py`

职责：实现 GPT-like policy network。

### 7.1 配置

```python
@dataclass
class ModelConfig:
    vocab_size: int
    max_position_embeddings: int = 512
    n_layers: int = 6
    n_heads: int = 6
    hidden_size: int = 384
    intermediate_size: int = 1536
    dropout: float = 0.1
    rope_base: float = 10000.0
    scorer_hidden_size: int = 384
```

### 7.2 模块

建议拆分：

```text
RMSNorm
RotaryEmbedding
CausalSelfAttention
SwiGLU
TransformerBlock
MahjongPolicyModel
```

### 7.3 Forward 接口

```python
class MahjongPolicyModel(nn.Module):
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        choice_positions: torch.LongTensor,
        action_positions: torch.LongTensor,
        action_mask: torch.BoolTensor,
        labels: torch.LongTensor | None = None,
    ) -> PolicyModelOutput:
        ...
```

返回：

```python
@dataclass
class PolicyModelOutput:
    logits: torch.FloatTensor
    loss: torch.FloatTensor | None = None
```

### 7.4 动作打分

实现动态动作 scorer：

```text
h_state = gather(hidden, choice_positions)
h_actions = gather(hidden, action_positions)
features = concat(h_state, h_actions, h_state * h_actions)
logits = scorer(features)
logits = logits.masked_fill(~action_mask, -inf)
```

`action_positions` 中 padding 位置可以填 0，但必须用 `action_mask` 屏蔽。

### 7.5 测试点

- synthetic batch forward 输出 `[B, Amax]`。
- padding action logit 被 mask。
- `labels=None` 时不计算 loss。
- `labels` 有效时 loss 是标量。
- 单 batch 训练数十步 loss 能下降。

## 8. `metrics.py`

职责：训练和验证指标计算。

### 8.1 指标

```python
class PolicyMetrics:
    def update(self, logits, labels, action_mask, decision_types):
        ...

    def compute(self) -> dict[str, float]:
        ...
```

第一版指标：

- `loss`
- `accuracy_top1`
- `accuracy_top3`
- `accuracy_by_decision_type/*`
- `accuracy_by_action_count/*`
- `invalid_action_rate`

`invalid_action_rate` 理论上应为 0，因为 logits 已 mask；如果非 0，说明 mask 或推理代码有 bug。

### 8.2 测试点

- top1/top3 计算正确。
- padding action 不会被计入预测。
- 分 action type 指标正确聚合。
- 空指标状态返回明确结果或报错。

## 9. `checkpoint.py`

职责：保存和加载训练状态。

### 9.1 保存内容

checkpoint 目录建议包含：

```text
checkpoint_dir/
  model.pt
  optimizer.pt
  scheduler.pt
  trainer_state.json
  model_config.json
  vocab.json
```

`trainer_state.json` 至少包含：

- `step`
- `epoch`
- `best_metric`
- `random_seed`
- `git_commit`，如果可用。
- `created_at`

### 9.2 测试点

- 保存后可恢复模型输出。
- 恢复 optimizer 后继续训练不报错。
- vocab/config 与 checkpoint 一起保存。

## 10. `train.py`

职责：单机监督训练循环。

### 10.1 输入参数

训练入口需要支持：

```text
--data PATH [PATH ...]
--data-format jsonl|mjson
--vocab PATH
--build-vocab
--output-dir PATH
--model-size tiny|small|base
--max-length 512
--batch-size 8
--grad-accum-steps 1
--lr 3e-4
--weight-decay 0.1
--max-steps N
--eval-every N
--save-every N
--num-workers N
--device auto|cpu|cuda
--amp bf16|fp16|none
--seed 42
```

### 10.2 训练流程

```text
parse args
  -> set seed
  -> load/build vocab
  -> build dataset
  -> build dataloader
  -> build model
  -> build optimizer/scheduler
  -> train loop
  -> periodic eval
  -> checkpoint
```

### 10.3 第一版优化器

建议：

- `AdamW`
- `lr=3e-4` for tiny/small 起步
- `weight_decay=0.1`
- cosine schedule with warmup
- gradient clipping `1.0`
- mixed precision 在 CUDA 上默认 `bf16`，如果设备不支持则回退 `fp16` 或 `none`

### 10.4 测试点

- `--max-steps 2` 冒烟训练。
- CPU 下可运行 tiny 模型。
- CUDA 不可用时 `--device auto` 自动回退 CPU。
- checkpoint 文件写入完整。
- 固定 seed 下小样本指标基本可复现。

## 11. `eval.py`

职责：独立评估 checkpoint。

### 11.1 输入参数

```text
--checkpoint PATH
--data PATH [PATH ...]
--data-format jsonl|mjson
--batch-size N
--max-batches N
--device auto|cpu|cuda
--report PATH
```

### 11.2 输出

报告保存为 JSON：

```json
{
  "samples": 10000,
  "loss": 1.23,
  "accuracy_top1": 0.42,
  "accuracy_top3": 0.71,
  "accuracy_by_decision_type": {},
  "accuracy_by_action_count": {},
  "invalid_action_rate": 0.0
}
```

### 11.3 测试点

- 能加载 checkpoint。
- 能输出 JSON report。
- `--max-batches` 生效。

## 12. `cli.py`

职责：提供统一命令行。

建议命令：

```text
mjgpt-train build-vocab DATA --out out/vocab.json
mjgpt-train train --data out/v0/decisions.long.jsonl --data-format jsonl --output-dir out/train/tiny
mjgpt-train train --data dataset/2018 --data-format mjson --output-dir out/train/2018-small
mjgpt-train eval --checkpoint out/train/tiny/checkpoint-last --data out/v0/decisions.long.jsonl
```

第一版可以先用子命令：

- `build-vocab`
- `train`
- `eval`

## 13. 配置文件

除 CLI 参数外，建议支持 JSON 配置：

```json
{
  "data": ["dataset/2018"],
  "data_format": "mjson",
  "output_dir": "out/train/2018-small",
  "model": {
    "preset": "small",
    "max_length": 512
  },
  "training": {
    "batch_size": 8,
    "grad_accum_steps": 4,
    "lr": 0.0003,
    "max_steps": 100000,
    "num_workers": 4,
    "amp": "bf16"
  }
}
```

配置文件适合长训练；CLI 参数适合快速覆盖。

## 14. 依赖

现有项目只有标准库依赖。训练阶段需要新增：

- `torch`

可选依赖：

- `tqdm`：进度条。
- `numpy`：指标和小工具。

第一版可以只强依赖 `torch`，用标准库打印训练日志，减少环境问题。

## 15. 目录输出约定

训练输出建议：

```text
out/train/run-name/
  config.json
  vocab.json
  train.log.jsonl
  eval.report.json
  checkpoint-last/
  checkpoint-best/
```

`train.log.jsonl` 一行一个事件：

```json
{"step": 100, "loss": 1.73, "lr": 0.00029, "tokens_per_sec": 12000}
{"step": 1000, "eval_accuracy_top1": 0.38, "eval_loss": 1.42}
```

## 16. 推荐实现顺序

### Step 1：训练样本和 tokenizer

实现：

- `samples.py`
- `tokenizer.py`
- 对应单元测试

验收：

- 可从现有 JSONL 读取样本。
- 构造出的模型输入无 `<EXECUTE>`。
- vocab 可保存和加载。

### Step 2：collator

实现：

- `collator.py`
- `PolicyBatch`

验收：

- 可组装变长文本和变长动作数。
- mask、label、position 全部正确。

### Step 3：tiny 模型

实现：

- `model.py`
- `ModelConfig`

验收：

- synthetic batch forward/backward。
- 单 batch loss 可下降。

### Step 4：离线训练 CLI

实现：

- JSONL dataset
- `train.py`
- `cli.py build-vocab`
- `cli.py train`

验收：

- `data-draft` 转出的 JSONL 可训练。
- `--max-steps 10` 正常结束并写 checkpoint。

### Step 5：评估 CLI

实现：

- `metrics.py`
- `eval.py`
- `cli.py eval`

验收：

- checkpoint 可评估。
- report JSON 完整。

### Step 6：流式 mjson 训练

实现：

- `MjsonStreamingPolicyDataset`
- worker 分片
- shuffle buffer

验收：

- `dataset/2018 --data-format mjson` 可启动训练。
- 内存稳定。
- 转换错误有清晰统计。

## 17. 第一批测试文件建议

建议新增：

```text
tests/test_training_samples.py
tests/test_training_tokenizer.py
tests/test_training_collator.py
tests/test_training_model.py
tests/test_training_metrics.py
tests/test_training_checkpoint.py
tests/test_training_cli.py
```

保持测试输入小而固定，优先用手写最小样本，不要让单元测试依赖全量 dataset。

## 18. 当前不做的工程项

以下内容暂缓，避免拖慢第一版闭环：

- 分布式训练。
- FSDP/DeepSpeed。
- Hugging Face Trainer 适配。
- 复杂 tokenizer 训练。
- 大规模特征缓存。
- 推理服务 API。
- Web 可视化。

先完成可复现的单机监督训练，再决定是否引入这些工程层。
