# Mahjong GPT-like Policy Network 架构设计

本文档定义麻将 AI 第一版神经网络的目标、输入边界、模型结构、动作打分方式和训练管线。它承接当前 `mjgpt-convert` v0 可信转换器的输出，目标是先验证“纯状态编码 + 合法动作集合”是否足以训练出稳定的监督策略模型。

## 1. 目标与范围

v1 的核心目标是训练一个 GPT-like 的监督策略网络：

- 输入：某一决策点之前的可见状态文本，以及该决策点的合法动作列表。
- 输出：在当前合法动作集合上的概率分布。
- 训练目标：行为克隆，预测日志中真实执行的合法动作。
- 强约束：模型只能在转换器给出的合法动作集合内选择动作，因此推理阶段的非法动作率应为 0。

v1 暂不做：

- 强化学习、自博弈或价值网络。
- 完整役种、振听、点数期望建模。
- 多模态或手写牌谱解析。
- 固定全局动作编号体系。

这些内容可以在监督策略基线稳定后再扩展。

## 2. 数据边界

当前转换器输出的单条样本包含：

- `state_text`：长文本状态编码。
- `legal_actions`：局部合法动作表。
- `choice_id`：真实动作在局部动作表中的编号。
- `execute`：真实执行动作文本。
- `validation_flags`：转换审计标记。

模型输入必须避免标签泄漏。训练时输入应只包含：

```text
<BOS>
... state prefix ...
<LEGAL_ACTIONS>
<A0> ... </A0>
<A1> ... </A1>
...
</LEGAL_ACTIONS>
<CHOICE>
```

不得把以下字段放入模型输入：

- 真实 `choice_id`。
- `<EXECUTE>` 及其后续内容。
- 任何只由真实动作推导出来、但在决策前不可见的信息。

监督标签为 `choice_id`，即当前样本合法动作列表中的局部下标。

## 3. 文本与词表

第一版使用长文本词表，保持转换器 v0 的可读编码，避免过早压缩状态信息。

### 3.1 分词策略

采用 whitespace/word-level tokenizer：

- 按空白切分主 token。
- 保留结构 token，例如 `<BOS>`、`<LEGAL_ACTIONS>`、`</LEGAL_ACTIONS>`、`<CHOICE>`。
- 保留动作边界 token，例如 `<A0>`、`</A0>`、`<A1>`、`</A1>`。
- 保留牌 token，例如 `1m`、`5mr`、`7z`。
- 保留事件、区域、相对座位等文本 token。

数值 token 第一版保留原文本，例如局数、本场、立直棒、点数、巡目等，不做 bucket。这样更容易和原始编码对齐，也便于排查转换错误。

### 3.2 特殊 token

最小特殊 token 集合：

- `<PAD>`：batch padding。
- `<UNK>`：词表外 token。
- `<BOS>`：样本开始。
- `<CHOICE>`：策略查询位置。

动作边界 token 可以作为普通词表 token 保留，但 tokenizer/collator 需要能识别 `</A*>` 的位置。

### 3.3 词表构建

词表从训练集流式扫描得到：

1. 按文件顺序读取 mjson。
2. 转换为决策样本。
3. 截断掉标签泄漏部分，仅扫描模型输入文本。
4. 统计 token 频次。
5. 按频次和保留 token 生成 vocab。

v1 不需要子词 tokenizer。只有当长文本词表出现严重稀疏、上下文长度压力或泛化问题时，再考虑 BPE/Unigram 或领域专用短 token。

## 4. 输入格式

每条样本在进入模型前整理为：

```text
<BOS>
<ROUND> ...
<SELF> ...
<PLAYERS> ...
<DORA> ...
<HISTORY> ...
<LEGAL_ACTIONS>
<A0> TYPE=DAHAI TILE=3m ... </A0>
<A1> TYPE=RIICHI TILE=3m ... </A1>
</LEGAL_ACTIONS>
<CHOICE>
```

其中 `<CHOICE>` 是 state/query 的读取位置。模型不做 next-token generation，而是在 `<CHOICE>` 对当前合法动作集合进行打分。

上下文长度初始设置为 512。如果真实样本超过上限，优先保留：

1. 当前手牌、副露、河牌、宝牌、分数、相对座位等状态。
2. 当前合法动作列表。
3. 最近历史事件。

不能截断合法动作列表，也不能截断 `<CHOICE>`。

## 5. 模型总览

模型采用 decoder-only Transformer：

```text
token ids
  -> token embedding
  -> RoPE positional encoding
  -> N x Transformer decoder block
       - RMSNorm
       - causal self-attention
       - RMSNorm
       - SwiGLU feed-forward
  -> final RMSNorm
  -> dynamic action scorer
  -> logits over legal actions
```

注意力使用 causal mask。即使当前任务不是语言模型生成，causal mask 仍然有两个好处：

- 与 GPT-like 结构一致，后续可以扩展为状态续写、辅助语言建模或统一序列建模。
- 保证 `<CHOICE>` 位置只能看见决策前输入，不能读取未来标签。

## 6. 动态动作打分

麻将决策的合法动作数是变长的，不适合第一版就强行映射成固定全局动作编号。因此 v1 使用动态动作打分。

### 6.1 位置定义

对每条样本提取：

- `choice_position`：`<CHOICE>` token 的位置。
- `action_positions[i]`：第 `i` 个合法动作结束 token `</Ai>` 的位置。
- `label`：真实动作的局部动作下标，即 `choice_id`。

动作表示使用动作结束位置的 hidden state，因为该位置已经读完该动作文本。

### 6.2 打分公式

设：

- `h_state = H[choice_position]`
- `h_action_i = H[action_positions[i]]`

每个动作的 logit 可用共享 scorer 计算：

```text
features_i = concat(h_state, h_action_i, h_state * h_action_i)
logit_i = MLP(features_i)
```

MLP 对所有动作共享参数。这样模型可以处理任意数量的合法动作，只要 batch 内 padding 后加 mask 即可。

也可以在后续实验中替换为 bilinear scorer：

```text
logit_i = h_state^T W h_action_i
```

第一版建议使用 MLP scorer，因为实现简单，表达力足够，便于排查。

### 6.3 Masked Classification

一个 batch 内合法动作数量不同，需要 padding 到 `Amax`：

```text
logits:      [B, Amax]
action_mask: [B, Amax]
labels:      [B]
```

padding 动作的 logit 在 softmax 前置为极小值：

```text
masked_logits = logits.masked_fill(~action_mask, -inf)
loss = cross_entropy(masked_logits, labels)
```

只要 `labels[b]` 总是落在 `action_mask[b] == true` 的位置，训练目标就是标准分类问题。

## 7. Batch Schema

模型 forward 推荐接收：

```text
input_ids:        LongTensor[B, T]
attention_mask:   BoolTensor[B, T]
choice_positions: LongTensor[B]
action_positions: LongTensor[B, Amax]
action_mask:      BoolTensor[B, Amax]
labels:           LongTensor[B] | None
```

返回：

```text
logits: FloatTensor[B, Amax]
loss:   FloatTensor[] | None
```

collator 负责：

- 文本 tokenization。
- padding/truncation。
- 提取 `<CHOICE>` 位置。
- 提取每个 `</Ai>` 位置。
- 对动作位置和动作 mask 做 padding。
- 验证 label 没有越界。

## 8. 推荐模型规模

第一阶段目标是证明训练闭环和动作打分有效，不追求一次到位的大模型。

| 配置 | Layers | Hidden | Heads | FFN Mult | Context | 预期用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tiny | 4 | 256 | 4 | 4x | 512 | 单机冒烟、单元测试、快速过拟合 |
| small | 6 | 384 | 6 | 4x | 512 | 第一版主力基线 |
| base | 12 | 768 | 12 | 4x | 512/768 | 数据管线稳定后的扩展 |

当前 RTX 4060 8GB 更适合从 `tiny` 或 `small` 起步。`base` 需要更谨慎地使用 mixed precision、gradient accumulation 和较小 batch。

## 9. 流式训练管线

不建议先把全量 dataset 转成一个巨大的 JSONL 再训练。更合适的是边转换边训练：

```text
mjson file iterator
  -> parse one game
  -> state replay
  -> legal action generation
  -> decision sample
  -> leakage-safe model input
  -> tokenizer
  -> collator
  -> model
```

### 9.1 IterableDataset

实现 `MahjongIterableDataset`：

- 输入一个或多个目录。
- 递归枚举 `.mjson` / `.mjson.gz` 文件。
- 按 DataLoader worker 做文件分片。
- 每次只打开一个文件并产出其中的决策样本。
- 每局结束后释放状态与规则缓存。
- 遇到转换错误时记录文件名和异常，按配置选择跳过或中止。

这样可以把内存占用控制在“当前文件 + 当前 batch”级别。

### 9.2 Shuffle 策略

流式数据不能像内存数据集一样全局 shuffle。v1 可采用：

- 训练文件列表按 epoch 打乱。
- worker 内维护一个有限大小 shuffle buffer。
- buffer 满后随机弹出样本。

这能减少同一局内相邻决策高度相关的问题。

### 9.3 性能预期

已有 50 文件转换基准约为：

- 31,050 条决策。
- 58.14 秒。
- 峰值内存约 1.0GB。

粗略估计当前 Python 纯转换速度约 500 条决策/秒。流式训练第一版通常会先被转换器限制，而不是 GPU 限制。后续优化方向：

- 减少状态文本重复构造。
- 引入短 token 编码。
- 缓存 tokenizer 结果。
- 多 worker 并行转换。
- 将合法动作生成中的重复计算下沉为增量状态。

## 10. Loss 与指标

训练 loss：

```text
CE(masked_logits, choice_id)
```

核心指标：

- `loss`：整体交叉熵。
- `top1_accuracy`：真实动作是否为最高分。
- `top3_accuracy`：真实动作是否在前三。
- `accuracy_by_decision_type`：按 `DAHAI`、`CHI`、`PON`、`KAN`、`RIICHI`、`RON`、`TSUMO` 等分类。
- `accuracy_by_action_count`：按合法动作数量分桶。
- `invalid_action_rate`：推理阶段应为 0，因为只在合法动作集合内 argmax。

审计指标：

- `validation_flags` 非空样本比例。
- 转换错误文件数。
- 超长样本截断比例。
- `<CHOICE>` 或 `</Ai>` 位置解析失败数。

## 11. 实现里程碑

### M1：离线可复现小闭环

- 实现 tokenizer 和 vocab builder。
- 实现 leakage-safe 输入构造。
- 实现 collator。
- 用 `data-draft` 跑通单元测试和小规模过拟合。

验收标准：

- 一个小 batch 能 forward/backward。
- 单局或少量样本可以过拟合到高准确率。
- 标签泄漏测试通过。

### M2：动态动作 scorer

- 实现 decoder-only Transformer。
- 实现 `choice_positions` 与 `action_positions` gather。
- 实现 masked cross entropy。
- 增加动作数量变长的 batch 测试。

验收标准：

- padding 动作不参与 loss。
- label 越界会明确报错。
- synthetic batch 的 loss 能下降。

### M3：流式训练

- 实现 `MahjongIterableDataset`。
- 支持 dataset 年份分片。
- 支持 shuffle buffer。
- 支持训练日志和 checkpoint。

验收标准：

- 可在 `dataset/2018` 上持续训练。
- 内存不随已读文件数量增长。
- 训练中断后可从 checkpoint 恢复。

### M4：全量基线

- 在全量 dataset 上训练 `small`。
- 输出验证集指标。
- 保存 vocab、config、checkpoint、训练报告。

验收标准：

- `top1_accuracy` 显著高于简单频率基线。
- `invalid_action_rate == 0`。
- 各主要动作类型均有分项指标。

## 12. 测试计划

需要新增的关键测试：

- tokenizer 保留特殊 token、牌 token 和动作边界 token。
- 输入构造不会包含 `<EXECUTE>` 或真实 `choice_id`。
- `</Ai>` 动作结束位置提取正确。
- collator 可处理不同合法动作数量的样本。
- padding 动作被 mask 后不影响 loss。
- label 越界、缺少 `<CHOICE>`、缺少动作边界时明确报错。
- tiny 模型能在 synthetic batch 上 forward/backward。
- 小样本训练 loss 能下降。

## 13. 主要风险

### 13.1 状态文本过长

长文本编码可读性强，但存在重复信息多、训练慢的问题。v1 先用它验证能力，后续可切换为短 token 编码或结构化 token。

### 13.2 转换器成为瓶颈

当前转换速度大约是每秒数百条决策。流式训练会简化存储，但不自动解决转换耗时。需要在训练代码中记录 dataloader 等待时间和 GPU 利用率。

### 13.3 合法动作生成仍是 v0 规则

v0 合法动作已能通过牌谱验证，但役种、振听等完整规则仍有边界。模型第一阶段只学习 v0 规则空间内的监督策略。

### 13.4 动作文本相似度

同一状态下多个打牌动作文本非常相似，模型必须依赖具体牌 token 和上下文差异。动态 scorer 对动作文本建模更自然，但需要确保动作边界和位置提取严格正确。

## 14. 推荐下一步

下一步应实现训练侧最小闭环：

1. `tokenizer.py`：词表构建、encode/decode、特殊 token 识别。
2. `dataset.py`：从转换器流式产出 leakage-safe 样本。
3. `collator.py`：padding、动作位置、mask、label。
4. `model.py`：tiny decoder-only Transformer + dynamic action scorer。
5. `train.py`：单机监督训练入口。

优先完成 `tiny` 配置在 `data-draft` 上的过拟合测试，再扩展到 `dataset/2018`。

## 15. v2 规划：结构化嵌入 (Structured Embeddings)

> **定位**：v2 值得做，且当前代码基础可行；但不进入当前 v1 最小训练闭环。
> 更合理的位置是——先把 v1 文本版跑出可复现 baseline，然后把结构化嵌入作为 v2 的第一优先级升级。
> 本节是 v2 的规划设计，也是后续实施的对照文档。

### 15.1 动机：v1 文本编码的瓶颈

基于 `out/v0/decisions.long.jsonl` 的 46,072 条样本统计：

| 指标 | 数值 |
|------|------|
| 平均输入长度 | ~195 tokens |
| p90 | ~250 tokens |
| p99 | ~294 tokens |
| 最大输入长度 | 390 tokens |
| data-draft 词表大小 | 718 token（其中纯数字 token 550） |
| 结构 token 占比 | ~57.8% |

问题不在于"序列撑满 512"（当前远未达到），而是：

1. **结构性标记浪费上下文**：`<ROUND>`, `</LEGAL_ACTIONS>` 等标记占文本的近 58%，不携带任何游戏信息却消耗了过半的注意力预算。
2. **数值 token 零泛化**：`25000` 和 `24900` 是两个独立 token，模型学到的是"这个字符串对应什么行为"而非"这个数值大小意味着什么"。
3. **大批量 token → 大 batch 内存**：虽然 data-draft 词表仅 718，但全量 dataset 的数值 token 会随着更多对局迅速膨胀到数千以上，稀疏的 embedding 参数是纯浪费。

全量 dataset 会进一步放大问题 2 和 3，但当前 data-draft 的数据已足以证伪文字编码的必要性。结合第 13.1 节的风险声明——"后续可切换为短 token 编码或结构化 token"——v2 的可行性条件已经具备。

### 15.2 核心思路

```
v1:  GameState → encode_state() → text str → tokenize() → ids → Embedding() → vectors
v2:  GameState → feature snapshot → FeatureEncoder() → vectors
```

从"状态 → 文本 → token ID → embedding 查表"三步损失式转换，切换为"状态 → 特征快照 → 专用编码器 → 直接产出 embedding 向量"。输入从 ~200 个语义稀疏的文本 token 压缩为 ~60–80 个信息密集的特征向量。

架构分类不变：仍然是 **decoder-only Transformer 策略网络**。改动仅触及输入层和样本格式，Transformer blocks、RoPE、SwiGLU FFN、动态动作 scorer 一行不碰。

### 15.3 特征快照 (Feature Snapshot)

**关键设计决策**：`input_features` 不能直接持有 `GameState` 对象。`GameState` 在流式转换过程中会继续被后续事件修改（mutable state 风险），且它包含内部缓存（如 agari LRU），不宜序列化。

应由 converter 在构建 `PolicySample` 时产出一个不可变、可 JSON 序列化的 primitive snapshot：

```python
# converter 产出时接在 legal_actions/choice_id 之后
feature_snapshot: dict = {
    "round_name": "E1-0",
    "honba": 0,
    "kyotaku": 0,
    "bakaze": "E",
    "oya": 0,
    "wall_left": 66,
    "dora_indicators": ["E"],
    "turn": 5,
    "actor": 0,
    "players": [
        {
            "wind": "E",
            "score": 25000,
            "riichi": False,
            "hand": ["1m", "2m", "3m", ..., "7z"],
            "draw": "5mr",
            "melds": [],
            "river": [
                {"tile": "4z", "riichi": False, "called": False},
                ...
            ],
        },
        # ... 其余三家，wind/score/riichi/melds/river 从 actor 视角使用相对关系表示
    ],
}
```

Feature snapshot 在文件中可以作为 `PolicySample` 的可选字段，训练时由 `FeatureEncoder` 消费；调试/审计时仍使用 `state_text`。两者共存，不互斥。

### 15.4 特征词汇表

不再使用 word-level tokenizer，改为对每类游戏信息做专用编码：

| 特征类型 | 取值空间 | 编码方式 | 说明 |
|----------|----------|----------|------|
| 物理牌 tile | 37 种（34 基牌 + 3 赤五） | `nn.Embedding(37, h)` | **需新增物理牌 index**：`tile_index()` 会把赤五 fallback 到基牌，不足以表示 5mr/5pr/5sr 的独立嵌入。新增 `tile_index_physical()` 返回 0–36。 |
| 风 wind | 4 种 | `nn.Embedding(4, h)` | 场风、自风、座位风 |
| 相对座位 seat_rel | 4 种 | `nn.Embedding(4, h)` | SELF/SHIMO/TOIMEN/KAMI |
| 分数 score | ℝ (归一化) | `MLP(1 → h → h)` | 输入 `(score - 25000) / 10000`，必要时 clip 到 `[-5, 15]` |
| 巡目/本场/供托/牌山剩余 | ℕ (小整数) | `nn.Embedding(64, h)` | 小范围整数，embedding 即可 |
| 宝牌指示牌 dora_indicator | 牌序列 | `tile_emb` 求和/池化 | 复用 tile embedding |
| 立直状态 riichi_flag | bool | `nn.Embedding(2, h)` | 是否已立直 |
| 副露 meld | 类型 + 来源牌 + 来源座 + 被鸣牌 | 结构化组合 | 每个副露编码为 1 个 dense 向量 |
| 河牌 river_tile | 牌 + riichi_flag + called_flag | `tile_emb + riichi_emb + called_emb` | 每张河牌一个向量 |
| 动作 action | kind + 目标 tile + 来源 + 来源座 | 结构化组合 | 每个动作编码为 k 个向量，见 15.7 |
| GLOBAL 标记 | — | `nn.Parameter` | 可学习的全局上下文 embedding |
| `<CHOICE>` 分隔符 | — | `nn.Parameter` | 可学习的分隔 embedding |
| 语义类型 type_emb | ≤20 种 | `nn.Embedding(20, h)` | 每行特征向量叠加 type_emb，标记"手牌/河牌/副露/..."语义角色 |

### 15.5 特征序列布局

序列压缩为 ~60–80 个信息密集的特征向量（每个代表一个游戏实体或一个动作元素），每行叠加 `type_emb[type_id]`：

```
序号  块名称                向量数  值来源
─────────────────────────────────────────────────
 0   [GLOBAL]              1       learnable param
 1   ROUND_NAME            1       str_emb (少量局名)
 2   HONBA                 1       int_emb
 3   KYOTAKU              1       int_emb
 4   BAKAZE                1       wind_emb
 5   WALL_LEFT             1       int_emb
 6   TURN                  1       int_emb
 7   DORA_INDICATORS       ≤5      tile_emb_phys
 8   WIND_SELF            1       wind_emb
 9   RIICHI_SELF           1       riichi_emb
10   SCORE_SELF           1       score_mlp
11   WIND_SHIMO           1       wind_emb
12   RIICHI_SHIMO         1       riichi_emb
13   SCORE_SHIMO          1       score_mlp
14   WIND_TOIMEN          1       wind_emb
15   RIICHI_TOIMEN         1       riichi_emb
16   SCORE_TOIMEN         1       score_mlp
17   WIND_KAMI            1       wind_emb
18   RIICHI_KAMI          1       riichi_emb
19   SCORE_KAMI           1       score_mlp
20   SELF_HAND             ≤14     tile_emb_phys
21   DRAW                  1       tile_emb_phys
22   MELD_0                k       meld 结构化编码
23   MELD_1                k       ...
...  (每个 MELD 块)
     RIVER_SELF_0          1       tile + riichi_flag + called_flag
     RIVER_SHIMO_0         1       ...
     ... (每张河牌，四家交错按巡序排列，最多 ~72 张)

     <CHOICE>              1       learnable param

     ACTION_0              k       action 结构化编码
     ACTION_1              k       ...
─────────────────────────────────────────────────
总计 (典型局面):           60–80 向量
```

### 15.6 核心模块设计

#### 物理牌 index (`tiles.py` 新增)

当前 `tile_index()` 把赤五映射到基牌（5mr → 5m），不适合结构化嵌入。需新增：

```python
def tile_index_physical(tile: str) -> int:
    """返回 0..36 的物理牌编号，保留赤五独立编码。
    0–33: 1m..7z (34 种基牌)
    34: 5mr
    35: 5pr
    36: 5sr
    """
    tile = normalize_tile(tile)
    if tile in {"5mr", "5pr", "5sr"}:
        return 34 + {"m": 0, "p": 1, "s": 2}[tile[1]]
    return tile_index(tile_base(tile))
```

#### FeatureEncoder (`src/mjgpt_training/feature_encoder.py`，新增)

```python
class FeatureEncoderConfig:
    n_tiles_phys: int = 37    # 物理牌种类
    n_winds: int = 4
    n_seats: int = 4
    n_int_types: int = 64     # 巡目/本场等
    n_meld_types: int = 4     # chi/pon/daiminkan/kakan/ankan
    n_action_kinds: int = 16  # 动作类型
    n_types: int = 20         # 特征块语义类型
    n_riichi: int = 2
    n_called: int = 2
    score_norm_mean: float = 25000.0
    score_norm_scale: float = 10000.0
    score_clip_min: float = -5.0
    score_clip_max: float = 15.0
    hidden_size: int = 384

class FeatureEncoder(nn.Module):
    def __init__(self, config):
        self.tile_emb   = nn.Embedding(config.n_tiles_phys, config.hidden_size)
        self.wind_emb   = nn.Embedding(config.n_winds, config.hidden_size)
        self.seat_emb   = nn.Embedding(config.n_seats, config.hidden_size)
        self.int_emb    = nn.Embedding(config.n_int_types, config.hidden_size)
        self.riichi_emb = nn.Embedding(config.n_riichi, config.hidden_size)
        self.called_emb = nn.Embedding(config.n_called, config.hidden_size)
        self.score_mlp  = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.type_emb   = nn.Embedding(config.n_types, config.hidden_size)
        self.global_emb = nn.Parameter(torch.randn(config.hidden_size))
        self.choice_emb = nn.Parameter(torch.randn(config.hidden_size))

    def _encode_score(self, score: float) -> Tensor:
        x = (score - self.config.score_norm_mean) / self.config.score_norm_scale
        x = max(self.config.score_clip_min, min(self.config.score_clip_max, x))
        return self.score_mlp(torch.tensor([x], dtype=torch.float32))

    def forward(self, snapshot_batch: list[dict]) -> Tensor:
        """返回 [B, total_vecs, hidden_size]"""
        ...
```

#### PolicySample 扩展 (`samples.py` 修改)

```python
@dataclass
class PolicySample:
    input_text: str
    # v2 新增：可选的 feature snapshot，不与 input_text 互斥
    feature_snapshot: dict | None = None
    ...
```

`feature_snapshot` 为 None 时走 v1 文本管线；不为 None 时走 v2 FeatureEncoder。converter 在构建 sample 时同时填写两个字段，A/B 对比时切换 `model.use_feature_encoder` 即可。

#### MahjongPolicyModel 修改 (`model.py`)

```python
class ModelConfig:
    use_feature_encoder: bool = False  # v2 开关
    feature_encoder: FeatureEncoderConfig | None = None
    # 其余字段不变
    ...

class MahjongPolicyModel(nn.Module):
    def __init__(self, config):
        if config.use_feature_encoder:
            self.feature_encoder = FeatureEncoder(config.feature_encoder)
            self.token_embedding = None
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.feature_encoder = None
        # blocks, norm, scorer 完全不变
        ...
```

### 15.7 动作结构化编码

v1 中每个动作用一个 `</Ai>` 位置表示。结构化编码下每个动作可能是多个向量（如 pon 包含 pon_type + source_tile + source_seat + consumed_tiles），需改用 **action_spans** 代替 `action_positions`：

```python
# v1
action_positions: list[int]  # [pos_a0, pos_a1, ...]

# v2
action_spans: list[tuple[int, int]]  # [(start, end), ...] 每个动作的向量区间
# 或等价地:
action_vector_positions: list[list[int]]  # [[v0, v1, v2], [v3, v4], ...]
```

scorer 对每个动作区间内的向量做 mean pooling（或取最后向量），得到 `h_action_i`，然后与 `h_state` 拼接做 MLP 打分。逻辑与 v1 的 gather 操作类似，只是输入从单 token 位置变为多向量区间。

### 15.8 关于 Causal Mask 顺序偏差的处置

v1 的 causal mask 导致后面的动作 representation 能看到前面的动作内容，反之不能。这个偏差确实存在，但 **不在 v2 第一版解决**：

1. 先用结构化的状态/数值/牌 embedding 跑通训练，产出一个可用的基线。
2. 再通过 A/B 指标（top1 按 action position 分解）判断顺序偏差是否真的影响效果。
3. 如果指标证实了偏差，再尝试修复方案（如 scorer MLP 的交互项 `h_state * h_action` 已在跨动作比较中隐式建模双向关系；若不够，再考虑混合 attention mask 或独立动作编码）。

不要过早把 attention mask 改造拉入 v2 第一版的工程范围。

### 15.9 实施路径

**当前最优路线**：

| 阶段 | 内容 | 前置条件 |
|------|------|----------|
| v1 完成 | 文本版训练闭环跑通，拿到 baseline top1 | 当前正在做 |
| converter 升级 | 在 record 中新增可选 `feature_snapshot`，不破坏 `state_text` | v1 稳定 |
| v2 实验分支 | `FeatureEncoder` 实现，先结构化状态侧（tile + score + meld + river + type_emb），动作仍复用文本 LEGAL_ACTIONS | converter 升级完成 |
| v2 基线 | tiny 在 data-draft 上过拟合，与 v1 做 A/B | FeatureEncoder 可用 |
| v2 完成 | 动作结构化 + 全量 dataset 训练 | v2 基线通过 |

**PR 拆分建议**：

- **PR #1**：状态侧结构化。新增 `feature_encoder.py`、`tile_index_physical()`；修改 `samples.py`（新增 `feature_snapshot`）、`model.py`（新增 `use_feature_encoder` 开关）、converter（产出 snapshot）。动作保留 v1 文本格式。
- **PR #2**：动作侧结构化。`action_spans` 替代 `action_positions`，FeatureEncoder 的动作编码模块，collator 适配。

**验收标准**：
- v2 tiny 在 data-draft 上的 top1 不低于 v1 文本版基线。
- 上下文利用率（有效信息向量占比）从 v1 的 ~42% 提升到 90%+。
- 分数相关决策（如以点数差为依据的攻防判断）在 v2 上的准确性有可观测提升。
- `feature_snapshot` 可 JSON 序列化/反序列化，与 `state_text` 共存无冲突。
