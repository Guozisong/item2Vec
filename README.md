# Attention Item2Vec：整篮上下文与余弦共购召回

**当前分支：`feature/attention-item2vec`。主方案：整篮上下文注意力＋共享商品向量的余弦联合训练。**

本分支的目标是利用整张购物篮的上下文学习商品向量，并直接训练最终用于单品检索的余弦关系。注意力用于训练，召回时导出固定商品向量计算TopK。行为训练不输入商品文本；文本用于冷启动回退与可选融合。

**本分支代码默认训练器仍为`gensim`，不会自动启用注意力；必须显式指定`--trainer attention`。** 旧Gensim/SGNS作为对照保留，命令放在[旧训练器对照](docs/usage-reference.md#旧训练器对照)。本地已验证CPU合成数据，真实数据效果与CUDA路径尚未验证。

旧方案见[feature/score-fusion分支](https://github.com/Guozisong/item2Vec/tree/feature/score-fusion)。

## 两个分支的关键区别

下表比较的是各分支的主方案。attention分支仍保留旧训练器作为对照，选择旧训练器时继续使用旧模型语义。

| 比较项 | `feature/score-fusion`：商品对模型 | `feature/attention-item2vec`：注意力模型 |
| --- | --- | --- |
| 训练输入 | 一个主商品，预测同篮另一个商品 | 篮内其他商品联合预测被移除的目标 |
| 样本构造 | n件商品生成n×(n−1)个有方向的正样本对 | n件商品生成n个留一目标样本；另对目标与上下文计算商品对损失 |
| 模型结构 | 两套向量表：输入U、上下文V；无整篮联合编码器 | 共享向量表E、一层多头自注意力、平均池化 |
| 训练目标 | SGNS正负样本损失，学习U到V的匹配 | 上下文预测损失＋商品对余弦对比损失，共同更新E |
| 单品召回打分 | U与候选V内积减偏置，再转换为百分位 | 归一化E之间的余弦，不使用旧候选偏置 |
| `complement 0`的分数 | 百分位排名，不能当成原始余弦 | 原始商品余弦；迁移与回退路径也用余弦 |
| 行为模型产物 | `behavior_item.npz`：输入/上下文向量、候选偏置 | 同名NPZ保存归一化商品向量，并额外保存`attention_model.pt` |
| 训练入口 | `--trainer gensim`或`--trainer sgns` | 主方案必须显式传`--trainer attention` |

两者都使用订单购物篮，文本只用于冷启动与可选融合；都支持自身行为→行为迁移→文本回退。**变化的核心是联合上下文学习，以及训练目标与最终商品余弦空间的直接对齐。** 注意力方案是否提高真实召回，仍需留出集验证。

同名`behavior_item.npz`不代表内部结构或分数相同。attention分支按元数据兼容旧模型；score-fusion没有新余弦类型的显式路由，加载注意力向量也会按旧兼容路径转换成百分位。模型与部署分支应对应，两个方案使用独立输出目录。

[分支区别](#两个分支的关键区别) · [运行步骤](#运行步骤) · [参数说明](#参数说明) · [注意力模型](#注意力模型) · [输入与输出](#输入与输出) · [召回规则](#召回规则) · [验证与运行参考](#验证与运行参考)

## 运行步骤

在项目根目录执行，先安装 `requirements.txt` 中的依赖。本地验证使用 Python 3.13；部署环境需满足这些固定依赖的版本要求。

```bash
python -m pip install -r requirements.txt
WORK_DIR=/ml/output/item2vec-attention
MODEL_DIR=/ml/output/dataset/m3e-base
```

`WORK_DIR` 是本次运行的数据和产物目录；`MODEL_DIR` 是已下载的本地 M3E-base 模型目录，请按环境修改。所有阶段使用同一个 `WORK_DIR`。

### 1. 准备商品与订单

```bash
bash scripts/fetch_data.sh \
  --item-input-csv /path/to/items.csv \
  --order-input-csv /path/to/orders.csv --work-dir "$WORK_DIR"
```

输出 `raw/item.csv` 和 `raw/order_item.csv`。已有这两份规范文件可跳过本步；ODPS 表、SQL 取数见[数据来源](docs/usage-reference.md#数据来源)。

### 2. 生成文本向量与商品索引

```bash
bash scripts/generate_embeddings.sh --model-dir "$MODEL_DIR" --work-dir "$WORK_DIR"
```

输出两份商品索引和768维文本向量 `embeddings/item.feat1CLS`。即使使用纯行为召回，也需要配套商品索引，现有查询入口还会加载文本向量以支持冷启动。

### 3. 过滤训练购物篮

```bash
bash scripts/preprocess_baskets.sh 30 5 3 --work-dir "$WORK_DIR"
```

三个数字依次表示：**最大篮30件、商品最低支持5单、最小篮3件**。商品数量按不同商品计数；处理后保留3～30件商品的篮。详细规则见[购物篮参数](#购物篮参数)。

### 4. 训练本分支的注意力模型

同一个工作目录中的行为模型会被本次训练结果替换。对照实验请使用独立输出目录。

```bash
bash scripts/train.sh 128 30 15 10 --trainer attention \
  --attention-heads 4 --temperature 0.1 --pair-loss-weight 1 \
  --context-batch-size 256 --learning-rate 0.001 \
  --seed 17 --device auto --work-dir "$WORK_DIR"
```

四个位置参数是 **向量维度128、最大篮30、负采样数量15、训练10轮**。`auto` 在 CUDA 可用时用 GPU，否则用 CPU。

训练输出 `embeddings/behavior_item.npz`。注意力模型还输出完整的 `embeddings/attention_model.pt`。未提供验证集时训练固定轮数；按验证集保存最佳轮次的方法见[训练与验证](docs/usage-reference.md#训练与验证)。

### 5. 查询单个商品

将 `ITEM_ID` 替换为商品编码：

```bash
bash scripts/query_similar.sh ITEM_ID 50 complement 0 --work-dir "$WORK_DIR"
```

输出 `results/query_<ITEM_ID>.csv`。

### 6. 批量导出 Top50

```bash
bash scripts/export_similarities.sh 50 512 complement 0 --work-dir "$WORK_DIR"
```

输出 `results/item_similarity_complement.csv`。`512` 是查询分块大小，与训练向量维度无关。

两条命令中的 `complement 0` 表示共购模式、文本融合权重为0；没有自身行为的商品仍会尝试行为迁移，参考不足时回退文本。省略模式时，脚本默认使用 `hybrid`。

## 参数说明

### 购物篮参数

```text
bash scripts/preprocess_baskets.sh [最大篮 [商品最低支持单数 [最小篮]]] [--work-dir DIR]
```

| 参数 | 程序默认值 | 说明 |
| --- | ---: | --- |
| 最大篮 | 30 | 篮内不同商品数上限 |
| 商品最低支持单数 | 5 | 商品至少出现在多少个首次过滤合格的篮中 |
| 最小篮 | 2 | 篮内不同商品数下限；运行示例显式设为3 |

处理顺序：完整订单分组 → 商品去重并移除目录外商品 → 按篮大小过滤 → 移除低支持商品 → 再检查最小篮大小。同细分类购物篮保留。最终篮中商品的出现次数可能因最后一步丢篮而少于首次统计的支持数。

也支持 `--min-basket-size N`，但不能与第三个位置参数同时设置。支持数省略时可由 `MIN_ORDER_COUNT` 环境变量设置；原始订单、索引或过滤参数变化后，需要重新预处理和训练。

### 通用训练参数

```text
bash scripts/train.sh [维度 最大篮 负采样数 轮数 [候选偏置强度]] [--trainer NAME] [--work-dir DIR]
```

前四个位置参数必须同时提供或全部省略，程序默认值为 `128 30 15 10`。

| 参数或选项 | 默认值 | 适用范围与约束 |
| --- | --- | --- |
| 维度 | 128 | attention 的维度须能被注意力头数整除 |
| 最大篮 | 30 | 必须与预处理一致；Gensim 将其作为完整篮上下文窗口 |
| 负采样数 | 15 | attention 每个目标抽取的负样本数；SGNS 每个正商品对的负样本数 |
| 轮数 | 10 | 固定训练轮数；提供验证集时为最多轮数 |
| `--trainer` | `gensim` | `gensim`、`sgns`、`attention` |
| `--seed` | 1 | 三种训练器均支持 |
| `--device` | `auto` | 仅 attention/sgns：`auto`、`cpu`、`cuda`；CUDA 不可用时显式指定会报错 |
| `--validation-orders CSV` | 不设置 | 仅 attention/sgns；按验证集指标选最佳轮次 |
| `--patience` / `--min-delta` | 3 / 0.0001 | 需要验证集；主指标连续若干轮未改善到阈值时早停 |
| `--processed-dir DIR` | 工作目录内 `processed/` | 读取共享预处理产物；输出目录仍须有配套商品索引 |

**训练的商品支持阈值必须与预处理一致。** 训练默认 `MIN_ORDER_COUNT=5`，最小篮大小自动读取预处理清单。例如预处理使用 `30 8 3`，训练时也必须设置：

```bash
MIN_ORDER_COUNT=8 bash scripts/train.sh 128 30 15 10 --trainer attention --work-dir "$WORK_DIR"
```

### 注意力专属参数

| 选项 | 默认值 | 含义 |
| --- | ---: | --- |
| `--attention-heads` | 4 | 单层自注意力的头数 |
| `--temperature` | 0.1 | 两项余弦对比损失的温度，必须大于0 |
| `--pair-loss-weight` | 1 | 商品对损失权重；0用于仅上下文预测的消融实验 |
| `--context-batch-size` | 256 | 每批目标样本数，一篮n件商品生成n个目标样本 |
| `--learning-rate` | 0.001 | 初始学习率，按训练进度线性降低至约1/4 |

只与 `--trainer attention` 一起使用。注意力模型固定排除整篮负样本，不接受下表中的 SGNS 参数，也不使用候选偏置校正。

### 对照用 PyTorch SGNS 参数

| 选项 | 默认值 | 含义 |
| --- | ---: | --- |
| `--basket-pair-budget` | 0 | 0关闭篮贡献平衡；正数按篮内商品对数量降低大篮损失权重，保留全部商品对 |
| `--negative-exclusion` | `basket` | `basket` 排除整篮商品，`target` 只排除当前正样本目标 |
| `--pair-batch-size` | 4096 | 每批正商品对数量上限 |
| `--hard-negative-ratio` | 0 | 困难负样本占比，默认关闭 |
| `--hard-negative-warmup` | 2 | 困难负样本开始前的随机采样轮数 |
| `--hard-negative-pool-size` | 128 | 困难负样本候选池大小 |

这些参数只与 `--trainer sgns` 一起使用；篮贡献权重与困难负样本例子见[SGNS 与候选校正](docs/usage-reference.md#sgns-与候选校正)。

### 对照用 Gensim / SGNS 候选偏置

第五个位置参数或 `--candidate-bias-weight` 设置校正强度，默认0.4，0关闭，两种写法不能同时使用：

```bash
bash scripts/train.sh 128 30 15 10 0.6 --trainer gensim --work-dir "$WORK_DIR"
```

该值在训练时随模型保存，重新导出不会修改旧模型的偏置。它与查询/导出的文本权重是不同参数；**attention 会忽略候选偏置强度，并在日志中说明。**

### 查询与导出参数

```text
bash scripts/query_similar.sh ITEM_ID [TOPK [RECALL_MODE [TEXT_WEIGHT]]] [--work-dir DIR]
bash scripts/export_similarities.sh [TOPK [BLOCK_SIZE [RECALL_MODE [TEXT_WEIGHT]]]] [--work-dir DIR]
```

`TOPK` 默认10，导出 `BLOCK_SIZE` 默认512；模式默认 `hybrid`，文本权重默认值由模式决定，见下方召回规则。选项可放在位置参数前后。

## 注意力模型

```text
购物篮逐个移除目标商品
  ├─ 其他商品 → 共享向量表 → 一层4头自注意力 → 平均池化 → 归一化
  │                                                   ↓
  │                                         与目标/负样本做余弦打分
  │                                                   ↓
  │                                             上下文预测损失
  └─ 目标、同单商品、负样本 → 同一向量表 → 归一化 → 商品对余弦损失

总损失 = 上下文预测损失 + pair-loss-weight × 商品对损失
最终召回向量 = 共享商品向量表中每个商品的归一化向量
```

默认128维、4头、前馈层 `128→256→128`，无位置编码，屏蔽补齐位置。每次平均使用完整上下文和随机非空子集的预测损失，随机子集包括单商品上下文；商品对损失在完整正上下文内取平均。负样本按最终训练频次的平方根抽取，排除当前整篮。

注意力参与训练，单商品查询直接使用固定商品向量做余弦检索，不需要运行注意力编码器。完整模型保存在 `attention_model.pt`，目前训练入口不提供断点续训。[详细设计](docs/superpowers/specs/2026-11-04-attention-item2vec-design.md)

## 输入与输出

| 输入文件 | 必需字段 | 约定 |
| --- | --- | --- |
| `raw/item.csv` | `prod_id,prod_description` | 同商品保留首条描述；用于文本编码和冷启动分类 |
| `raw/order_item.csv` | `order_id,prod_id,dt` | `dt` 为 `YYYYMMDD`；可带 `user_id`，当前训练不使用 |

编码按字符串读取，保留前导零。一个完整 `order_id` 对应一个购物篮，不能用用户ID代替。索引与向量必须来自同一商品目录。

```text
<work-dir>/
├── raw/          item.csv、order_item.csv
├── processed/    baskets.jsonl、support.npz、manifest.json
├── embeddings/   item2index.json、index2item.json、item.feat1CLS
│                 behavior_item.npz、attention_model.pt（仅attention）
├── results/      query_<ITEM_ID>.csv、item_similarity_<mode>.csv
├── history.json       attention/sgns逐轮训练记录
└── validation.json    attention/sgns有验证集时的最佳验证指标
```

`processed/manifest.json` 记录过滤参数、计数和文件哈希；训练前核验输入是否配套。`behavior_item.npz` 用于商品召回；注意力产物显式记录 `trainer=attention, behavior_score=cosine`。

## 召回规则

| 模式 | 默认文本权重 | 用途 |
| --- | ---: | --- |
| `complement` | 0 | 共购召回，建议显式传入 |
| `similar` | 0.85 | 偏语义相似 |
| `hybrid` | 0.60 | 融合相似度，查询/导出脚本默认模式 |

`complement` 按 **自身行为 → 相近商品行为迁移 → 文本回退** 处理。行为和迁移路径只召回有行为向量的商品；文本回退可使用整个目录。文本融合权重设为0也不关闭冷启动回退。

| 输出字段 | 含义 |
| --- | --- |
| `master_prod_id` / `slave_prod_id` | 主商品编码 / 候选商品编码 |
| `similarity` | attention共购模式：余弦或融合余弦；Gensim/SGNS共购模式：百分位；similar/hybrid：融合余弦 |
| `recall_source` | 仅complement：`behavior`、`behavior_transfer`、`text_fallback` |

自动排除主商品自身，候选不足时可能少于TopK；无行为且文本向量为零时不输出该主商品。分数不是购买概率，不同模型的阈值不能直接复用。迁移与文本回退是推断结果，不代表主商品已发生过共购。

## 验证与运行参考

- [使用参考](docs/usage-reference.md)：ODPS/SQL取数、留出验证、自动选模、流水线、后台运行、冷启动细则。
- [注意力实现与本地验证](docs/2026-11-04-attention-validation.md)：430项测试和合成数据结果；尚无真实数据效果结论。
- [SGNS验证记录](docs/2026-11-02-balanced-sgns-validation.md)：贡献平衡、负样本与选模的检查范围。
- [购物篮与候选打分实验](docs/2026-10-29-item2vec-co-basket-results.md)、[冷启动实验](docs/2026-10-29-cold-start-results.md)。

脚本位于 `scripts/`，Python代码位于 `src/item2vec/`。原始数据、凭据、模型和 `outputs/` 不提交到Git。
