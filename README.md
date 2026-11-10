# Attention Item2Vec：整篮上下文共购召回

分支：`feature/attention-item2vec`。利用整篮上下文和商品对联合训练，导出可直接做余弦检索的商品向量。

本分支只提供 **attention** 训练：共享商品向量表＋单层多头自注意力，联合优化整篮上下文预测和商品对余弦。`--trainer attention` 可省略，默认就是注意力模型。

## 快速开始

准备商品CSV（`prod_id,prod_description`）、订单CSV（`order_id,prod_id,dt`，日期为`YYYYMMDD`）和本地M3E-base模型。商品编码按字符串读取，一个完整订单对应一个购物篮。

在项目根目录执行；修改CSV路径、模型目录和示例商品编码：

```bash
python -m pip install -r requirements.txt
WORK_DIR=/ml/output/item2vec-attention
MODEL_DIR=/ml/output/dataset/m3e-base
ITEM_ID=86141  # 替换为要查询的商品编码

# 1. 准备规范数据
bash scripts/fetch_data.sh \
  --item-input-csv /path/to/items.csv \
  --order-input-csv /path/to/orders.csv --work-dir "$WORK_DIR"

# 2. 生成文本向量和商品索引
bash scripts/generate_embeddings.sh --model-dir "$MODEL_DIR" --work-dir "$WORK_DIR"

# 3. 过滤购物篮：最大30件、商品最低支持5单、最小3件
bash scripts/preprocess_baskets.sh 30 5 3 --work-dir "$WORK_DIR"

# 4. 训练：128维、最大篮30、负采样15、10轮
bash scripts/train.sh 128 30 15 10 --trainer attention \
  --context-batch-size 256 --learning-rate 0.001 \
  --seed 17 --device auto --work-dir "$WORK_DIR"

# 5. 查询单个商品Top50
bash scripts/query_similar.sh "$ITEM_ID" 50 complement 0 --work-dir "$WORK_DIR"

# 6. 批量导出每个商品Top50
bash scripts/export_similarities.sh 50 512 complement 0 --work-dir "$WORK_DIR"
```

已有规范的`raw/`数据可跳过第1步；已有配套索引和文本向量可跳过第2步。使用独立`WORK_DIR`保存数据和模型，训练会替换该目录中的行为模型。

## 关键参数

| 命令 | 参数顺序 | 示例含义 |
| --- | --- | --- |
| `preprocess_baskets.sh 30 5 3` | 最大篮、最低支持单数、最小篮 | 保留3～30件商品的合格篮 |
| `train.sh 128 30 15 10` | 向量维度、最大篮、负采样数、轮数 | 默认值即此示例；每个目标抽取15个负样本，训练10轮 |
| `query_similar.sh ID 50 complement 0` | 商品、TopK、模式、文本权重 | 共购模式，文本融合权重0 |
| `export_similarities.sh 50 512 complement 0` | TopK、分块大小、模式、文本权重 | 每个商品导出50个候选，按512条分块 |

以下命名参数直接加在 `train.sh` 命令后；不传时使用默认值：

| 训练参数 | 默认值 | 作用与设置说明 |
| --- | --- | --- |
| `--context-batch-size` | 256 | 每批最多处理的目标样本数；越大通常越占显存，显存不足时调小 |
| `--learning-rate` | 0.001 | 初始学习率，训练中线性降低至约1/4；必须大于0 |
| `--attention-heads` | 4 | 自注意力头数；向量维度须能被头数整除 |
| `--temperature` | 0.1 | 对比损失温度；越小越强调高低相似度的差别，必须大于0 |
| `--pair-loss-weight` | 1 | 商品对余弦损失的权重；0表示只训练上下文预测，必须非负 |
| `--seed` | 1 | 随机种子；快速开始显式使用17，比较配置时保持一致 |
| `--device` | `auto` | 自动选择CUDA或CPU；也可指定`cpu`或`cuda`，CUDA不可用时指定会报错 |
| `--validation-orders CSV` | 不设置 | 提供独立验证订单，按验证指标保存最佳轮次；不设置则保存最后一轮 |
| `--patience` / `--min-delta` | 3 / 0.0001 | 验证主指标连续3轮未改善超过0.0001时早停；仅在提供验证集时可设置 |

**批大小按目标样本计，不按购物篮计。** 一篮3件商品会生成3个“其他商品 → 目标商品”样本。批大小参数名为 `--context-batch-size`，导出命令中的 `512` 是计算分块大小，两者独立。

例如，把训练批大小改为512、学习率改为0.0005：

```bash
bash scripts/train.sh 128 30 15 10 \
  --context-batch-size 512 --learning-rate 0.0005 \
  --attention-heads 4 --temperature 0.1 --pair-loss-weight 1 \
  --seed 17 --device auto --work-dir "$WORK_DIR"
```

训练最大篮须与预处理一致；最低支持默认5，若改为8，预处理用`30 8 3`，训练也须设置`MIN_ORDER_COUNT=8`。提供验证集时轮数为训练上限，验证主指标为有行为商品的宏平均Recall@50；训练与验证订单不能重叠，切分方法见[训练与验证](docs/usage-reference.md#训练与验证)。

## 模型与结果

训练同时优化整篮上下文预测和商品对余弦；查询直接使用共享的归一化商品向量，不运行注意力。

- `embeddings/behavior_item.npz`：商品召回使用的行为向量及元数据。
- `embeddings/attention_model.pt`：注意力编码器和共享向量表的完整状态。
- `results/query_<商品编码>.csv`：单商品查询结果。
- `results/item_similarity_complement.csv`：全部商品的共购候选。

结果包含`master_prod_id`、`slave_prod_id`、`similarity`和`recall_source`。`complement`分数是**余弦**，不是购买概率。

召回顺序：**自身行为 → 相近商品行为迁移 → 文本回退**。`complement 0`不做文本融合，但无行为商品仍可能回退文本。`recall_source`取`behavior`、`behavior_transfer`或`text_fallback`，分别对应三条路径。查询排除自身，候选不足时少于TopK。文本向量不参与行为训练，但现有查询入口需要它来支持回退。

## 更多说明

项目代码按职责分为：`data_fetch.py`取数、`embedding.py`文本编码、`preprocessing.py`购物篮处理、`attention.py`模型、`training.py`训练入口、`inference.py`查询与导出。训练只学习订单行为；文本用于推断时的融合和回退。

- [使用参考](docs/usage-reference.md)：完整参数、ODPS/SQL取数、验证选模、流水线和后台运行。
- [模型设计](docs/superpowers/specs/2026-11-04-attention-item2vec-design.md)与[本地验证](docs/2026-11-04-attention-validation.md)：已验证CPU合成数据；真实效果和CUDA尚未验证。
