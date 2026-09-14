# Item2Vec 文本与行为分数融合流水线

本项目分别生成 M3E/BERT 商品文本向量和 Item2Vec 用户行为向量，在推理阶段融合两路余弦相似度，用于商品召回和相似度检索。对外操作统一通过 `scripts/` 下的 Bash 脚本完成，Python 实现位于 `src/item2vec/`。

## 项目结构

```text
.
├── dataset/
│   ├── m3e-base/                  # 本地预训练模型
│   ├── raw/                       # 环境模板与 ODPS 原始 CSV
│   └── downstream/                # 流水线生成的向量与映射
├── scripts/
│   ├── fetch_data.sh              # 从 ODPS 获取原始数据
│   ├── generate_embeddings.sh     # 生成商品文本向量和索引
│   ├── train.sh                   # 独立训练商品行为向量
│   ├── query_similar.sh           # 查询单个商品的相似商品
│   ├── export_similarities.sh     # 批量导出商品相似度
│   └── run_pipeline.sh            # 按顺序执行完整流水线
├── src/item2vec/
│   ├── data_fetch.py
│   ├── embedding.py
│   ├── io.py
│   └── training.py
├── tests/
├── pyproject.toml
└── requirements.txt
```

## 环境准备

建议在虚拟环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

将 M3E-base 模型放入 `dataset/m3e-base/`，并从模板创建本地环境文件：

```bash
cp dataset/raw/.env.example dataset/raw/.env
```

数据下载脚本从 `dataset/raw/.env` 加载以下环境变量：

- `ALI_ACCESS_ID`（必填）
- `ALI_SECRET_ACCESS_KEY`（必填）
- `ALI_PROJECT`（必填）
- `ALI_ENDPOINT`（可选）

不要将凭据写入源码、日志或提交记录。

## 运行流水线

各阶段可独立运行：

```bash
bash scripts/fetch_data.sh
bash scripts/generate_embeddings.sh
bash scripts/train.sh
# 自定义：行为向量维度、最大购物篮大小、负采样数、训练轮数
bash scripts/train.sh 128 30 15 10
```

`order_item.csv` 由 ODPS 最近 30 个日历日的查询生成，必须包含 `order_id`、`user_id`、
`prod_id` 和 `dt`。训练以 `order_id` 为购物篮边界：每个订单内商品会去重并按商品索引
规范排序；保留 2–30 个不同商品的购物篮。超过 30 个不同商品的批量订单会被排除并记录数量。
单商品订单仍会计入商品订单数，但不产生训练商品对。

`train.sh` 的位置参数依次为 `VECTOR_SIZE MAX_BASKET_SIZE NEGATIVE EPOCHS`，默认值为
`128 30 15 10`。省略参数时使用默认训练配置；指定训练参数时需按该顺序完整提供四个值。
`MIN_ORDER_COUNT` 环境变量控制商品获得行为向量所需的最小订单数，默认值为 `5`。
行为训练使用固定的全购物篮 SGNS（skip-gram with negative sampling）。

训练生成独立行为向量 `behavior_item.npz`，其中包含 `vectors`、`item_ids`、
`order_counts` 和 `metadata`。文件写入后会先验证这些内容，再以原子替换方式发布。

相似度检索同时读取文本向量和行为向量。使用 `RECALL_MODE` 选择默认融合权重：

- `similar`：文本 `0.85`，行为 `0.15`
- `complement`：文本 `0.20`，行为 `0.80`
- `hybrid`：文本 `0.60`，行为 `0.40`（默认）

可选的 `TEXT_WEIGHT` 会覆盖所选模式的文本权重；`FULL_CONFIDENCE_ORDERS` 默认为 `50`。
每个商品的行为置信度为 `min(order_count / FULL_CONFIDENCE_ORDERS, 1)`，商品对取两者中较低的
置信度，并以其缩放行为权重、将剩余权重分配给文本相似度。任一商品缺少有效行为向量时，
该商品对完全回退为文本相似度。

例如：

```bash
bash scripts/train.sh
RECALL_MODE=similar bash scripts/export_similarities.sh 20 512
RECALL_MODE=complement FULL_CONFIDENCE_ORDERS=80 bash scripts/export_similarities.sh
RECALL_MODE=hybrid TEXT_WEIGHT=0.7 bash scripts/query_similar.sh ITEM_ID 10
```

`query_similar.sh` 的位置参数为 `ITEM_ID [TOPK [RECALL_MODE [TEXT_WEIGHT]]]`；
`export_similarities.sh` 的位置参数为 `[TOPK [BLOCK_SIZE [RECALL_MODE [TEXT_WEIGHT]]]]`。
两者也会读取上述环境变量。默认 `TOPK=10`、`BLOCK_SIZE=512` 和 `RECALL_MODE=hybrid`。

也可以按“拉取数据 → 生成文本向量 → 训练行为向量”的顺序运行完整流水线；流水线在训练完成后停止：

```bash
bash scripts/run_pipeline.sh
```

脚本使用严格错误处理；任一阶段失败时，完整流水线会立即停止。

## 输出文件

数据拉取阶段在 `dataset/raw/` 生成：

- `item.csv`：商品 ID 与商品描述
- `order_item.csv`：用户商品行为序列来源

文本向量与训练阶段在 `dataset/downstream/` 生成：

- `item2index.json`：商品 ID 到向量索引的映射
- `index2item.json`：向量索引到商品 ID 的映射
- `item.feat1CLS`：M3E/BERT 商品文本向量
- `behavior_item.npz`：已验证并原子发布的行为向量、商品 ID、订单数与训练元数据；无有效行为信号的商品行为向量为零
- `query_<ITEM_ID>.csv`：单商品 Top-K 余弦相似结果
- `item_similarity_<mode>.csv`：全量商品 Top-K 相似结果，`<mode>` 为 `similar`、`complement` 或 `hybrid`

查询与批量导出的 CSV schema 保持不变：`master_prod_id`、`slave_prod_id`、`similarity`。

原始 CSV、模型权重和下游生成物均为本地运行资产，不应提交到版本库。
