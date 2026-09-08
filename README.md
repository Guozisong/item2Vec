# Item2Vec 融合向量流水线

本项目将 M3E/BERT 商品文本向量与 Item2Vec 用户行为向量融合，生成可用于商品召回和相似度检索的向量产物。对外操作统一通过 `scripts/` 下的 Bash 脚本完成，Python 实现位于 `src/item2vec/`。

## 项目结构

```text
.
├── .env.example
├── dataset/
│   ├── m3e-base/                  # 本地预训练模型
│   ├── raw/                       # ODPS 下载的原始 CSV
│   └── downstream/                # 流水线生成的向量与映射
├── scripts/
│   ├── fetch_data.sh              # 从 ODPS 获取原始数据
│   ├── generate_embeddings.sh     # 生成商品文本向量和索引
│   ├── train.sh                   # 训练并导出融合向量
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

将 M3E-base 模型放入 `dataset/m3e-base/`。数据下载脚本从 `dataset/raw/.env` 加载以下环境变量：

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
```

也可以按“拉取数据 → 生成文本向量 → 训练融合向量”的顺序运行完整流水线：

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
- `trained_item.featCLS`：融合用户行为后的商品向量
- `item_cosine_similarity.csv`：商品 Top-K 余弦相似结果

原始 CSV、模型权重和下游生成物均为本地运行资产，不应提交到版本库。
