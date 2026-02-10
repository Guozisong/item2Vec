# Item2Vec Project

本项目利用预训练的 BERT 模型 (M3E-base) 生成商品 Embedding，并结合用户行为数据 (购物篮) 训练 Item2Vec 模型，最终融合两者生成高质量的商品向量。

## 项目结构

```
p:\pycharm_workSpace\item2Vec/
├── dataset/
│   ├── downstream/        # 存放生成的 Embedding 和索引文件
│   ├── m3e-base/          # 预训练的 BERT 模型文件
│   └── raw/               # 原始数据及数据获取脚本
├── item2Embedding.py      # 使用 BERT 生成初始商品 Embedding
├── item2VecWithBert.py    # 结合 BERT Embedding 和 Item2Vec 进行训练
├── utils.py               # 工具函数 (加载模型、计算相似度等)
└── requirements.txt       # 项目依赖
```

## 环境依赖

请使用以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

主要依赖库包括：
- torch
- pandas
- numpy
- transformers
- gensim
- scikit-learn
- odps

## 使用说明

### 1. 数据准备
如果是从 ODPS 获取数据，请配置 `dataset/raw/get_data_from_odps.py` 中的 `access_id`, `access_key` 等信息，然后运行该脚本下载数据：

```bash
python dataset/raw/get_data_from_odps.py
```
该步骤会生成 `lianhua_item.csv` (商品信息) 和 `lianhua_order_item.csv` (订单数据)。

### 2. 生成 BERT Embedding
运行 `item2Embedding.py` 利用商品描述生成初始的语义向量：

```bash
python item2Embedding.py
```
该步骤会：
- 构建商品 ID 到索引的映射 (`item2index.json`, `index2item.json`)
- 使用 M3E-base 模型生成商品描述的 Embedding (`lianhua_item.feat1CLS`)

### 3. 训练 Item2Vec 并融合
运行 `item2VecWithBert.py` 结合用户行为数据微调向量：

```bash
python item2VecWithBert.py
```
该步骤会：
- 读取订单数据构建购物篮序列
- 初始化 Word2Vec 模型
- 融合 BERT 向量与 Word2Vec 向量
- 基于用户行为进行微调训练
- 输出最终的商品 Embedding

## 核心逻辑
- **item2Embedding.py**: 使用预训练模型 (如 M3E) 对商品描述文本进行编码，提取 CLS token 或 Mean Pooling 作为语义向量。
- **item2VecWithBert.py**:
    1. 将购物篮中的商品转换为索引序列。
    2. 使用 Word2Vec 学习商品共现关系。
    3. 将 BERT 语义向量作为先验知识，与 Word2Vec 向量进行加权融合。
    4. 继续在购物篮序列上进行微调，使向量既包含语义信息又包含行为信息。
