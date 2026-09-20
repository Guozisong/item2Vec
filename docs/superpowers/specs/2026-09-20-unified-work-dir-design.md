# Item2Vec 统一工作目录设计

## 目标

为所有 Bash 阶段提供统一的 `--work-dir DIR` 参数，使 PAI 可视化建模中的挂载目录成为全流程产物根目录。后续阶段从同一工作目录读取前序产物。

## 目录结构

```text
<work-dir>/
├── raw/
│   ├── item.csv
│   └── order_item.csv
├── embeddings/
│   ├── item2index.json
│   ├── index2item.json
│   ├── item.feat1CLS
│   └── behavior_item.npz
└── results/
    ├── query_<ITEM_ID>.csv
    └── item_similarity_<mode>.csv
```

- `fetch_data` 写入 `raw/`。
- `generate_embeddings` 从 `raw/` 读取商品数据，写入 `embeddings/`。
- `train` 从 `raw/` 和 `embeddings/` 读取数据，写入 `embeddings/`。
- `query_similar` 和 `export_similarities` 从 `embeddings/` 读取模型产物，写入 `results/`。

省略 `--work-dir` 时使用项目内的 `outputs/`。脚本按需创建输出子目录。

## 外部输入

运行产物与外部输入分开管理：

- ODPS 配置默认读取项目内 `dataset/raw/.env`，`fetch_data.sh` 支持 `--env-file FILE` 覆盖。
- M3E/BERT 模型默认读取项目内 `dataset/m3e-base/`，`generate_embeddings.sh` 支持 `--model-dir DIR` 覆盖。

`--env-file` 和 `--model-dir` 可以指向 PAI 的独立挂载目录，不复制到工作目录。

## Bash 接口

```text
bash scripts/fetch_data.sh [--work-dir DIR] [--env-file FILE]
bash scripts/generate_embeddings.sh [--work-dir DIR] [--model-dir DIR]
bash scripts/train.sh [VECTOR_SIZE MAX_BASKET_SIZE NEGATIVE EPOCHS] [--work-dir DIR]
bash scripts/query_similar.sh ITEM_ID [TOPK [RECALL_MODE [TEXT_WEIGHT]]] [--work-dir DIR]
bash scripts/export_similarities.sh [TOPK [BLOCK_SIZE [RECALL_MODE [TEXT_WEIGHT]]]] [--work-dir DIR]
bash scripts/run_pipeline.sh [--work-dir DIR] [--env-file FILE] [--model-dir DIR]
```

现有位置参数及环境变量语义保持不变。选项允许放在位置参数之前或之后。未知选项、重复选项和缺少选项值均返回用法错误。

`run_pipeline.sh` 将同一个工作目录传给 `fetch_data.sh`、`generate_embeddings.sh` 和 `train.sh`，并把 `--env-file`、`--model-dir` 只传给对应阶段。

每个阶段开始时打印解析后的工作目录，便于检查 PAI 挂载路径。

## Python 边界

- `data_fetch.py` 接收显式输出目录，不再从当前工作目录推导 `dataset/raw`。
- 其他 Python 模块继续接收 Bash 解析后的明确路径。
- 推理函数分别接收向量目录与结果目录，读取和写入不再共用一个目录参数。

## 错误处理

- 输入文件或目录不存在时，错误信息包含实际解析后的绝对路径。
- 工作目录无法创建时，由严格 Bash 错误处理终止。
- 后续阶段缺少前序产物时立即失败，不创建无效结果。
- 原有训练、向量和推理数据校验保持不变。

## 兼容性

- 训练和推理的位置参数及环境变量保持兼容。
- 默认产物目录从 `dataset/` 调整为 `outputs/`，属于本次明确的目录重构。
- 原有 `dataset/raw/.env` 和 `dataset/m3e-base/` 继续作为默认外部输入。
- `.gitignore` 忽略新的 `outputs/`。

## 验证

自动化测试覆盖：

1. 每个脚本省略 `--work-dir` 时使用项目内 `outputs/`。
2. 每个脚本使用自定义工作目录时读写正确子目录。
3. `run_pipeline.sh` 向各阶段传递同一个工作目录。
4. 外部配置文件和模型目录只传给对应阶段。
5. 选项可位于位置参数之前或之后。
6. 未知、重复和缺值选项被拒绝。
7. 原有位置参数、召回模式、文本权重和订单置信度行为不变。
8. 全量测试与 `git diff --check` 通过。
