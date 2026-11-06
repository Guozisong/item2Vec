# 使用参考

本参考属于`feature/attention-item2vec`分支，仅支持注意力模型。快速开始见 [README](../README.md)。以下命令在项目根目录执行，沿用README中设置的 `WORK_DIR` 和 `MODEL_DIR`。

## 数据来源

商品和订单各自必选一种来源，两路可以混用：

| 来源 | 商品参数 | 订单参数 | 读取方式 |
| --- | --- | --- | --- |
| 本地CSV | `--item-input-csv FILE` | `--order-input-csv FILE` | 不读取ODPS凭据 |
| ODPS表 | `--item-input-table NAME` | `--order-input-table NAME` | 全表读取，无隐含日期过滤 |
| SQL文件 | `--item-input-query-file FILE` | `--order-input-query-file FILE` | 按SQL条件取数 |

```bash
bash scripts/fetch_data.sh \
  --item-input-table your_catalog_table \
  --order-input-query-file /path/to/orders.sql \
  --env-file /path/to/.env --work-dir "$WORK_DIR"
```

ODPS配置模板：

```bash
cp dataset/raw/.env.example dataset/raw/.env
```

| 配置项 | 兼容的小写名 | 要求 |
| --- | --- | --- |
| `ALI_ACCESS_ID` | `access_id` | 必填 |
| `ALI_SECRET_ACCESS_KEY` | `access_key` | 必填 |
| `ALI_PROJECT` | `project` | 必填 |
| `ALI_ENDPOINT` | `endpoint` | 默认杭州VPC Endpoint |

非空大写值优先。`--env-file` 默认 `dataset/raw/.env`。CSV允许额外列和任意列顺序，输出统一为项目字段，可选保留订单的 `user_id`。两份输入均校验后，才逐个原子替换规范CSV；校验失败保留原文件。

当前没有“默认近30天”限制，时间范围由SQL显式控制。例如近90个日历日：

```sql
SELECT order_id, user_id, prod_id, dt
FROM unisrec_raw_data
WHERE dt >= to_char(dateadd(getdate(), -89, 'dd'), 'yyyymmdd');
```

需要商品池约束时，在商品SQL中加入 `lh_rec_gds_station_base_pool_tmp` 等业务过滤；项目不自动添加。

文本编码从本地M3E/BERT模型取CLS向量，输出768维float32；自动选择CUDA或CPU，每批4条，最大512 Token。默认模型目录为 `dataset/m3e-base/`，可通过 `--model-dir` 修改。

## 训练与验证

训练提供 `--validation-orders CSV`。训练原始订单必须与验证订单无交集，且训练订单最晚日期早于验证订单的最早订单归属日期；训练原始 `raw/order_item.csv` 必须保留以供核验。

训练期单独预处理，验证期用于选型，测试期在参数冻结后确认。按完整订单的最晚日期切分；一张订单不会分散到多个时期。默认最后14天测试、此前14天验证，可通过 `--test-days`、`--validation-days` 调整。

下面是一条完整的注意力留出验证流程。使用独立实验目录，复用相同商品目录的索引和文本向量：

```bash
EVAL_DIR=/ml/output/experiments/attention-holdout
mkdir -p "$EVAL_DIR/split" "$EVAL_DIR/embeddings"

# 1. 从全部订单中分出训练、验证、测试期
python scripts/evaluate_item2vec.py split "$WORK_DIR/raw/order_item.csv" \
  --train "$EVAL_DIR/split/train.csv" \
  --validation "$EVAL_DIR/split/validation.csv" \
  --test "$EVAL_DIR/split/test.csv"

# 2. 实验工作目录只放训练期订单；商品目录保持一致
bash scripts/fetch_data.sh \
  --item-input-csv "$WORK_DIR/raw/item.csv" \
  --order-input-csv "$EVAL_DIR/split/train.csv" --work-dir "$EVAL_DIR"
cp "$WORK_DIR/embeddings/item2index.json" \
   "$WORK_DIR/embeddings/index2item.json" \
   "$WORK_DIR/embeddings/item.feat1CLS" "$EVAL_DIR/embeddings/"

# 3. 仅预处理训练期，用验证期选择最佳轮次
bash scripts/preprocess_baskets.sh 30 5 3 --work-dir "$EVAL_DIR"
bash scripts/train.sh 128 30 15 10 --trainer attention \
  --validation-orders "$EVAL_DIR/split/validation.csv" \
  --patience 3 --min-delta 0.0001 \
  --seed 17 --device auto --work-dir "$EVAL_DIR"

# 4. 配置冻结后导出，以测试期确认效果
bash scripts/export_similarities.sh 50 512 complement 0 --work-dir "$EVAL_DIR"
python scripts/evaluate_item2vec.py evaluate "$EVAL_DIR/split/test.csv" \
  "$EVAL_DIR/results/item_similarity_complement.csv" --top-k 50 \
  --item-csv "$EVAL_DIR/raw/item.csv"
```

每轮主指标为有行为商品的宏平均Recall@50，连续3轮未改善至少0.0001时早停，保存最佳轮次；阈值可调整。无验证集时保存固定轮次结果。逐轮记录在实验目录 `history.json`，最佳指标在 `validation.json`。

`evaluate` 输出宏平均召回、商品对召回、按留出同单次数加权的召回、NDCG、覆盖率、不同候选数与Top100候选占比。`--item-csv` 增加同/跨细分类加权召回，未知类别不进入分类指标分母。全目录导出包括冷启动回退，因此总体指标与训练日志中固定有行为集合的指标应分开解读。

比较多个模型时，对每个模型使用相同的训练/验证/测试切分、相同目录索引和相同导出模式。不要用含测试订单的全量模型评估留出效果，也不要根据测试结果反复调参。

`--processed-dir DIR` 可共享预处理产物，父目录必须保留配套的 `raw/order_item.csv`；每个模型的输出工作目录仍须有匹配的索引。最终全量重训使用已冻结参数和选定轮数，另存工作目录，不再用已纳入训练的验证期选轮数。

## 冷启动与融合

`complement` 无自身行为时，在同细分类中找参考商品：至少2个、最多5个、文本余弦不低于0.8，按温度0.1的softmax加权迁移商品行为向量。attention迁移结果归一化后计算余弦；参考不足则文本回退。

细分类从 `raw/item.csv` 描述中 `，商品品牌为` 前的文本取得，须保留商品CSV；描述不包含该标记时该商品类别未知。迁移向量不写回训练产物。

`similar` / `hybrid` 使用订单置信度：

```text
商品置信度 = min(商品总订单数 / FULL_CONFIDENCE_ORDERS, 1)
商品对置信度 = 两个商品置信度的较小值；任一缺行为时为0
实际行为权重 = (1 - TEXT_WEIGHT) × 商品对置信度
最终分数 = (1 - 实际行为权重) × 文本余弦 + 实际行为权重 × 行为余弦
```

`FULL_CONFIDENCE_ORDERS` 默认50，`complement` 忽略它。文本权重范围0～1，优先级为位置参数、`TEXT_WEIGHT` 环境变量、模式默认值；`RECALL_MODE` 可设置省略模式时的默认模式。

## 完整流水线

`run_pipeline.sh` 依次执行取数、文本编码、预处理、训练，失败即停止，不自动查询或导出。示例使用attention：

```bash
bash scripts/run_pipeline.sh \
  --item-input-query-file /path/to/items.sql \
  --order-input-query-file /path/to/orders.sql \
  --env-file /path/to/.env --model-dir "$MODEL_DIR" \
  --min-basket-size 3 --trainer attention --attention-heads 4 \
  --device auto --work-dir "$WORK_DIR"
```

取数前先做训练参数预检。流水线使用128维、最大篮30、负采样15、10轮；最低支持默认5，可通过 `MIN_ORDER_COUNT` 环境变量修改，最小篮默认2，可显式设置。注意力头数须能整除128。需要其他维度、轮数或负采样数时使用README中的独立阶段命令。

省略 `--trainer` 默认使用attention。公共选项 `--work-dir` 默认项目 `outputs/`，`--model-dir` 默认 `dataset/m3e-base/`，位置参数和命名选项可以前后放置。

## 后台运行与检查

手动后台运行示例：

```bash
mkdir -p "$WORK_DIR/logs"
nohup bash scripts/train.sh 128 30 15 10 --trainer attention \
  --seed 17 --device auto --work-dir "$WORK_DIR" \
  > "$WORK_DIR/logs/train.out" 2>&1 &
tail -f "$WORK_DIR/logs/train.out"
```

不要同时向同一工作目录训练。`nohup` 只能应对终端断开，不能防止机器休眠、重启或资源不足，也不代表程序会自动开展后续所有实验。

开发检查：

```bash
PYTHONPATH=src python -m pytest -q
for script in scripts/*.sh; do bash -n "$script" || exit; done
```

预处理清单包含过滤计数：`small_baskets` 记录首次过滤中低于下限的篮；`dropped_after_support` 记录移除低支持商品后不够最小篮大小的篮。`support.npz` 顺序写入，不对输出文件执行回退seek，与NumPy读取格式兼容。

最终保留 `raw/`、`processed/`、`embeddings/` 与需要的 `results/`。实验缓存可在不再需要时清理；历史报告引用的临时目录可能已不存在。

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

### 训练参数

```text
bash scripts/train.sh [维度 最大篮 负采样数 轮数] [--trainer attention] [--work-dir DIR]
```

前四个位置参数必须同时提供或全部省略，程序默认值为 `128 30 15 10`。

| 参数或选项 | 默认值 | 适用范围与约束 |
| --- | --- | --- |
| 维度 | 128 | 向量维度须能被注意力头数整除 |
| 最大篮 | 30 | 必须与预处理一致 |
| 负采样数 | 15 | 每个目标抽取的负样本数 |
| 轮数 | 10 | 固定训练轮数；提供验证集时为最多轮数 |
| `--trainer` | `attention` | 唯一支持的训练器，可省略 |
| `--seed` | 1 | 随机种子 |
| `--device` | `auto` | `auto`、`cpu`、`cuda`；CUDA 不可用时显式指定会报错 |
| `--validation-orders CSV` | 不设置 | 按验证集指标选最佳轮次 |
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

负样本固定排除当前整篮商品，按最终训练频次的平方根抽取。

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

注意力参与训练，单商品查询直接使用固定商品向量做余弦检索，不需要运行注意力编码器。完整模型保存在 `attention_model.pt`，目前训练入口不提供断点续训。[详细设计](superpowers/specs/2026-11-04-attention-item2vec-design.md)

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
│                 behavior_item.npz、attention_model.pt
├── results/      query_<ITEM_ID>.csv、item_similarity_<mode>.csv
├── history.json       逐轮训练记录
└── validation.json    有验证集时的最佳验证指标
```

`processed/manifest.json` 记录过滤参数、计数和文件哈希；训练前核验输入是否配套。`behavior_item.npz` 用于商品召回；注意力产物显式记录 `trainer=attention, behavior_score=cosine`。

查询和导出仅接受上述注意力产物；其他训练器的历史产物需要重新训练，不能在本分支直接加载。

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
| `similarity` | 共购模式：余弦或融合余弦；similar/hybrid：融合余弦 |
| `recall_source` | 仅complement：`behavior`、`behavior_transfer`、`text_fallback` |

自动排除主商品自身，候选不足时可能少于TopK；无行为且文本向量为零时不输出该主商品。分数不是购买概率，不同模型的阈值不能直接复用。迁移与文本回退是推断结果，不代表主商品已发生过共购。
