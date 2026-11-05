# 使用参考

本参考属于`feature/attention-item2vec`分支，主流程是注意力模型；旧训练器仅作对照。主流程与分支区别见 [README](../README.md)。以下命令在项目根目录执行，沿用README中设置的 `WORK_DIR` 和 `MODEL_DIR`。

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

`attention` 和 `sgns` 提供 `--validation-orders CSV`。训练原始订单必须与验证订单无交集，且训练订单最晚日期早于验证订单的最早订单归属日期；训练原始 `raw/order_item.csv` 必须保留以供核验。

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

## 自动选模

`tune_item2vec.py` **仅覆盖原Gensim与PyTorch SGNS，尚未包含attention**。准备好商品索引后运行：

```bash
EXP_DIR=/ml/output/experiments/balanced-sgns
python scripts/tune_item2vec.py prepare --work-dir "$WORK_DIR" \
  --experiment-dir "$EXP_DIR" --candidate-bias-weight 0.4 --device auto
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase ablation
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase dimensions
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase confirmation
python scripts/tune_item2vec.py test --experiment-dir "$EXP_DIR"
```

候选校正强度须与基线一致；省略时读取现有模型元数据，缺失时必须显式指定。默认按最后14天测试、此前14天验证切分，仅训练期预处理，先做旧配方与A0～A3对照，再选维度、进行三种子确认。

检查宏平均召回、全来源加权召回、低频召回和候选集中度。没有合格方案时 `selection.json` 记录 `keep_baseline`。最终测试只评估已冻结方案与基线，模型和 `test_report.json` 保存在实验目录，不覆盖生产模型。

输入、代码、配置和模型均校验哈希，阻止混用依赖版本。相同已完成配置可以复用；不完整运行会报错，应使用新实验目录重试。详细门槛见[实施方案](superpowers/plans/2026-11-02-balanced-sgns-and-model-selection.md)。

## 旧训练器对照

以下命令仅在需要比较旧方案时使用，不是注意力主流程。`--trainer gensim`仍为程序默认值；使用旧训练器时，attention分支也继续按旧模型打分。

先按README的步骤1～3为每个对照准备独立工作目录，再分别训练，避免覆盖注意力模型：

```bash
BASELINE_DIR=/ml/output/item2vec-baseline

# Gensim旧方案
bash scripts/train.sh 128 30 15 10 --trainer gensim \
  --seed 17 --work-dir "$BASELINE_DIR"

# 或在另一个配套工作目录中使用PyTorch SGNS
SGNS_DIR=/ml/output/item2vec-sgns
bash scripts/train.sh 128 30 15 10 --trainer sgns \
  --basket-pair-budget 0 --negative-exclusion basket \
  --hard-negative-ratio 0 --seed 17 --device auto --work-dir "$SGNS_DIR"
```

商品对模型的`complement`输出百分位，注意力模型输出余弦；对照应比较召回指标，而不是直接比较两种模型的分数大小或复用同一分数阈值。

## SGNS 与候选校正

`--basket-pair-budget 0` 关闭贡献平衡。预算为正时，每个商品对及其负样本损失的权重为 `min(1, 预算/[篮大小×(篮大小−1)])`，保留全部合格商品对。例如10件商品生成90个有方向的对，预算30时每对权重为1/3，总权重30。

`--negative-exclusion basket` 排除当前整篮商品，`target` 仅排除当前正样本的目标。负样本分布使用最终训练篮频次的0.5次幂，不使用全部原始订单频次。

困难负样本默认关闭。总负采样15、`--hard-negative-ratio 0.2`、预热2轮时，前2轮随机采样，第3轮起尝试12个随机负样本加3个模型高分负样本；候选池默认128。贡献平衡和困难负样本是否有效均需验证。

Gensim和SGNS的候选偏置在训练时计算：一个候选与所有有效输入向量的最高10个内积的均值，乘以校正强度；不足10个取全部。`complement` 从内积中减去该偏置，再转换为百分位排名。非零文本权重融合行为分数与文本余弦各自的百分位。

旧模型缺少上下文向量时使用输入向量余弦，但仍输出百分位；有上下文、缺少偏置时使用未经校正的内积。重新训练才能获得新上下文或新校正设置。attention通过显式元数据选择原始余弦路径，不使用此偏置。

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

省略 `--trainer` 仍使用Gensim；`--candidate-bias-weight` 可透传给原模型训练，默认0.4。公共选项 `--work-dir` 默认项目 `outputs/`，`--model-dir` 默认 `dataset/m3e-base/`，位置参数和命名选项可以前后放置。

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
