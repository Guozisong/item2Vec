# 使用参考：Skip-gram 商品对模型

本文件属于`feature/score-fusion`分支。首次使用请按[README](../README.md)中的六步流程执行；以下保留分支对照、完整参数、取数和验证说明。命令均在项目根目录运行，沿用README设置的`WORK_DIR`和`MODEL_DIR`。

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

## 本分支模型结构

```text
合格购物篮[A,B,C]
  → 商品对：A→B、A→C、B→A、B→C、C→A、C→B
  → 主商品查输入向量表U，正/负候选查上下文向量表V
  → SGNS损失：提高正商品对分数，降低抽取负样本分数
  → 导出U、V及候选偏置
  → 单品查询：U_i·V_j − bias_j → 百分位排名 → TopK
```

这里没有把篮内其他商品联合编码成一个上下文向量，也没有直接优化U表内商品之间的余弦。`similar/hybrid`可以使用行为余弦融合，但它们不等于本分支`complement`的行为打分。

## 输入与工作目录

| 文件 | 必需字段 | 约定 |
| --- | --- | --- |
| `raw/item.csv` | `prod_id,prod_description` | 同一商品保留第一条描述；同时用于文本编码和冷启动品类识别 |
| `raw/order_item.csv` | `order_id,prod_id,dt` | `dt` 为 `YYYYMMDD`；可带 `user_id`，当前训练不使用 |

编码按字符串读取，保留前导零。`order_id` 定义完整购物篮，不能用用户ID代替。取数允许额外列和任意列顺序，输出时统一字段并移除额外列。

```text
<work-dir>/
├── raw/          item.csv、order_item.csv
├── processed/    baskets.jsonl、support.npz、manifest.json
├── embeddings/   item2index.json、index2item.json、item.feat1CLS、behavior_item.npz
└── results/      query_<ITEM_ID>.csv、item_similarity_<mode>.csv
```

| 公共选项 | 默认值 | 适用脚本 |
| --- | --- | --- |
| `--work-dir DIR` | 项目内 `outputs/` | 全部阶段和完整流水线 |
| `--env-file FILE` | `dataset/raw/.env` | 取数、完整流水线；纯 CSV 模式不读取 |
| `--model-dir DIR` | `dataset/m3e-base/` | 文本编码、完整流水线 |

选项可放在位置参数前后。脚本位于 `scripts/`，Python 实现位于 `src/item2vec/`，测试位于 `tests/`。原始数据、模型、凭据及 `outputs/` 不提交到 Git。

## 各阶段参数

### 1. fetch_data：准备商品与订单 CSV

商品和订单**分别必选一种来源**，两路可以混用；`run_pipeline.sh` 接受相同来源参数。

| 来源 | 商品参数 | 订单参数 | 读取方式 |
| --- | --- | --- | --- |
| CSV | `--item-input-csv FILE` | `--order-input-csv FILE` | 本地 CSV，无需凭据 |
| ODPS 表 | `--item-input-table NAME` | `--order-input-table NAME` | 全部行，不加隐含过滤 |
| SQL 文件 | `--item-input-query-file FILE` | `--order-input-query-file FILE` | 执行文件中的 SQL |

例如从商品表和订单 SQL 取数：

```bash
bash scripts/fetch_data.sh \
  --item-input-table your_catalog_table \
  --order-input-query-file /path/to/orders.sql \
  --env-file /path/to/.env --work-dir "$WORK_DIR"
```

ODPS 模式可从模板创建配置：

```bash
cp dataset/raw/.env.example dataset/raw/.env
```

| 配置名 | 兼容的小写名 | 说明 |
| --- | --- | --- |
| `ALI_ACCESS_ID` | `access_id` | 必填 |
| `ALI_SECRET_ACCESS_KEY` | `access_key` | 必填 |
| `ALI_PROJECT` | `project` | 必填 |
| `ALI_ENDPOINT` | `endpoint` | 可选，默认杭州 VPC Endpoint |

非空大写配置优先。日期窗口和商品池过滤由 SQL 显式控制，当前没有默认近30天查询。例如近90个日历日的订单 SQL：

```sql
SELECT order_id, user_id, prod_id, dt
FROM unisrec_raw_data
WHERE dt >= to_char(dateadd(getdate(), -89, 'dd'), 'yyyymmdd');
```

需要原商品池约束时，在商品 SQL 中加入 `lh_rec_gds_station_base_pool_tmp` 过滤。SQL 结果仍需满足上面的输入字段约定。两份数据均通过字段与非空校验后，才逐个原子替换输出；校验失败保留旧文件。

### 2. generate_embeddings：生成文本向量与索引

```bash
bash scripts/generate_embeddings.sh --model-dir /path/to/m3e-base --work-dir "$WORK_DIR"
```

输入 `raw/item.csv`，输出两份商品索引及768维 float32 文本向量 `embeddings/item.feat1CLS`。自动选择 CUDA 或 CPU，每批4条、最大512 Token，取 CLS 向量。商品索引与向量必须配套使用。

### 3. preprocess_baskets：生成训练购物篮

```text
bash scripts/preprocess_baskets.sh [MAX_BASKET_SIZE [MIN_CO_BASKET_ORDERS [MIN_BASKET_SIZE]]] [--work-dir DIR]
```

| 位置 | 参数 | 默认值 | 含义 |
| --- | --- | --- | --- |
| 1 | `MAX_BASKET_SIZE` | `30` | 最大不同商品数 |
| 2 | `MIN_CO_BASKET_ORDERS` | `5` | 商品在有效购物篮中的最低订单数；省略时可用 `MIN_ORDER_COUNT` 环境变量设置 |
| 3 | `MIN_BASKET_SIZE` | `2` | 最小不同商品数，范围为2至最大篮大小 |

例如 `30 5 3` 表示保留3～30件商品的篮，商品至少出现在5个有效篮中。也兼容 `--min-basket-size N`，但不能与第三个位置参数同时使用。

处理顺序：按完整订单分组 → 商品去重、移除目录外商品 → 按篮大小过滤 → 统计商品支持数、移除低频商品 → 再次检查最小篮大小。同细分类购物篮保留；全部订单中的已知商品仍计入商品总订单数。

输出 `processed/` 下的购物篮、支持数组和清单。清单记录参数、过滤计数及文件哈希：`small_baskets` 包括首次过滤中低于下限的空篮和单商品篮，`dropped_after_support` 记录移除低频商品后低于下限的篮数。原始订单、索引或过滤参数改变后，需重新预处理并训练。

`support.npz` 使用流式 ZIP 顺序写入，写入期间不对输出文件执行 `seek()` 回退，也不需要在本地临时目录中转。写入完成后仍通过清单校验，文件格式与现有 `np.load()` 读取方式兼容。

### 4. train：训练行为向量

```text
bash scripts/train.sh [VECTOR_SIZE MAX_BASKET_SIZE NEGATIVE EPOCHS [CANDIDATE_BIAS_WEIGHT]] [--work-dir DIR]
```

前四个位置参数须同时提供或全部省略，默认依次为 `128 30 15 10`；第五个为可选的候选偏置校正强度，默认 `0.4`。

| 参数 | 含义 |
| --- | --- |
| `VECTOR_SIZE` | 行为向量维度 |
| `MAX_BASKET_SIZE` | 必须与预处理清单一致，也用作完整购物篮上下文窗口 |
| `NEGATIVE` | 每个正样本的负采样数量 |
| `EPOCHS` | 训练轮数 |
| `CANDIDATE_BIAS_WEIGHT` | 候选偏置校正强度，必须是有限非负数；`0` 关闭校正 |

例如使用0.6的校正强度：

```bash
bash scripts/train.sh 256 30 15 10 0.6 --work-dir "$WORK_DIR"
```

也支持命名参数 `--candidate-bias-weight 0.6`，可配合默认训练参数使用，但不能与第五个位置参数同时指定。该值随模型记录在 `metadata.candidate_bias_weight` 中，查询与导出使用模型中保存的偏置；只重新导出不会更改旧模型的校正强度。它与查询/导出的文本权重参数不同。

`MIN_ORDER_COUNT` 环境变量默认5，须与预处理的第二个位置参数一致；最小篮大小自动读取清单，无需重复传入。例如预处理使用 `30 8 3`，训练时应设置 `MIN_ORDER_COUNT=8`。

训练只读取校验通过的 `processed/` 产物。使用 SGNS，负采样指数0.5，高频抽样关闭（`sample=0`），训练后预计算候选偏置。结果写入 `embeddings/behavior_item.npz`，包含输入/上下文向量、候选偏置、商品编码、总订单数和训练配置。

默认训练器仍为 Gensim。可使用新的 PyTorch SGNS，通过每篮损失权重平衡大篮贡献，排除当前整篮商品作为负样本：

```bash
bash scripts/train.sh 512 30 15 10 0.4 --trainer sgns \
  --basket-pair-budget 30 --negative-exclusion basket \
  --hard-negative-ratio 0 --seed 17 --device cuda --work-dir "$WORK_DIR"
```

`--basket-pair-budget 0` 关闭贡献平衡；正预算使用 `min(1, 预算/[篮大小×(篮大小−1)])` 加权正、负样本损失，保留全部合格商品对。`--pair-batch-size` 默认4096，控制正样本批大小。负样本分布只使用最终训练篮频次，指数0.5。困难负样本默认关闭；设置 `--hard-negative-ratio 0.2` 后，前2轮随机采样，从第3轮尝试12个随机＋3个模型高分负样本（总数15时），候选池默认128；可分别用 `--hard-negative-warmup`、`--hard-negative-pool-size` 调整。

指定 `--validation-orders CSV` 后，每轮验证，默认连续3轮主指标无明显改善时早停，保存最佳轮次；`--patience` 和 `--min-delta` 默认3、0.0001。验证订单须与训练订单无交集、时间更晚，训练原始CSV必须存在以供核验。没有验证文件时按指定轮数训练，保存固定轮次结果。训练历史为工作目录下 `history.json`，最佳验证结果为 `validation.json`。

`--device auto` 默认选择CUDA或CPU；显式CUDA不可用会报错。SGNS专属参数用于Gensim、重复参数或非法组合会报错；完整流水线在取数前先做训练参数预检。`--seed` 两种训练器均支持，默认1。`--processed-dir DIR` 可在独立输出目录训练时读取共享预处理产物，该目录的父目录须保留配套 `raw/order_item.csv`；输出目录仍须有配套商品索引。

推荐使用独立实验目录进行选模，先完成步骤2的商品索引准备：

```bash
EXP_DIR=/ml/output/experiments/balanced-sgns
python scripts/tune_item2vec.py prepare --work-dir "$WORK_DIR" \
  --experiment-dir "$EXP_DIR" --candidate-bias-weight 0.4 --device cuda
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase ablation
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase dimensions
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase confirmation
python scripts/tune_item2vec.py test --experiment-dir "$EXP_DIR"
```

校正强度应与本轮基线一致；省略时从现有行为模型的元数据读取，缺少元数据时必须显式指定。默认按最后14天测试、此前14天验证切分，仅对训练期预处理；同一订单整体归入最晚日期。实验先对照旧配方与A0～A3，再按验证结果决定维度和三种子确认。主指标为有行为商品的macro Recall@50，另检查全来源加权召回、低频召回和候选集中度；不读取测试标签选参数。无合格方案时 `selection.json` 记录 `keep_baseline`，不会强行升级。

实验输入、实现代码、冻结配置和模型均校验哈希，并阻止在同一实验中混用依赖版本。已完成的相同配置可复用；不完整的运行会明确报错，使用新实验目录重试。最终测试只评估冻结方案和基线，失败不自动改参数；各模型及 `test_report.json` 留在实验目录，不覆盖生产模型。本地检查范围见[验证记录](2026-11-02-balanced-sgns-validation.md)。详细规则和验收门槛见[实施方案](superpowers/plans/2026-11-02-balanced-sgns-and-model-selection.md)。生产全量重训应使用冻结配方、种子17和验证最佳轮次的中位数，另存模型，不再把已纳入训练的验证期用于早停。

### 5. query_similar：查询单个商品

```text
bash scripts/query_similar.sh ITEM_ID [TOPK [RECALL_MODE [TEXT_WEIGHT]]] [--work-dir DIR]
```

`ITEM_ID` 必填，`TOPK` 默认10；模式与权重见下一节。输出 `results/query_<ITEM_ID>.csv`，文件名中的不安全字符会替换为下划线。

### 6. export_similarities：批量导出

```text
bash scripts/export_similarities.sh [TOPK [BLOCK_SIZE [RECALL_MODE [TEXT_WEIGHT]]]] [--work-dir DIR]
```

`TOPK` 默认10，`BLOCK_SIZE` 默认512，只控制分块规模。输出 `results/item_similarity_<mode>.csv`。

两种查询均排除主商品自身。有效候选不足时可能少于 `TOPK` 条；无行为且文本向量为零的主商品不输出结果。

## 召回策略与结果字段

| 模式 | 默认文本权重 | 行为打分 | 适用场景 |
| --- | ---: | --- | --- |
| `complement` | `0` | 输入向量与候选上下文向量的内积，减去候选偏置，再转百分位排名 | 共购召回 |
| `similar` | `0.85` | 行为余弦，按商品订单数控制融合强度 | 语义相似 |
| `hybrid` | `0.60` | 同上 | 文本与行为融合，脚本默认模式 |

权重取值0～1，位置参数优先于 `TEXT_WEIGHT` 环境变量，最后使用模式默认值。`RECALL_MODE` 环境变量可设置未传入位置参数时的模式。

**`complement` 的三条路径：**

| 主商品情况 | 处理方式 | `recall_source` |
| --- | --- | --- |
| 有行为向量 | 使用自身行为向量召回 | `behavior` |
| 无行为，但有可靠参考 | 加权迁移同细分类参考商品的行为向量，再召回 | `behavior_transfer` |
| 参考不足、类别未知或缺少商品类别文件 | 按文本相似度回退 | `text_fallback` |

行为与迁移路径仅从有行为向量的商品中选候选；文本回退可选全部目录商品。迁移需要至少2个参考，最多取5个，文本余弦不低于0.8，以温度0.1的 softmax 权重平均输入行为向量。细分类从 `raw/item.csv` 描述中 `商品品牌为` 前的文本取得，因此应保留该文件。迁移向量不写回训练产物。

候选偏置是该候选与全部有效输入向量的最高10个内积均值乘以校正强度（默认0.4，不足10个取全部），在训练时预计算。非零文本权重会融合行为分数与文本余弦各自的百分位排名。旧模型缺少上下文向量时使用输入向量余弦；缺少偏置时使用未经校正的内积。重新训练才能使用新的负采样设置、上下文向量和偏置；已有产物可直接使用冷启动回退链。

**`similar/hybrid` 的订单置信度：**

```text
商品置信度 = min(商品总订单数 / FULL_CONFIDENCE_ORDERS, 1)
商品对置信度 = 两件商品置信度的较小值；任一件无行为向量时为0
实际行为权重 = (1 - TEXT_WEIGHT) × 商品对置信度
最终分数 = (1 - 实际行为权重) × 文本余弦 + 实际行为权重 × 行为余弦
```

`FULL_CONFIDENCE_ORDERS` 环境变量默认50，`complement` 忽略它。

| 输出字段 | 含义 |
| --- | --- |
| `master_prod_id` | 主商品编码 |
| `slave_prod_id` | 候选商品编码 |
| `similarity` | `similar/hybrid` 为融合余弦，`complement` 为候选百分位排名分数 |
| `recall_source` | 仅 `complement` 输出，取值见上表 |

`complement` 分数不是购买概率，不能沿用旧余弦分数阈值。行为迁移和文本回退是推断结果，不代表该主商品已经发生的共购。直接同单次数用于训练与离线评估，不参与推理排序。

## 完整流水线与后台运行

`run_pipeline.sh` 依次执行 **fetch_data → generate_embeddings → preprocess_baskets → train**，失败即停止，训练结束后不自动查询或导出。

```bash
bash scripts/run_pipeline.sh \
  --item-input-query-file /path/to/items.sql \
  --order-input-query-file /path/to/orders.sql \
  --env-file /path/to/.env --model-dir /path/to/m3e-base \
  --min-basket-size 3 --work-dir "$WORK_DIR"
```

完整流水线使用各阶段默认训练参数（128维、最大篮30、负采样15、10轮）；`--min-basket-size` 默认2，仅传给预处理。若需256维或其他训练参数，使用快速开始中的独立阶段命令。

完整流水线可传 `--candidate-bias-weight 0.6`，仅传给训练阶段，默认0.4。

任一耗时阶段均可使用 `nohup`：

```bash
mkdir -p "$WORK_DIR/logs"
nohup bash scripts/train.sh 256 30 15 10 --work-dir "$WORK_DIR" \
  > "$WORK_DIR/logs/train.out" 2>&1 &
tail -f "$WORK_DIR/logs/train.out"
```

不要同时对同一工作目录执行相同阶段。`nohup` 仅处理终端断开，不能防止重启或资源不足。

## 验证与报告

```bash
# 自动测试与脚本语法检查
PYTHONPATH=src python -m pytest -q
for script in scripts/*.sh; do bash -n "$script" || exit; done

# 按订单最晚日期切分训练、验证和测试期
python scripts/evaluate_item2vec.py split /path/to/order_item.csv \
  --train /tmp/train.csv --validation /tmp/validation.csv --test /tmp/test.csv

# 对相同训练期模型的导出结果进行对照
python scripts/evaluate_item2vec.py evaluate /tmp/test.csv \
  /path/to/baseline.csv /path/to/candidate.csv --top-k 50 \
  --item-csv /path/to/item.csv
```

时间切分默认最后14天测试、此前14天验证，其余训练，可用 `--test-days`、`--validation-days` 调整。分别对训练期订单运行预处理和训练，用验证期选型、测试期确认；不能用包含测试订单的全量模型评估留出效果。

评估输出商品对召回、按留出同单次数加权的召回、NDCG、主商品覆盖、不同候选数和前100候选占比。提供 `--item-csv` 时增加同/跨细分类加权召回，未知类别对不计入这两项分母。

实验方法、效果和限制见：

- [购物篮、Item2Vec 训练与候选打分验证](2026-10-29-item2vec-co-basket-results.md)
- [冷启动行为迁移验证](2026-10-29-cold-start-results.md)

最终工作目录应保留 `raw/`、`processed/`、`embeddings/` 及需要的 `results/`。实验模型、对照导出和缓存可在实验后清理；历史报告中的实验路径可能已被清理。
