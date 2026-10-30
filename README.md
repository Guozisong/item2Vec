# Item2Vec 商品共购召回

用 M3E/BERT 编码商品文本，用 Item2Vec 学习订单购物篮中的商品关系。共购模式 `complement` 按 **自身行为 → 相近商品行为迁移 → 文本回退** 召回，并标明结果来源。也支持侧重语义的 `similar` 和融合模式 `hybrid`。

## 快速开始

要求 Python 3.9+，将本地 M3E-base 模型放入 `dataset/m3e-base/`，然后在项目根目录执行：

```bash
python -m pip install -r requirements.txt
WORK_DIR=/ml/output/output

# 1. 准备商品和订单数据；也可使用 ODPS 表或 SQL 文件
bash scripts/fetch_data.sh \
  --item-input-csv /path/to/items.csv \
  --order-input-csv /path/to/orders.csv --work-dir "$WORK_DIR"

# 2. 生成文本向量与商品索引
bash scripts/generate_embeddings.sh --work-dir "$WORK_DIR"

# 3. 最大篮30件、商品最低支持5单、最小篮3件
bash scripts/preprocess_baskets.sh 30 5 3 --work-dir "$WORK_DIR"

# 4. 行为向量256维、最大篮30件、负采样15、训练10轮
bash scripts/train.sh 256 30 15 10 --work-dir "$WORK_DIR"

# 5. 查询单品；将 ITEM_ID 替换为商品编码
bash scripts/query_similar.sh ITEM_ID 50 complement 0 --work-dir "$WORK_DIR"

# 6. 批量导出每个商品的 Top-50
bash scripts/export_similarities.sh 50 512 complement 0 --work-dir "$WORK_DIR"
```

已有原始数据可从第2步开始；已有配套文本向量与索引可从第3步开始。所有阶段使用同一 `--work-dir`。查询与导出默认模式是 `hybrid`，共购场景请显式指定 `complement`。

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

- [购物篮、Item2Vec 训练与候选打分验证](docs/2026-10-29-item2vec-co-basket-results.md)
- [冷启动行为迁移验证](docs/2026-10-29-cold-start-results.md)

最终工作目录应保留 `raw/`、`processed/`、`embeddings/` 及需要的 `results/`。实验模型、对照导出和缓存可在实验后清理；历史报告中的实验路径可能已被清理。
