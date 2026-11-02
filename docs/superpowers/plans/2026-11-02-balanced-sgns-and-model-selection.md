# 购物篮贡献平衡、负样本改进与验证集选模型实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. 本方案按当前会话顺序实施，交付前依代码审阅技能进行独立只读审查；提交、推送和生产模型替换另按用户授权执行。

**Goal:** 在保持 Item2Vec 行为共现学习与现有召回格式的前提下，控制大购物篮的训练贡献，降低当前篮内的假负样本，并用独立验证集选择训练轮次和配置。

**Architecture:** 保留 Gensim 训练入口作为对照和现有默认路径，新增可选择的 PyTorch SGNS 训练器。新训练器消费同一份预处理购物篮，使用两张商品向量表、加权负采样损失和逐轮验证；仍导出现有 `behavior_item.npz`，查询、导出和冷启动行为迁移复用现有实现。

**Tech Stack:** Python、NumPy、PyTorch Embedding/SparseAdam、pytest、现有 Bash 脚本；使用项目已有依赖，记录实际运行版本，不在本轮升级依赖。

---

## 1. 范围与现状

方案状态：已实现训练、采样、验证与实验管理入口，已完成本地验证；DLC CUDA吞吐与全量对照实验尚未执行。默认训练器仍为Gensim，实验不会覆盖生产模型。

基于当前提交 `1790c96` 核对：

- `training.py:train_item2vec` 使用 Gensim SGNS；完整窗口、`sample=0`、负采样指数 0.5、初始/结束学习率 0.002/0.0005，默认 8 个 CPU 工作线程。
- 已有输入向量和上下文向量。`complement` 的行为打分为输入与上下文内积减去候选偏置；非零文本权重会另行融合文本百分位排名。
- `evaluation.py` 已支持完整订单的时间切分和留出期指标，但没有逐轮验证、早停和自动选择最佳训练轮次。
- `load_processed_baskets` 当前把全部购物篮转换成 Python 字符串列表；新训练器应增加紧凑数组读取方式，保持旧调用行为。
- 当前预处理清单中的 `valid_basket_counts` 是支持度过滤前的计数。负采样分布必须从最终保留的训练篮重新计算，不能误用原始订单次数或过滤前次数。
- 上次截至 2026-10-29 的全量审计有 4,143,253 个训练篮、8,906 个可训练商品。11～30 件篮仅占订单约 3.70%，却贡献约 40.13% 的理论正样本对。这些是上次数据版本的结果，本轮按实际快照重算。

本轮保留原有 `30 5 2` 预处理口径。原始篮大小、80% 匹配完整度、最终支持度迭代过滤等属于另一组数据质量实验，不与本轮训练改造混在一起。

候选继续由模型向量打分；共现计数只用于负采样分布、评估标签和诊断，不作为召回排序分数。本轮不增加商品文本训练输入、多原型向量、用户塔或序列模型。

## 2. 推荐路线与可替代路线

推荐先做**全部正样本对的加权 SGNS**：保持每个训练篮和商品都参与学习，便于隔离质量变化；代价是正样本对数量没有减少，训练速度需要实测。

可替代路线是每个篮每轮只抽取一部分商品对：能减少计算，但会增加覆盖波动和采样实现复杂度。只有加权方案正确性和吞吐验证完成后，确有训练时长瓶颈，再单独实验这条路线。

直接删除大篮最简单，但可能丢失长尾商品的唯一共现机会，不作为本轮实现。

## 3. 数据隔离与实验目录

先按完整 `order_id` 切分，再仅用训练期执行预处理。订单内取最大 `dt` 决定所属时期，沿用 `split_orders`；不能将同一个订单的明细拆入不同集合。

- 最后 14 天为测试期，之前 14 天为验证期，更早数据为训练期。
- 若沿用 2026-07-31～2026-10-29 的快照：训练截至 10-01，验证 10-02～10-15，测试 10-16～10-29。实际日期由冻结快照计算，不硬编码。
- 支持度、有效商品集合、负样本分布、权重归一化系数与低频分组全部只来自训练期。
- 验证订单只提供评估标签，不用于补齐商品、排除历史假负样本、构造困难负样本或计算负样本频次。
- 现有全量模型已见过留出期订单，不能作为有效基线；旧配方必须在同一训练期重训。
- `prepare` 记录原始文件、索引、预处理产物和各时间切分文件的 SHA-256、行数、订单数与时间边界。
- 直接使用训练 CLI 加验证文件时，也必须检查订单交集和时间先后。扫描训练原始 CSV，检查训练订单号不在验证订单集合中，训练最大日期早于验证各订单最大日期的最小值；不使用验证明细行的最小日期误判跨日订单。

所有实验输出写入新实验目录，不覆盖现有工作目录的模型：

```text
outputs/experiments/balanced-sgns-<snapshot>/
  protocol.json                 # 数据哈希、指标、实验配置、固定评分参数
  split/{train,validation,test}.csv
  train/raw/order_item.csv      # 只含训练期，供现有哈希校验使用
  train/processed/              # 训练期唯一预处理结果
  runs/<run-id>/
    raw/item.csv                # 普通文件副本，用于类目/冷启动读取
                                # 训练直接读取共享 train/processed，不复制整份购物篮
    embeddings/                 # 独立模型；配套索引/文本向量为普通文件副本
      behavior_item.npz         # 有验证时保存最佳轮次，无验证时保存固定轮次结果
    config.json
    history.json
    validation.json
  selection.json                # 验证完成后冻结的配置、轮次、种子和模型哈希
  test_report.json              # 冻结配置与基线的最终测试报告
```

现有加载器会检查 `processed_dir.parent/raw/order_item.csv`。实验管理器直接传入共享的 `train/processed` 路径；单独调用训练脚本时用 `--processed-dir` 指定，输出仍位于各自的工作目录。使用普通文件副本，避免DLC挂载不支持链接；共享输入不得被训练器改写。完整流水线先复用训练CLI的无副作用参数预检，非法配置在任何取数/生成/预处理写入之前报错。

## 4. 购物篮贡献平衡

设最终保留的购物篮 b 含 k 个不同训练商品：

```text
P_b = k × (k − 1)               # 有方向的正样本对数量
w_b = min(1, C / P_b)           # C=30 为首轮实验值
```

`C=0` 表示关闭贡献平衡，此时所有权重为 1。C 是损失贡献预算，不是删单阈值，也不是强制抽取 30 对。

| 篮大小 | 原始对数 | C=30 时每对权重 | 未归一化的整篮正样本权重之和 |
| --- | ---: | ---: | ---: |
| 2 | 2 | 1 | 2 |
| 5 | 20 | 1 | 20 |
| 10 | 90 | 1/3 | 30 |
| 20 | 380 | 约0.079 | 30 |
| 30 | 870 | 约0.034 | 30 |

每个正样本及其 K 个负样本构成一个损失单元：

```text
ell(i,j) = softplus(−u_i·v_j) + sum_n softplus(u_i·v_n)
Z = sum_b(P_b × w_b) / sum_b(P_b)
loss_batch = mean(w_b × ell(i,j)) / Z
```

权重同时作用于正项和该正样本对应的负项，避免只减弱正项却保留全部负项。Z 是训练集固定系数，使关闭/开启权重时损失尺度可比；不能在每个小批内用 `sum(weights)` 归一化，否则同一篮或同一长度的批次会抵消权重。

完整计算参考，供测试逐项比对：

```python
import numpy as np
import torch.nn.functional as F

def basket_weights(sizes, budget):
    sizes = np.asarray(sizes, dtype=np.int64)
    pairs = sizes * (sizes - 1)
    weights = np.ones(len(sizes), dtype=np.float64)
    if budget:
        weights = np.minimum(1.0, budget / pairs)
    normalizer = float((pairs * weights).sum() / pairs.sum())
    return weights, normalizer

def weighted_sgns_loss(positive_logits, negative_logits, weights, normalizer):
    per_pair = F.softplus(-positive_logits) + F.softplus(negative_logits).sum(dim=1)
    return (weights * per_pair).mean() / normalizer
```

公开入口须在调用参考计算前拒绝非法篮大小、非有限/负预算和空训练集。每个篮的有方向商品对每轮覆盖一次，不生成自身对。

## 5. 紧凑数据与训练器

`sampling.py` 定义 `BasketTable(items, offsets)`：items 为连续 `int32` 商品索引，offsets 为 `int64` 篮边界，篮数为 `len(offsets)-1`。`load_processed_baskets(..., compact=True)` 逐行校验 JSONL 后直接填充紧凑数组，不先构建全部 Python 字符串列表；默认 `compact=False` 保留现有返回形式。

- 仅为最终训练篮内的商品创建可训练向量，建立 active index 与目录 index 的双向映射。
- 从紧凑数组重算最终训练商品频次；`order_counts` 仍按旧契约保存原始训练期订单计数，不能用负样本频次覆盖它。
- 使用输入/上下文两张 `nn.Embedding(..., sparse=True)`，优化器为 SparseAdam。输入初始化为 `[-0.5/d, 0.5/d]` 均匀分布，上下文初始化为零。
- 首轮使用 FP32、单 CPU 或单 CUDA；不同时引入混合精度、分布式训练或新正则项。初始/结束学习率先固定 0.002/0.0005，按计划训练步数线性调度；若小规模稳定性验证失败，先修正并冻结所有 SGNS 对照共同采用的设置。
- 每轮打乱篮顺序；在最多 256 个篮的块内使用按长度缓存的商品对位置模板生成有界数组，再打乱块内商品对，切成批次。禁止把全部正样本对展开为 Python 元组列表，也避免逐商品对 Python 训练循环。
- 商品对顺序、随机负样本与困难候选池使用独立的随机数流，防止打开困难采样时连正样本顺序也随之变化。
- `pair_batch_size=4096` 为起始值；正式实验前记录正样本对/秒、单轮耗时、峰值 CPU/GPU 内存，再确定实际批大小。加权方案不承诺减少计算量。
- 同一环境同一种子应可复现实验配置和 CPU 小样本结果；不同硬件及 Gensim 多线程不承诺逐位一致，正式结果通过多种子复核。

## 6. 负样本改进

### 6.1 随机负样本与当前整篮排除

```text
q(j) ∝ final_train_occurrences(j)^0.5
```

只从训练期 active 商品抽取，有放回，负样本之间允许重复。对每个正样本 (i,j)，排除当前完整训练篮中的所有商品，包括 i、j 及其他伙伴，而不只是排除 j。

注意：这只避免把**当前篮内已知正商品**当负样本，不能声称解决所有假负样本。相同商品在其他购物场景中仍可能是合理搭配。

实现采用批量抽样加掩码重采样，最多 8 轮；仍未填满的槽位按该篮条件化后的 q 直接补采样。若该篮外没有正概率商品，明确报错，不能死循环、静默减少负样本数或悄悄放回正商品。

### 6.2 少量困难负样本，作为独立可关闭实验

- 总负样本数保持 15；`hard_negative_ratio=0.2` 时使用 12 个随机负样本和 3 个困难负样本。
- 前 2 轮只用随机负样本；从第 3 轮开始混合。
- 按 q 抽取每个正样本的 128 个候选，排除当前篮全部商品后，用当前 `u_i·v_candidate` 选择较高分者。候选打分必须 detach；对被选中的负样本计算正常 SGNS 梯度。
- 困难候选不足时用合法随机负样本补齐；记录实际困难样本比例及补齐次数。
- 不把“同类目”“文本相似”直接定义为负标签，不读取验证/测试期的商品对。
- 不使用负采样概率修正来伪装成全量 softmax；本方案训练目标仍是 SGNS。混合困难样本会改变训练分布，收益必须单独评估。
- 默认关闭困难负样本。只有A3在初轮同维度对照中按第7节的守门条件、主指标及NDCG并列规则优于A2，才进入后续配方筛选；最终仍须满足验收条件。

## 7. 验证、早停与模型选择

### 7.1 固定评价对象

行为主实验统一使用 `complement`、文本权重 0、现有候选偏置方法。校正强度在实验前冻结为基线实际值，省略时从基线元数据读取，缺少元数据必须显式传入，示例为 0.4；本轮不与训练配置一起调这个参数。

每轮调用生产共用的向量打分/排序逻辑，使用 FP32 和相同的并列排序规则；不为训练验证另写一种余弦分数。训练期 active 商品集合在各 SGNS 实验间固定，因为加权不删篮、不删商品。

评价保留两个明确分母：

1. **warm 核心指标**：验证订单只保留训练期 active 商品，再构建至少含两个 active 商品的篮及真实商品对；用于隔离行为模型质量。
2. **全来源指标**：沿用现有 `evaluate_predictions` 在原始验证订单上的口径；无行为预测的来源仍计入分母，作为整体覆盖和加权召回守门指标。不能把 warm 指标与旧全来源指标直接比较。

warm 主指标定义：

```text
warm_macro_recall_at_50
  = mean_source(该来源 Top50 命中的不同真实目标数 / 该来源不同真实目标总数)
```

这不同于现有跨来源累加分子分母的 `pair_recall_at_k`，需要新增，不改变现有指标含义。

同时报告：整体 weighted Recall@50、warm NDCG@50、低频来源 macro Recall@50、跨细分类加权 Recall@50、来源覆盖、不同候选数、Top100 候选占比。低频来源定义为最终训练篮出现次数小于 20 的 active 商品，分组只算一次且固定；没有可评估来源时报告 `null` 与样本量，不报告 0。

固定验证标签在训练开始前准备一次，后续只更新模型预测；避免每轮重复扫描验证 CSV 和构造真实商品对。缓存必须绑定数据哈希、目录索引和评价 cohort，不能跨快照复用。

### 7.2 最佳轮次

- `epochs=10` 为首轮最大轮数，每轮验证一次。
- 主指标改善超过绝对值 `0.0001` 才更新最佳模型并重置等待计数；连续 3 轮未改善则停止。
- 保存的是最佳验证轮次的向量，不是最后一轮。每轮日志记录 loss、实际样本数、各篮大小有效贡献、负样本重采样/困难样本比例、学习率、训练/验证耗时和所有验证指标。
- 最佳模型和历史使用临时文件加原子替换；写出失败不得覆盖已有最佳产物。
- 不以训练 loss 下降、Top100 占比下降或单品案例更“多样”作为单独选模依据。

### 7.3 建议验收门槛

以下是本轮建议的判定门槛，不是收益承诺，写入 `protocol.json` 后不得根据测试集修改：

- 相对同后端 A0，三种子平均 warm macro Recall@50 提升至少 3%。A0 为 0 时改为比较绝对差并报告，不能计算相对提升。
- 全来源 weighted Recall@50 相对 A0 和同训练期 G0 各自下降不超过 1%；同时报告绝对百分点。
- 训练期 active 商品覆盖不减少；低频来源 macro Recall@50 相对 A0 下降不超过 1%。
- Top100 候选占比不高于 A0；必须同时查看召回指标，避免单纯抑制热门商品。
- 低频来源不足20个或没有低频标签时不通过守门；若相对G0的差异处于新方案的多种子波动范围内，标为证据不足，不自动宣布优于旧配方；若无候选通过门槛，保留现有配方。

对通过守门条件的候选，按三种子平均 warm macro Recall@50 选择；差异小于 0.0001 时依次比较 warm NDCG、较小维度、较早轮次。记录每个种子结果及均值/标准差，不只展示最好的一次。

## 8. 对照实验矩阵

首轮固定 512 维、负样本 15、指数 0.5、最大 10 轮、种子 17，先确认正样本预算 30；不做全参数笛卡尔积。

| 编号 | 训练器 | 贡献预算 | 负样本排除 | 困难比例 | 用途 |
| --- | --- | ---: | --- | ---: | --- |
| G0 | 现有 Gensim | 关闭 | 保持原实现 | 0 | 同训练期旧配方基线，固定10轮 |
| A0 | PyTorch SGNS | 关闭 | 只排除正目标 | 0 | 新后端基础对照 |
| A1 | PyTorch SGNS | 30 | 只排除正目标 | 0 | 隔离购物篮贡献平衡 |
| A2 | PyTorch SGNS | 30 | 当前整个篮 | 0 | 隔离当前篮假负样本排除 |
| A3 | PyTorch SGNS | 30 | 当前整个篮 | 0.2 | 隔离困难负样本 |

G0 与 A0 的优化器、实现细节不同，不承诺数值等价；不能将跨后端变化全部归因于贡献平衡。A0～A3 使用相同优化器、初始化、学习率计划、数据和固定评分设置。

先完成这5个对照并检查质量与吞吐，再决定是否扩大实验。对验证阶段较好的训练配方补充 128、256 维，512 维复用已有结果。最后对候选配方及 A0 增加种子 42、2026，合并种子 17 的结果做三种子确认。若候选不能超过同训练期 G0 的整体指标守门要求，不替换现有配方。

G0、A0～A3、两个额外维度和最后的四个多种子补充合计至多约 11 次训练；早停可减少轮次，不预设训练时长。预算 60、负样本指数/数量调整及新学习率搜索留作后续单独实验。

## 9. 测试集与全量重训

完成验证选配方、种子确认后写出 `selection.json`，包含选择理由、冻结配置、各确认种子对应的最佳模型路径/哈希、各种子最佳轮次及其中位数。不从三个种子中挑一个成绩最好的作为唯一结果。

- 仅对冻结的获选方案与A0各自的三个种子最佳模型，以及G0基线，运行一组最终测试；报告各种子结果和均值/标准差。训练循环、实验调参与选模模块不得读取测试订单。
- 测试结果不满足守门要求则报告失败并保留旧方案，不在同一测试窗口反复调参择优；后续改动使用新的时间窗口确认。
- 测试报告完成后，可以另建生产目录，用冻结超参及各种子最佳轮次的中位数全量重训；生产种子预先固定为17，不从测试成绩选种子。全量训练不再使用已纳入训练的验证文件早停。
- 全量模型与留出期获选模型分别保存；不能将全量重训后的模型重新在同一测试集上评分并宣传为独立测试效果。
- 实验不自动替换 `/ml/output/output/embeddings/behavior_item.npz`，不自动推送或部署。

## 10. 文件职责与兼容性

| 文件 | 改造内容 |
| --- | --- |
| 新增 `src/item2vec/sampling.py` | BasketTable、有限内存商品对生成、篮权重、负样本分布和整篮/困难采样 |
| 新增 `src/item2vec/sgns.py` | 两张稀疏向量表、加权损失、训练循环、验证回调、最佳模型返回 |
| 新增 `src/item2vec/model_selection.py` | 实验协议、隔离目录准备、逐阶段实验、跨种子比较、冻结配置和最终测试 |
| 修改 `src/item2vec/preprocessing.py` | 仅给加载器新增紧凑读取选项，保持预处理规则及清单版本不变 |
| 修改 `src/item2vec/io.py` | 复用的原子JSON写入，拒绝非有限值并保护原文件 |
| 修改 `src/item2vec/evaluation.py` | 可复用验证标签、新增 macro/低频来源指标，保留现有指标分母与 CLI 行为 |
| 修改 `src/item2vec/training.py` | 训练器分发、新参数校验、验证入口、产物元数据；保留现有 Gensim 函数契约 |
| 新增 `scripts/tune_item2vec.py` | `prepare/run/test` 薄 CLI，调用 model_selection 模块 |
| 修改 `scripts/train.sh`、`scripts/run_pipeline.sh` | 新参数传递、非法组合检查，旧位置参数和默认流程仍可用 |
| 修改 `README.md` | 增加训练器、验证和实验命令的简明说明，链接详细方案 |
| 新增/修改对应测试 | 数据、采样、损失、梯度、最佳轮次、隔离、CLI 与导出兼容性 |

训练返回统一的 NumPy 输入/上下文矩阵供现有 `write_behavior_embedding` 保存。只将 active 商品散射回完整目录行，其他商品保持零向量。保持字段 `vectors/context_vectors/candidate_bias/item_ids/order_counts/metadata` 及目录顺序不变；可在 metadata 增加训练器、配置、最终训练频次摘要/哈希、最佳轮次、种子和数据指纹，不能改变旧字段含义。

非零向量判定、冷启动迁移与候选校正函数保持原语义。模型写出沿用现有“本地暂存、顺序复制到输出目录、原子替换”方式，保留 DLC 挂载兼容性；不新增失败后回退重试写法。

## 11. 拟新增 CLI 契约

旧命令继续可用，第五个位置参数仍是候选校正强度：

```bash
bash scripts/train.sh 512 30 15 10 0.4 --work-dir "$WORK_DIR"
```

| 参数 | 拟定行为 |
| --- | --- |
| `--trainer gensim\|sgns` | 默认 gensim；本轮实验显式使用 sgns |
| `--basket-pair-budget N` | sgns 默认0；非负整数，30为贡献平衡实验值 |
| `--negative-exclusion target\|basket` | sgns 默认basket；A0/A1显式使用target |
| `--hard-negative-ratio R` | sgns 默认0；有限数且 `0<=R<1`，困难数 `floor(negative*R)` |
| `--hard-negative-warmup N` | 默认2；非负整数 |
| `--hard-negative-pool-size N` | 默认128；正整数，且不少于计划困难样本槽位数 |
| `--pair-batch-size N` | 默认4096；正整数 |
| `--processed-dir DIR` | 仅训练脚本：读取共享预处理目录，保持独立输出 |
| `--validation-orders PATH` | sgns可选；指定后启用逐轮验证和最佳轮次保存 |
| `--patience N` | 验证时默认3；正整数 |
| `--min-delta N` | 验证时默认0.0001；有限非负数 |
| `--seed N` | 未指定时维持种子1；正式实验显式使用17/42/2026 |
| `--device auto\|cpu\|cuda` | sgns默认auto；显式cuda但不可用时直接报错 |

sgns专属参数显式用于gensim时应在读取大文件前报错，不能静默忽略。显式指定patience/min-delta但没有验证文件时也应报错。所有新命名参数检查缺值、重复、非法值；保留第五位置参数与同名命名参数的冲突校验。Bash空数组须兼容现有 macOS Bash 3.2。

以下是**实施完成后**的目标用法，当前不可直接执行。实验准备负责确保 run 目录中的原始订单与预处理都只属于训练期：

```bash
python scripts/tune_item2vec.py prepare \
  --work-dir "$WORK_DIR" --experiment-dir "$EXP_DIR" \
  --validation-days 14 --test-days 14 --candidate-bias-weight 0.4 --device cuda

python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase ablation
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase dimensions
python scripts/tune_item2vec.py run --experiment-dir "$EXP_DIR" --phase confirmation

python scripts/tune_item2vec.py test --experiment-dir "$EXP_DIR"
```

confirmation阶段完成后冻结selection；test必须验证selection与数据/模型哈希，不从测试成绩重新挑选候选。已经完成的同一输入测试可幂等读取报告，不覆盖成另一组“最好结果”。

单次A2训练的目标用法：

```bash
bash scripts/train.sh 512 30 15 10 0.4 \
  --trainer sgns --basket-pair-budget 30 --negative-exclusion basket \
  --hard-negative-ratio 0 --pair-batch-size 4096 \
  --processed-dir "$EXP_DIR/train/processed" \
  --validation-orders "$EXP_DIR/split/validation.csv" \
  --patience 3 --seed 17 --device cuda \
  --work-dir "$EXP_DIR/runs/A2-seed17"
```

## 12. 按依赖顺序实施

### Task 1：冻结评估协议，扩展可复用指标

**Files:** `src/item2vec/evaluation.py`、`tests/test_evaluate_item2vec.py`。

- [x] 先添加以下macro口径测试并确认失败；新增键名为`macro_recall_at_k`，原有`pair_recall_at_k`等不改名、不改定义。

```python
def test_macro_recall_gives_each_source_equal_weight():
    import pandas as pd
    import pytest
    from item2vec.evaluation import evaluate_predictions
    orders = pd.DataFrame([
        ('1', 'A'), ('1', 'B'),
        ('2', 'A'), ('2', 'C'),
        ('3', 'A'), ('3', 'D'),
    ], columns=['order_id', 'prod_id'])
    predictions = pd.DataFrame([
        ('A', 'B'), ('B', 'A'), ('C', 'A'), ('D', 'A'),
    ], columns=['master_prod_id', 'slave_prod_id'])
    result = evaluate_predictions(orders, predictions, top_k=1)
    assert result['macro_recall_at_k'] == pytest.approx((1/3 + 1 + 1 + 1) / 4)
    assert result['pair_recall_at_k'] == pytest.approx(4/6)
```

- [x] 将真实商品对准备与预测评估拆开，`evaluate_predictions`仍作为兼容包装；macro计算为每个来源的召回率平均。新增测试覆盖缺失预测计0、低频分组为空返回null、warm与全来源分母不同、缓存与未缓存结果一致。
- [x] 执行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_evaluate_item2vec.py`；既有测试及新增测试全部通过。
- [x] 冻结协议中的指标、对照、评分参数和验收门槛，后续测试结果不反向修改它们。

### Task 2：紧凑购物篮与贡献权重

**Files:** 新增`src/item2vec/sampling.py`、`tests/test_sampling.py`；修改`preprocessing.py`、`tests/test_preprocessing.py`。

- [x] 用`BasketTable(items, offsets)`增加紧凑存储测试，比较默认列表与紧凑读取的每一个篮和支持度数组完全相等，并保持现有哈希/越界/支持度校验。
- [x] 先写以下权重测试，运行并确认导入/接口失败，再按第4节参考实现补齐校验及计算。

```python
def test_budget_caps_basket_mass_without_dropping_pairs():
    import numpy as np
    from item2vec.sampling import basket_weights
    sizes = np.array([2, 5, 10, 20, 30])
    weights, normalizer = basket_weights(sizes, 30)
    pairs = sizes * (sizes - 1)
    np.testing.assert_allclose(pairs * weights, [2, 20, 30, 30, 30])
    assert normalizer == np.sum(pairs * weights) / np.sum(pairs)
    off_weights, off_normalizer = basket_weights(sizes, 0)
    np.testing.assert_array_equal(off_weights, np.ones(5))
    assert off_normalizer == 1
```

- [x] 实现有界块商品对生成；在含`[0,1]`和`[1,2,3]`的测试数据中，验证一轮准确产生2+6个有方向商品对、没有自身对、每个pair携带正确篮编号/权重。相同seed/epoch顺序一致，改变seed只改变顺序而不改变多重集合。
- [x] 执行采样与预处理测试；以小型快照测量紧凑数组占用和生成峰值内存，确认不会展开全数据商品对。

### Task 3：安全负采样与困难负采样

**Files:** `sampling.py`、`tests/test_sampling.py`。

- [x] 明确接口：`NegativeSampler(counts, exponent, seed)`；`sample(source_ids, basket_ids, baskets, k, exclusion, hard_ratio, model, epoch, warmup_epochs, pool_size, target_ids=None)`返回`[batch,k]`合法active商品索引和采样诊断。`model=None`只允许困难采样未启用/尚处预热期。
- [x] 先写以下整篮排除测试并运行确认失败；counts仅表示最终训练出现次数。

```python
def test_negatives_exclude_every_item_in_the_current_basket():
    import numpy as np
    from item2vec.sampling import BasketTable, NegativeSampler
    baskets = BasketTable(np.array([0, 1, 2], dtype=np.int32),
                          np.array([0, 3], dtype=np.int64))
    sampler = NegativeSampler(np.array([100, 80, 60, 40, 20]), .5, seed=17)
    negatives, stats = sampler.sample(
        source_ids=np.array([0]), basket_ids=np.array([0]), baskets=baskets,
        k=15, exclusion='basket', hard_ratio=0, model=None,
        epoch=1, warmup_epochs=2, pool_size=128)
    assert negatives.shape == (1, 15)
    assert set(np.asarray(negatives).ravel()) <= {3, 4}
    assert stats['negative_count'] == 15
```

- [x] 按第6节实现条件随机采样；测试只剩一个合法候选仍返回K个样本、无合法候选明确失败、重复允许、零频商品不被抽中、重采样8轮后正确进入条件补采样。
- [x] 用固定候选池和固定向量测试困难采样：前2轮困难数0，第3轮K=15、R=.2时困难槽位3；选中的是合法池内高分商品，篮内高分商品仍被排除，打分选择无梯度而最终负样本损失有梯度。
- [x] 在指定人工频次下进行固定seed分布测试，确认随机采样比例近似`count**.5`的条件概率；设置合理统计容差，不能逐次要求恰好等于理论频率。
- [x] 执行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_sampling.py`。

### Task 4：PyTorch SGNS与逐轮最佳模型

**Files:** 新增`src/item2vec/sgns.py`、`tests/test_sgns.py`。

- [x] 定义`weighted_sgns_loss`为第4节公式；定义`train_sgns(baskets, item_count, config, validate_epoch=None)`，返回`vectors`、`context_vectors`和`metadata`字典。validate_epoch接收目录顺序的NumPy输入/上下文矩阵及epoch，返回第7节指标；未提供时固定训练epochs并明确记录`selection_mode=fixed_epochs`。
- [x] 先写解析损失和梯度测试：所有logits为0、K=2、weights为1时损失为`3*log(2)`；只改变某个篮权重时，正项和负项梯度按同一比例变化，Z使用全局值。
- [x] 按第5节实现两张稀疏Embedding、SparseAdam、训练循环及active映射；测试未参与训练的目录行仍为0，两张向量都为有限数，训练商品集合与未加权对照一致。
- [x] 用验证回调返回主指标`[0.10,0.12,0.11,0.105,0.10]`，patience=3：断言最佳epoch=2、在epoch5停止，返回/保存epoch2对应向量而不是epoch5。另测min_delta、首轮保存、NaN指标报错、没有可评价warm来源时报错而不是按loss选模。
- [x] 训练/验证异常、非有限loss或梯度、模型写出失败均停止运行且保留已有最佳产物，不静默换优化器、设备或训练目标。
- [x] 用多个重复篮训练CPU小数据，验证留出真实搭配的得分/召回相对初始化有提升；不以随机初始化下“一定单步loss下降”作为测试条件。
- [x] 本地执行SGNS、采样和人工数据闭环测试。
- [ ] DLC再执行一个CUDA小样本完整训练及保存/加载检查。

### Task 5：训练CLI、产物兼容与流水线

**Files:** `training.py`、`scripts/train.sh`、`scripts/run_pipeline.sh`、`tests/test_training.py`、`tests/test_scripts.py`、`tests/test_candidate_bias.py`。

- [x] 先用脚本stub测试第11节所有新参数的传递、默认gensim、旧四/五位置参数不变、重复缺值及跨后端非法组合在大文件读取前失败。
- [x] 将新参数分发给sgns；保留`train_item2vec`旧签名和返回值供现有调用/测试使用。Torch新模块在选择sgns后加载，不能让gensim和帮助命令依赖新训练模块的初始化。
- [x] 验证文件启用时做第3节数据隔离检查，准备一次标签，调用现有`rank_items`及候选偏置计算进行逐轮验证。
- [x] 统一调用现有artifact writer保存获选输入/上下文向量、原始训练期order_counts及新增metadata；有验证时不得在循环结束后用最后一轮覆盖最佳模型。
- [x] 对人工小数据运行旧查询和导出，验证新产物能被`load_trained_artifacts`读取、商品ID和维度匹配、打分与验证一致、无行为商品保持迁移/文本回退链。
- [x] 对写出注入不支持seek的目标流和失败流，确认目标只接受顺序复制、旧产物未损坏、tmp清理正确。
- [x] 执行训练、脚本、候选校正及召回相关测试；对所有脚本执行`bash -n`，检查macOS默认Bash的空数组参数路径。

### Task 6：实验管理与防泄漏选模

**Files:** 新增`model_selection.py`、`scripts/tune_item2vec.py`、`tests/test_model_selection.py`；修改`tests/test_evaluate_item2vec.py`。

- [x] `prepare`只准备切分、训练预处理、共享输入和protocol；`run`仅接收train/validation路径；`test`只有在selection冻结且模型/数据哈希匹配时才读取test。
- [x] 先写跨时间完整订单、训练/验证订单号重叠、训练源哈希变动、低频分组只用训练频次等测试。用文件读取拦截器使任何prepare/run训练路径读取test内容立即失败；允许prepare按原始订单生成test文件，但之后训练模块不得打开它。
- [x] 实现A0～A3、维度及三种子确认的固定配置；所有结果记录绝对指标、相对差、样本数、配置与模型哈希。维度阶段只补缺少的run，不覆盖已完成模型。
- [x] 用人工指标测试：最高macro但weighted守门失败不能胜出；低频评估缺失不能自动通过；无方案合格保留基线；相近macro按第7节顺序选择；不能只取最佳seed。
- [x] 测试只保存一次冻结选择的报告；重复相同输入读取既有报告，数据或selection哈希变更明确拒绝复用。
- [x] 执行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_model_selection.py tests/test_evaluate_item2vec.py`。

### Task 7：完整检查、吞吐验证与实验交付

**Files:** `README.md`、本方案；实验数据和报告只写入被忽略的`outputs/`。

- [x] README只补新入口、默认行为、验证选模语义及指向本方案的链接；注明目标命令已在实现后验证，不能把预期功能写成已上线。
- [x] 执行全套测试 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider`、全部shell语法检查及`git diff --check`，记录实际结果，不沿用此前333项的旧结论。
- [x] 本地CPU小样本及人工数据实验闭环完成；记录见`docs/2026-11-02-balanced-sgns-validation.md`。
- [ ] 在实际DLC环境验证设备、依赖版本、内存和吞吐；GPU全量训练时长由此实测，不以本地测试替代远端验证。
- [ ] 冻结快照与协议，依次运行G0/A0～A3、维度筛选、多种子确认；任何失败先定位，不根据测试集挑新参数。
- [ ] 交付验证对照表、冻结selection、最佳轮次模型、最终测试报告、明确的通过/未通过结论及全量重训配置。记录各篮大小加权贡献，确认平衡目标实际发生且没有损失训练商品覆盖。

## 13. 参考与验证边界

SGNS/Item2Vec框架参考：[Item2Vec论文](https://arxiv.org/abs/1603.04259)。稀疏Embedding和优化器选择参考项目所列版本的[PyTorch Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)、[SparseAdam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SparseAdam.html)官方文档。

权重预算30、困难比例20%、候选池128、早停耐心3以及验收门槛均为本项目拟定的实验设置，不能理解为论文保证或已经验证的最佳值。本地测试与小数据闭环只能证明实现正确性；全量指标提升、DLC CUDA运行及生产替换仍须另行验证。
