# Attention Item2Vec Implementation Plan

**Goal:** 在独立分支实现整篮注意力、商品余弦联合训练及兼容旧模型的完整召回链。

**Architecture:** 新建独立attention训练器，使用现有BasketTable及NegativeSampler；共享归一化商品向量参与上下文预测及直接商品对目标。模型元数据显式声明cosine，推断选择同空间余弦。保留旧训练器与旧产物读取语义。

**Tech Stack:** PyTorch、NumPy、现有Python/Bash命令行、pytest。

按用户授权在当前会话直接执行，不启动真实数据后台训练，不自动提交或推送。

- [x] 基线检查：运行现有全部测试，确认Git状态及新分支。
- [x] 模型和样本：新增src/item2vec/attention.py及tests/test_attention.py。先验证目标移除、整篮负采样、无序和padding不变性、共享余弦损失、训练与导出一致，再实现最小模型及训练循环。迭代完整及随机非空上下文，每个目标内平均两视图与全部正商品对。
- [x] 训练集成：修改training.py、evaluation.py、scripts/train.sh、scripts/run_pipeline.sh；先增加CLI及端到端测试，再接入attention选项、逐轮验证和最佳轮保存。注意力专属选项不用于旧训练器，拒绝无效组合。
- [x] 推断集成：修改inference.py，保持旧六元组加载接口兼容；先验证显式cosine元数据、原始余弦排序、冷启动行为迁移、旧产物路径，再增加score路由。不使用旧候选偏置。
- [x] 文档及验证：更新README及设计状态；运行窄测试、全回归、Bash语法检查和独立CPU小数据训练/导出验证，审阅实现与规范差异，修复重要问题。验收覆盖已知共购组组内余弦高于组间。

验收证据：docs/2026-11-04-attention-validation.md；全量430测试通过，合成数据训练和余弦查询/导出通过；代码审阅发现的篮大小重复扫描已修复。真实数据后台训练未启动。
