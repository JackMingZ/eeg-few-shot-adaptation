# 结果目录索引

所有目录均位于本地 `<project-root>`。数据和训练产物体积较大，已从 Git 默认跟踪排除，但保留在本机。

|目录|性质|如何阅读|
|---|---|---|
|results/ 顶层|历史合成演示|97.58%/100% 等不是实际运动想象结果|
|results/real/recording_smoke|真实 A01T 运行检查|只验证 CUDA、梯度和重载，不计为跨被试结果|
|results/real/three_subject_pilot|输入除100的早期失败批次|3目标×3种子；源模型全部一类输出；完整保留|
|results/real/source_only_diagnostic_repro|源数据比例诊断|只加载源被试2、3；不是目标测试结果|
|results/real/three_subject_microvolt|保留微伏的三被试探索|3目标×3种子；每轮仅1人训练、1人验证；不能替代全数据结果|
|results/real/source_count_three|源被试数量补充实验|固定目标1、验证2，源训练3/4/5，3种子；只供与嵌套源集合比较|
|results/real/nine_subject_exploratory|完整九被试探索矩阵|目标9人×种子3个；仅有 metrics.json 的组合才完成|

每个实验根目录的 REPORT.md、all_metrics.csv、paired_gains.csv 和 cost_summary.csv 自动汇总已完成组合。不要跨输入比例或源集合直接混合平均。各 target 目录含 config.json、splits.json、code/ 快照、权重、逐试次预测及曲线。

audit.json 是独立一致性复核结果；工程检查通过不等于统计显著或方法有效。STUDY_LOG.md 记录研究解释和当前局限。

主实验的环境、数据哈希、预处理、划分、命令、逐被试结果、成本、审计和局限汇总在 [DETAILED_EXPERIMENT_LOG.md](DETAILED_EXPERIMENT_LOG.md)。
