# CPS 强弱模型协作实验报告

## 摘要

本实验用一个通知分流 demo 检查 CPS 在真实 LLM 调用中的作用。CPS 程序保留运行状态，把需要模型判断的部分拆成明确的 effect，再把模型返回值接回程序流程。强模型、弱模型、本地校验器和 ProgramPatch 都在这个流程里工作。

通知分流任务读入一条通知，输出 `ignore`、`create_task`、`create_calendar_event` 或 `draft_reply`。这个场景便于观察三件事：强模型调用能否减少，弱模型是否能承担局部判断，模型经验能否写成可复用的程序补丁。

最终权威结果来自 `.cps-real-full-v11/`。安装后的 Program `v0002` 使用弱模型做语义匹配，再用本地工具校验和模板化输出。HeldoutTest 上，GeneralizedPatch 的 quality 为 `0.8375`，StrongDirect 为 `0.725`；strong calls/event 从 `1.0` 降到 `0.0`；语义 fast path 命中 `85/160` 条，hit rate 为 `53.125%`；false fast-path rate 为 `0.0`。

这支持一个具体判断：CPS 可以把强模型参与过的优化和审计结果转成程序结构，让后续请求更多地由弱模型和本地程序处理。相同模式也适合端云协同、手机 GUI Agent、强模型优化 skill 后交给弱模型执行、以及生产环境中的周期性强模型审计。

## Demo 场景

数据集共 400 条通知：

| 数据部分 | 数量 | 用途 |
|---|---:|---|
| ProfileTrain | 120 | 提供 profile evidence，供 optimizer 生成语义规则 |
| PatchValidation | 80 | 在安装前验证 patch |
| HeldoutTest | 160 | 评估安装后的程序 |
| AdversarialTest | 40 | 检查困难负例和误触发风险 |

实验比较了这些运行方式：

| 名称 | 做法 | 用途 |
|---|---|---|
| StrongDirect | 每条通知都交给强模型 | 强模型基线 |
| WeakOnly | 每条通知都交给弱模型 | 弱模型基线 |
| CPS-Unoptimized | 初始 CPS 程序，无语义 patch | 优化前程序 |
| ExactMemo | 只命中规范化后相同的文本 | 排除简单背诵 |
| GeneralizedPatch | 安装语义 ProgramPatch 后运行 | 测试 CPS patch 的运行效果 |
| NoSemanticWeak | 关闭弱语义匹配 | 隔离语义匹配贡献 |

## Optimizer 路径

v11 的 optimizer 不再接收 Rust 预先合成的 `candidate_plan`。Rust 只提供四类输入：

- profile evidence：来自 ProfileTrain 的语义 cluster、正例、反例和 canonical 输出。
- failure-cluster evidence：来自 `data/notification/optimizer_hard_negative_evidence.jsonl` 的 hard-negative 例子。
- validation constraints：输出类型、Rust regex 约束、negative guard 语义。
- schema：强模型必须返回的 semantic patch plan 结构。

真实 `gpt-5.5` 在 `phase=optimize`、`task_name=optimize_semantic_patch_plan` 下返回最终 semantic plan。该 raw response 就是最终 plan 来源：

- `optimizer_source: strong_model_generated_semantic_patch_plan`
- `raw_optimizer_response_is_final_plan: true`
- generated rules: 6
- generated negative guards: 5
- installed registry source: `optimizer_strong_model`

Rust 后续只做 schema validation、safety validation、gate validation 和 ProgramPatch rendering，不替换强模型生成的 rules 或 negative guards。

v10 曾暴露一个 adversarial false fast-path：`deadline passed and the work is complete` 被误匹配为 deadline task。v11 把这类失败加入 failure evidence，强模型生成了 `deadline_request_without_document_v1` 的 completed/past-deadline negative guard。最终 adversarial false fast-path rate 降回 `0.0%`。

## GeneralizedPatch 的运行路径

Program `v0002` 的主路径如下：

1. 弱模型执行 `semantic_fast_path_match`，判断当前通知是否匹配已知语义规则，并抽取 `kind`、`title`、`datetime_hint` 等槽位。
2. 匹配成功后，程序调用 `validator_apply`。校验内容包括规则 id、置信度、输出结构和 negative guard。
3. 校验通过后，程序调用 `template_emit` 生成最终 action 并返回。
4. 匹配失败或校验失败时，程序调用弱模型 fallback：`draft_action_from_event`。

这条路径把弱模型限制在较小的判断任务上。最终 action 由本地程序校验并生成，fast path 的行为可以从程序和 trace 中审计。

## Heldout 上的模型调用

HeldoutTest 共有 160 条通知。GeneralizedPatch 的实际调用结构如下：

| 路径 | 样本数 | 每条样本的模型调用 |
|---|---:|---|
| Fast path 命中 | 85 | 1 次弱模型语义匹配 |
| Fallback | 75 | 1 次弱模型语义匹配 + 1 次弱模型普通处理 |
| 强模型 | 0 | 没有触发 |

总弱模型调用数为 235，强模型调用数为 0。Fast-path hit rate 为 `53.125%`，其余样本走弱模型 fallback。强模型仍参与 StrongDirect 基线、PatchValidation 对照、optimizer 和 shadow audit；GeneralizedPatch 在 heldout 在线执行中没有逐条调用强模型。

## 结果

核心 heldout 指标如下：

| 指标 | GeneralizedPatch | StrongDirect | ExactMemo |
|---|---:|---:|---:|
| Quality | 0.8375 | 0.725 | 0.71875 |
| Strong calls/event | 0.0 | 1.0 | 1.0 |
| Weak calls/event | 1.46875 | 0.0 | 0.0 |
| Fast-path hit rate | 53.125% | 0.0% | 0.0% |
| False fast-path rate | 0.0% | 0.0% | 0.0% |

ExactMemo 在 heldout 上没有命中，GeneralizedPatch 命中 85 条。这说明 GeneralizedPatch 覆盖了语义相近的样本，而非复用完全相同的文本。

最终报告记录了 `program_version_advanced: true`、`patch_gate_accepted: true` 和 `experiment_passed: true`。Heldout 预测文件中的 `cps_generalized_patch` 使用 Program `v0002`，说明结果来自安装后的程序。

AdversarialTest 上，fast path 的 false action 为 0。风险主要来自 fallback：40 个困难负例中有 13 个被弱模型 fallback 错误地产生了行动，hard-negative false-action rate 为 `32.5%`。后续工作应优先加固 fallback 的拒绝能力。

## 成本和账本

最终 v11 报告的单次运行账本如下：

| 口径 | 数值 |
|---|---:|
| v11 report-run marginal API spend | 0.9237335 美元 |
| v11 optimize phase spend | 0.105235 美元 |
| v11 cache hit rate | 80.5801% |
| v11 model-call records | 1586 |
| v11 unknown phase records | 0 |
| v11 event-named by_run files | 0 |

v11 因为 optimizer evidence 和 generated plan 改变，语义匹配相关请求大量重新 miss。这个数字用于说明最终报告的调用归因和预算账本完整，不能当作冷启动全量实验成本。

按当前仓库内所有 `.cps-real*/budgets/spend.jsonl` 可见记录汇总，项目到目前为止记录到的累计边际 API 花费为 `13.32784850` 美元。正式财务口径仍应以 API 平台账单为准。

## Shadow Audit

Shadow audit 比较 GeneralizedPatch fast-path 输出和 shadow StrongDirect 输出。v11 检查了 93 个 fast-path hit，其中 29 个有语义差异，disagreement rate 为 `31.1828%`，critical disagreement 为 0。它是质量监控限制，不是本轮 gate failure。

## 限制

通知分流 demo 只覆盖一个受控任务。跨任务泛化、端云部署收益、自动 skill 生成和更复杂 GUI 操作仍需要单独实验。

Fallback 仍有 adversarial 风险。Fast path 在当前测试里没有误放行动，弱模型 fallback 在困难负例上仍会误触发。

项目累计成本已经在 `reports/cumulative_experiment_spend.json` 中按本地 state dir 汇总；正式财务口径仍以 API 平台账单为准。

最终验收证据限定为 `reports/notification_triage_real_v1.bundle/` 和 `.cps-real-full-v11/`。旧的 `.cps-real-full-v*` 目录只作为历史成本账本保留，不作为最终验收 artifact。

## 结论

这次实验验证了 CPS 在一个真实 LLM demo 中的协作价值。强模型在 optimize 阶段生成 semantic ProgramPatch plan；弱模型承担局部语义判断；本地程序负责校验、模板化输出和状态管理。安装 ProgramPatch 后，GeneralizedPatch 在 heldout 上达到 `0.8375` 的 quality，strong calls/event 为 `0.0`，fast-path hit rate 为 `53.125%`，false fast-path rate 为 `0.0`。

更一般地说，CPS 让强模型能力可以转成后续运行时可复用、可审计的程序结构。这条路线适合继续在端云协同、GUI Agent 和 skill 优化场景中验证。

## 证据索引

- `reports/notification_triage_real_v1.json`
- `reports/notification_triage_real_v1.md`
- `reports/notification_triage_real_v1.cost.json`
- `reports/notification_triage_real_v1.quality.json`
- `reports/cumulative_experiment_spend.json`
- `reports/label_provenance.md`
- `reports/authoritative_audit_scope.md`
- `reports/proposal_self_check.md`
- `reports/notification_triage_real_v1.bundle/`
- `.cps-real-full-v11/experiments/notification_triage_real_v1/report.json`
- `.cps-real-full-v11/experiments/notification_triage_real_v1/predictions/`
- `.cps-real-full-v11/model_calls/calls.jsonl`
- `.cps-real-full-v11/budgets/spend.jsonl`
- `data/notification/`
- `experiments/notification_triage.real.yaml`

## 撰写说明

本文由 Codex 根据仓库内实验产物和审计结果整理，没有引入外部文献或额外数据。报告中的路径均为仓库相对路径。本文的主张只覆盖当前实验材料能支持的范围。
