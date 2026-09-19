# 国内机构与商业客户：v3 模型区分

本说明覆盖 CHANNEL_MODEL.md、INFORMATION_MODEL.md 中旧的国内转交企业流程。2026-09-16 修订。全部参数均为探索参数，无现实校准。A/B 入门基线保持原样；主研究模型为 src/channel_choice.py。

## 两条路径

国内路径：员工提交 → 国内正式机构处理 → 形成支持补救的结果 → 等待履行或结果落实 → 解决 / 未落实。

客户路径：员工向雇主的商业客户提交 → 客户审核 → 商业介入 → 企业响应 → 解决 / 未解决。

国内路径不是向企业求助。国内机构、企业、商业客户的职能不同。这里的机构是正式争议处理渠道的聚合抽象，不把行政处理、仲裁、司法程序说成同一法律程序，也不假定任何机构均有直接强制执行权。“落实”聚合自主履行及适用制度下的落实过程，尚未细分。

## 参数与公式

保留旧配置键以方便读取，但 `domestic_success` 在 v3 明确定义为机构支持补救概率的基准系数，不再是企业处理概率，也不是最终解决率。旧版数值与新版结果不能直接等同。

对案件 i，特征 m_i 为探索性的案件支持条件，取值 [0,1]：

P(机构支持补救)=domestic_success × (0.5+0.5m_i)。

P(落实成功 | 机构支持补救且进入落实处理)=domestic_implementation_success。

若没有竞争渠道、没有队列截尾、等待充分，两阶段独立抽签时最终成功概率为两者乘积；有限期实际解决率还受到等待、容量、退出及另一渠道先解决的影响。

| 参数 | 含义 | 默认值 |
|---|---|---|
| domestic_delay | 机构处理最短等待轮数 | 2 |
| domestic_capacity | 每轮机构处理总案件上限 | 12 |
| domestic_success | 机构支持补救概率系数 | 0.55 |
| domestic_implementation_enabled | 是否启用国内结果落实 | true |
| domestic_implementation_delay | 支持结果形成后最短落实等待 | 1 |
| domestic_implementation_capacity | 每轮国内落实处理总案件上限 | 12 |
| domestic_implementation_success | 已进入落实处理的条件成功概率 | 0.85 |
| firm_response_enabled | 是否启用客户介入后的企业响应，仅客户路径 | true |
| firm_delay / firm_capacity | 客户介入后企业等待 / 每家企业每轮处理上限 | 1 / 3 |

两类容量独立，是最小模型假设，不代表现实企业执行资源无限。失败是本轮模型路径结束、案件仍未解决，不代表真实法律认定员工无理，也未模拟重新申请、复议等程序。关闭落实开关时，已获支持案件保持等待，而非自动失败。

材料形成是独立事件。现有材料仅表示可向客户展示的案件资料或处理记录，可能在未获支持时形成；不得把所有材料标为“官方违法认定”。客户采信材料产生帮助是可关闭的探索假设，尚无现实验证。

## 事件顺序与去重

每轮：生成争议 → 观察上一轮以前的行动/结果 → 更新获知信息与主观判断 → 同步选择首次或追加渠道 → 国内机构处理 → 客户审核 → 国内落实和客户介入后的企业响应分别评价 → 统一结案与取消其他待处理路径 → 统计。

当同一轮两条路径均成功，resolved_by=d+e，但一个案件仅解决一次、赔偿一次。同轮产生的材料与解决经历下轮才可被观察。

国内状态新增 awaiting_implementation，始终计入待处理投诉，不能视为成功经历。案件新增 d_institution_outcome 和 d_implementation_at；d_responded 表示机构处理时间，resolved 表示实际解决时间。新增每轮机构待处理量、落实待处理量、支持数、落实成功数。

机构和落实随机流分开，新增流未改变既有流编号。不同情景共用网络、人员、初始条件及按轮/员工索引的潜在随机数。

## 与论文研究问题的关系

“国内渠道改善”必须指出具体改善哪部分：机构支持效果、机构速度、落实效果或落实速度。不要一次全改。

现有 information_experiments 的改善作用于机构支持系数。通知信号仍是独立的主观信息输入，不意味着员工知道真实概率，更不等于最终解决率；研究中需另做落实改善及通知内容的敏感性分析。

观察员工是否先选国内、先选客户、同时选择，以及未解决时是否追加另一渠道。应同时报告未投诉、未解决、机构支持但尚未落实的比例，避免以机构处理量冒充解决量。

## 运行与版本

```powershell
.\run.ps1 -m unittest discover -s tests -v
.\run.ps1 -m experiments.information_experiments --config configs/information_small.json --repeats 2 --out outputs/information_small_v2
.\run.ps1 -m experiments.information_experiments --repeats 20 --out outputs/information_experiment_v2
```

原 outputs/information_experiment 和 outputs/channel_choice_v2 为历史版本，不会自动更新。新版请查 information_experiment_v2 和对应 design.json 的完整参数、版本及源码哈希。不能只替换图标题而继续使用旧结果。
