# 支持性材料与国内落实时序

2026-09-19。接续图片左图研究，检验国内认定/支持和实际落实之间的时间差。
上一轮材料在客户审核前可用的机会少，故本轮预定时序对照，不修改材料增益以制造曲线。

## 固定规则与对照

- 国内初次响应延迟2轮、支持概率参数0.55、落实概率0.85。
- 支持后落实延迟分别0、1、4、8轮。
- 每种延迟比较材料无作用、仅客观材料作用、客观+主观材料作用。
- 材料形成规则、客户流程、先验、成本、网络、企业归属、客户容量不变。
- 材料当轮形成不能当轮用于客户审核；延迟0时国内可在支持当轮落实，但两侧同轮成功仍联合记账一次。
- 所有参数是探索参数；不主张真实机关有相应概率或时长。

总计12情景×连续20种子；先2种子小规模。每个种子保持相同潜在案件、个体差异和事件随机数流。
改变延迟会改变实际事件使用的轮次随机数，不保证每个案件经历相同成败。
落实延迟1的三种材料模式与上一轮支持参数0.55中心完全相同，用作跨实验回归核查。

## 判断标准

在每种落实延迟内计算客观材料开关差，分别看客户使用、客户条件成功、全部争议案件解决。
完整模式减仅客观模式用于诊断额外主观预期作用，不能当成客观互补。
进一步计算最大延迟的材料开关差减最小延迟的材料开关差，检验时序是否改变材料的总作用。
全部比较以配对种子bootstrap逐项95%区间报告；未做多重比较调整。

只看客户成功率上升不足以声称社会福利改善；延迟可能降低国内解决、增加并用和等待成本。
本轮不估计福利净收益。国内变慢时材料作用增强也不意味着故意拖慢国内渠道更好。
每批统一观察24轮，未解决比例由1减全部解决比例获得；有限窗口结果不等于最终终生结果。

## 运行

```powershell
.\run.ps1 -m experiments.substitution_complementarity --config configs/material_timing.json --out outputs/material_timing_small --repeats 2
.\run.ps1 -m experiments.material_timing_readout --out outputs/material_timing_small
.\run.ps1 -m experiments.substitution_complementarity --config configs/material_timing.json --out outputs/material_timing_v1
.\run.ps1 -m experiments.material_timing_readout
```

主要文件：FINDINGS.md为解释，metrics.csv为逐种子原始指标，paired_contrasts.csv为同延迟材料对照，
timing_interactions.csv为时序交互，implementation_comparison.png为水平图，material_effects.png为配对作用图。
所有使用比例以对应争议案件为分母；客户路径成功以客户使用案件为分母。共同中文图例，阴影/误差线为逐项95%区间。

本轮未改行为核心，只扩展实验维度。中图企业差距和右图持续争议反馈仍是下一阶段。
