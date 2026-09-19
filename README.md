# 国内正式渠道与外部商业客户渠道：多主体仿真原型

本项目当前只保留围绕“国内正式渠道—外部商业客户渠道制度竞争”的可运行代码。所有默认参数都是演示或探索参数，尚未用真实行为数据校准，仿真结果不能直接当作现实预测。

## 当前研究主线

- [IMAGE_RESEARCH_MAINLINE.md](IMAGE_RESEARCH_MAINLINE.md)：五张图片对应的总研究主线。
- [COMPETITION_MODEL.md](COMPETITION_MODEL.md)：固定条件下首次选择与后续追加的双渠道基线。
- [SUBSTITUTION_COMPLEMENTARITY.md](SUBSTITUTION_COMPLEMENTARITY.md)：国内支持与客户渠道使用的替代/互补对照。
- [DEPENDENCE_GAP.md](DEPENDENCE_GAP.md)：企业客户依赖与企业间结果差距。
- [CONTINUOUS_COMPETITION.md](CONTINUOUS_COMPETITION.md)：持续新争议、信息传播与企业预防反馈。

## 安装与测试

推荐 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

在本项目环境中也可以使用项目脚本：

```powershell
.\run.ps1 -m unittest discover -s tests -q
```

当前保留测试共 74 项，覆盖渠道选择、案件生命周期、客户网络、持续争议、依赖扫描和配对实验。

## 运行当前实验

先运行小规模演示：

```powershell
.\run.ps1 -m experiments.competition --small --repeats 2 --out outputs/competition_small
.\run.ps1 -m experiments.substitution_complementarity --small --repeats 2 --out outputs/substitution_small
.\run.ps1 -m experiments.dependence_gap --small --repeats 2 --out outputs/dependence_small
.\run.ps1 -m experiments.continuous_competition --small --repeats 2 --out outputs/continuous_small
```

正式实验入口：

```powershell
.\run.ps1 -m experiments.competition --repeats 30 --out outputs/competition_v1
.\run.ps1 -m experiments.substitution_complementarity --repeats 20 --out outputs/substitution_complementarity_v1
.\run.ps1 -m experiments.dependence_gap --repeats 50 --out outputs/dependence_gap_v1
.\run.ps1 -m experiments.continuous_competition --repeats 20 --out outputs/continuous_competition_v1
```

材料落实延迟的独立对照使用：

```powershell
.\run.ps1 -m experiments.material_timing_readout
```

配置文件位于 `configs/`，分别对应上述实验。所有批量实验使用配对随机种子；输出目录中的 CSV、PNG、JSON 和 Markdown 报告保存实际配置、指标和结果。

## 代码结构

### 核心模型

- `src/channel_choice.py`：案件级双渠道选择、等待、追加和处理流程。
- `src/cohort_trends.py`：按争议批次统计首次选择和后续渠道使用趋势。
- `src/customer_network.py`：企业—商业客户网络与客户可达性。
- `src/wait_learning.py`：等待经历和结果观察对主观判断的探索性更新。
- `src/continuous_competition.py`：持续新争议、信息传播、客户介入和企业预防反馈。

### 实验脚本

- `experiments/competition.py`：三组基线（仅国内、仅客户、双渠道）。
- `experiments/competition_refinement.py`：基线的对称性、等待学习和观察期限诊断。
- `experiments/substitution_complementarity.py`：国内支持与客户使用的替代/互补。
- `experiments/dependence_gap.py`：客户依赖扫描及企业分组差距。
- `experiments/continuous_competition.py`：持续争议和传播×预防机制对照。
- `experiments/cohort_trends.py`：动态批次趋势。
- `experiments/customer_network.py`、`experiments/channel_experiments.py`：网络和渠道组件的独立实验。

### 输出与文档

- `outputs/`：已有演示和正式实验结果；旧结果目录仅作历史记录，不由当前代码重建。
- `MODEL.md`：基础规则、假设、公式和限制。
- `PAPER_RESEARCH_PLAN.md`：论文问题、指标和写作计划。
- `paper/`：中文工作稿和文献核查，不能视为投稿定稿。

## 指标解释

客户使用人数、客户使用比例、首次选择、国内→客户追加、双渠道并用、解决案件数和未解决案件数分别统计。两个渠道可以在同一案件上并用，所以两个使用比例相加可能超过 100%；分母会在报告中明确。成功案件处理时间必须与未解决比例一起看，不能只报告已解决案件的平均时间。

## 研究边界

模型用于比较明确规则下的机制和反事实差异，不能据此断言现实中未来一定会更多使用客户渠道，也不能把客户渠道等同于外国监管机构、政府机关或交易所。新闻事件只可作为背景，不能直接作为参数证据。论文投稿前仍需完成文献检索、现实数据或专家证据、参数校准和更系统的敏感性分析。
