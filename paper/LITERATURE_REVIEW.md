# 文献核查与本项目定位

核查日期：2026-09-17。范围为公开网页、作者/高校存档与出版平台。属于初步定向检索，不是系统综述；未检索完所有数据库，未证明全球首创。全文可读不等于代码开放。本轮没有确认这些文献提供可运行源代码，也没有下载全文到本地。

## 已核查文献与用途

|编号|文献与访问方式|本次访问程度|与模型的关系及限制|
|---|---|---|---|
|R1|Amengual (2010), *Complementary Labor Regulation: The Uncoordinated Combination of State and Private Regulators in the Dominican Republic*. World Development 38(3), 405–414. [免费作者PDF](https://web.mit.edu/amengual/www/Complementary_Labor_Regulation.pdf)，DOI 10.1016/j.worlddev.2009.09.007|PDF可读取，核查摘要、引言及涉及投诉的段落|已讨论公共与私人劳动治理的互补；本项目不能声称首次提出两者关系。它不是本项目员工效用公式的依据。|
|R2|Harrison, Parejo & Wielga (2024), *The value of complaints mechanisms in the private labour regulation of GVCs: A case study of the Fair Labor Association*. International Labour Review 163(1), 73–94. [高校全文入口](https://wrap.warwick.ac.uk/id/eprint/176853/)，DOI 10.1111/ilr.12405|已打开22页开放PDF，核查摘要和第4节|区分进入、办理和结果，对渠道可达性与过程分离有参考价值。FLA是多利益相关方组织，不能直接等同于商业客户。|
|R3|Sonderegger-Wakolbinger & Stummer (2015), *An agent-based simulation of customer multi-channel choice behavior*. Central European Journal of Operations Research 23, 459–477. [出版社摘要](https://link.springer.com/article/10.1007/s10100-015-0388-5)|出版社摘要与书目信息可读；未取得免费全文|多渠道选择ABM、异质性和社会互动已有先例。研究对象是零售消费者，不能直接验证劳动投诉行为；具体模型细节和代码仍待获取。|
|R4|Babineau & Stephens (2024), *Hotlines, Private Regulation, and Farm Migrant Labor Rights: Effective Grievance Mechanisms and the Role of Accessibility*. Geography Research Forum 43, 53–80. [期刊入口](https://grf.bgu.ac.il/index.php/GRF/article/view/631)，[PDF入口](https://grf.bgu.ac.il/index.php/GRF/article/view/631/547)|期刊搜索索引可见摘要与PDF首页；直接打开返回错误，不能声称本次阅读全文|提示私人申诉渠道可达性有研究依据；不是对本模型二元可达性、0.8/0.9参数的估计依据。|
|R5|Grimm et al. (2020), *The ODD Protocol for Describing Agent-Based and Other Simulation Models: A Second Update to Improve Clarity, Replication, and Structural Realism*. JASSS 23(2), 7. [免费网页](https://www.jasss.org/23/2/7.html)，DOI 10.18564/jasss.4259|开放全文，核查协议结构|用于规范模型描述及设计依据，不提供劳动投诉参数。|
|R6|ten Broeke, van Voorn & Ligtenberg (2016), *Which Sensitivity Analysis Method Should I Use for My Agent-Based Model?* JASSS 19(1), 5. [免费网页](https://www.jasss.org/19/1/5.html)，DOI 10.18564/jasss.2857|开放全文，核查摘要及方法讨论|说明局部单因素检查与更完整的敏感性分析有区别；本项目不能据12个块声称已完成全局分析。|
|R7|Collins, Koehler & Lynch (2024), *Methods That Support the Validation of Agent-Based Models: An Overview and Discussion*. JASSS 27(1), 11. [免费网页](https://www.jasss.org/27/1/11.html)，DOI 10.18564/jasss.5258|开放全文，核查定义及验证方法概述|帮助区分实现核验和现实有效性，不能把测试通过等同于实证验证。|
|R8|FLA, *Third Party Complaint Grievance Mechanism* 信息说明。[官方免费PDF](https://www.fairlabor.org/wp-content/uploads/2025/04/ENG-FLA-TPC-Info-Sheet.pdf)|已读取2页PDF；URL上传目录为2025/04，文档日期未核实|表格要求说明此前向工厂、买方或劳动机构反映问题的结果，提示跨渠道经历值得记录。不构成所有程序允许并行投诉的法律依据，也不证明追加概率。|

## 不应声称的创新

- 公共与私人劳动治理可能互补：R1已有研究。
- 使用ABM研究多渠道选择，或加入异质性、同事互动：R3已有相近方法。
- 私人投诉入口的可达性和结果有效性重要：R2/R4已研究。
- 使用配对种子、bootstrap、守恒检查：属于研究质量措施，本身不是新算法。
- 同时使用两条渠道即制度互补、投诉减少即制度胜出：这些推论并未由本项目识别。

## 可以保留但尚未确立的贡献表述

当前项目可以准确声称：建立案件级、可追踪首次选择和双向追加的探索性ABM；通过单渠道/共存反事实对照区分使用覆盖变化与案件解决；通过关闭追加、同事影响及对称诊断说明默认结果的来源。

该组合是否构成足够的新颖性，需要与动态渠道选择、投诉升级、私人治理的具体模型逐项比较。仅本次未检索到完全相同模型，不等于不存在。

论文优先检验“忽略后续追加是否会改变渠道竞争的判断”，而不是把“追加机会更多所以提交更多”包装成新发现。应比较追加带来的实际解决变化、方向不对称及其对预设条件的依赖。

## 假设证据等级

|模型项|当前证据等级|写作要求|
|---|---|---|
|正式与私人劳动治理可同时存在|相关经验研究，R1|说明不同地区/行业不可直接外推|
|私人渠道有进入、处理、补救等环节|相关研究，R2|材料可支持过程区分，不能把FLA替换为所有商业客户|
|可能存在先前跨渠道反映经历|官方说明，R8|仅支持记录历史的必要性，不证明自由并用|
|最大化增量预期收益的个体选择|探索性假设|尚无本场景行为估计|
|观察同事会降低成本或更新成功判断|探索性假设|R3只提供异领域建模先例，不是劳动行为验证|
|每案每渠道一次、等候3轮后追加|实现简化|需说明制度/行为适用边界|
|客户依赖与企业响应呈线性关系|探索性假设|不能直接用境外收入占比校准|
|默认参数及分布|探索性假设|不得标为经验校准|

## 本次检索记录与未完成工作

检索式包括：`Amengual Complementary Labor Regulation`；`worker grievance mechanisms public private regulation complaint channel choice`；`agent based model complaint channel choice grievance public private governance`；`An agent-based simulation of customer multi-channel choice behavior`；`sequential complaint channel choice model`。

另发现 Frasquet、Ieva、Ziliani 的2021年 *Complaint behaviour in multichannel retailing: a cross-stage approach*（DOI 10.1108/IJRDM-03-2020-0089）。出版社/DOI直连本次打开失败，作者高校页面和搜索结果可核查题录；列作下一步精读线索，未把其具体结论作为本文证据。

待完成：获取R3全文并核对决策/时序规则；对上述2021年顺序投诉研究做一手全文核查；在可访问的学术数据库复核检索并做前向/后向引文追踪；选择可核实的商业客户直接受理案例，审查本文聚合流程与之相符的范围。不要把本检索表当作PRISMA式系统综述。
