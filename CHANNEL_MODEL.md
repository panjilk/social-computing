# 国内正式渠道与外部商业客户：v2 模型说明

**历史版本说明。当前 v3 规则以 [DOMESTIC_PROCESS.md](DOMESTIC_PROCESS.md) 为准。** 下文企业共用处理队列的国内机制已被替换，旧输出保留作版本记录。新默认输出为 outputs/channel_choice_v3。

后续已增加独立可选的信息实验，见 [INFORMATION_MODEL.md](INFORMATION_MODEL.md)：在指定轮次改变实际国内效果，比较公告立即到达、沿网络到达和不主动告知。下文描述不启用该模块的 v2 基础规则；材料、企业依赖等基础机制保持原定义。

当前实现 src/channel_choice.py，入口 experiments/channel_experiments.py，配置 configs/channel_choice.json。旧 A/B 阈值模型独立保留。v2 输出 outputs/channel_choice_v2；旧 outputs/channel_choice 是修正前结果，不能混用。所有参数均为探索值，未经现实校准；用户图片仅提供研究问题，曲线、三七分解与现实机构判断均未作为事实输入。

## 1. 主体与案件

默认 120 名员工、6 家企业、45 轮抽象时间。ER 网络平均度期望 6；可选 WS（重连 .15）、BA（m=degree/2）。员工均衡随机分配企业，网络与企业归属独立，尚未强化同事边。

员工拥有材料质量、成本、主观预期及客户可联系性。国内 D 是聚合正式处理流程，不等同于某一机构或法律程序。客户 E 是雇主的商业采购方，不是境外政府、监管机构或证券交易所；“外部”不自动表示境外。客户审核与企业处理分开。

一名员工同时最多一个未解决案件；新争议编号 employee:created_round。案件状态 unsubmitted、pending、closed_unresolved、resolved。渠道状态 unused → submitted → forwarded → successful/closed_unresolved；客户也可直接拒绝。另一渠道解决时取消在办路径，记 cancelled_by_resolution。取消不是失败，流程结束不是案件解决。每案每渠道最多提交/处理一次，没有重试机制。

解决与 1 单位补偿每案最多一次；同时两路径成功标记 d+e，但不重复计算。补偿没有现实货币含义。resolved_by 是模型事件归属，不是现实因果识别。

## 2. 企业差异与独立随机源

企业 has_customer~Bernoulli(.8)，员工在企业有客户时以 .9 概率可联系客户。可联系性固定，并在决策中视为员工已知；未模拟联系方式扩散。

企业依赖：抽取 u~Uniform(-1,1)，减去样本均值，再除以最大绝对偏差得到 z；零偏差（含只有一个企业）时 z=0。d=dependence_mean+dependence_spread*z，默认 .5 和 .1。半宽超过均值到 [0,1] 边界的距离时报错，不截断。相同种子固定半宽改变均值时，实际样本均值改变而方差不变。它是构造的有限样本分布，不是独立均匀分布，更不是境外收入占比。

员工材料质量 m~Uniform(.2,1)，是可核实性的探索代理，不代表案件真伪；新案件沿用员工 m。成本为配置成本 D=.18/E=.28 乘 Uniform(.5,1.5)。

SeedSequence([seed,100,process_id]) 分开网络、初始条件、员工差异、企业差异、信息共享及各事件。事件数组按轮次×员工预生成，增加观察期限保持原有随机前缀。不同处理时间读取不同轮次随机数，这是时间配对的边界。版本变化涉及多项机制，不能将旧版与新版总差异归因于某一修正。

## 3. 员工观察什么

社交边上以 observation_probability（默认 1）抽取固定有向共享关系，独立于其他随机源。默认邻居共享信息；.5 是不完全可见性对照，0 完全不可见。不是全社会可见。

分别记录每条渠道的近期提交 recent、在办 pending、已知成功 success、已知流程结束未解决 failure。默认 observation_mode=pending_or_recent：行动信息是近期提交与在办的去重员工并集；长期未结束的投诉不会因窗口到期消失。recent 模式只看近期提交，用于机制对照。当轮提交下一轮才可见。

行动比例 q 的分母是所有邻居人数，包含未共享者；孤立节点为 0。感知成本 c(t)=c_initial*(1-social_weight*q)，默认 social_weight=.2。行动不直接改变主观成功概率。

success/failure 从各路径结束的下一轮起保留 visibility_window=3 轮，按可见邻居去重人数 S、F 计数。拒绝和企业处理失败计 F，在办不计失败，因其他路径解决而取消不计成功/失败。同一员工在不同案件出现不同结果时可分别进入 S 与 F，但每种结果内去重。这是可共享路径结果的探索假设，不代表现实员工能够准确识别因果。

结果更新：p_post=(a*p_initial+S)/(a+S+F)，p_social=(1-w)*p_initial+w*p_post。a=outcome_prior_strength 默认 2，w=outcome_learning_weight 默认 .3。每轮从固定初始值与窗口重算，不把同一观察结果重复累积成新证据。无已知结果时维持初始值；w=0 关闭结果更新。不是跨案件累计贝叶斯学习。

## 4. 实际效果、主观信号与等待

初始 p_D=clip(domestic_prior+domestic_information*(domestic_signal-.55)+epsilon_D,0,1)。p_E=clip(customer_prior+perceived_dependence*(d-.5)+epsilon_E,0,1)。默认 prior .55/.45，information=.7，signal=.55，perceived_dependence=.25，epsilon~Uniform(-.2,.2)。

domestic_signal 独立于客观 domestic_success。主扫描只变客观效果，初始预期不变；单独设信号变化与效果/信号共同变化。固定初始信念不代表以后社会观察不改变信念。

社会更新后，如果国内材料在上一轮或更早形成，p_E 加 perceived_evidence_bonus=.18 并截断。客观 evidence_help 与主观加成分别控制；关闭实际帮助不自动改变主观材料价值。

每 review_interval=1 轮复评。首次提交后至少 escalation_wait=3 轮才追加未使用渠道。allow_escalation=False 禁止追加，但未投诉者仍可在后续首次行动，原有处理继续。

预计等待 T_D/T_E 默认 3，独立于实际 domestic_delay/customer_delay 和容量。不使用实际队列的全知等待时间。单独比较实际变慢（未知）、预计变慢、两者共同变慢（已知）。

未用路径折现概率 z=p*exp(-delta*T)。在办路径已经等待 age 时：z=p*exp(-eta*max(0,age-T))*exp(-delta*max(1,T-age))。默认 delta=.035、eta=.12。超期降低预期但不是已知失败；已结束失败路径不提供继续等待收益。

## 5. 选择效用

新增集合 A 为无新增、D、E 或 D+E，受可联系性、开关及既有使用限制。R 为既有在办路径。U(A)=B*J({z_k:k 属于 R 并 A})-新增成本-新增双渠道协调成本。

B=1，协调成本 .02，在新增行动首次构成双渠道案件时收取。沉没成本不重复扣除。无新增也可能包含继续等待收益。J(empty)=0，J(z)=z；两条路径令 low=min(z)、high=max(z)，J=high+(1-rho)*low*(1-high)。rho=subjective_route_overlap 默认 0（独立机会），rho=1 是完全嵌套的主观成功机会；.8 为敏感性对照。这只调整主观重叠，不改变客观事件相关结构，不涵盖负相关。

按最大效用选，平局依无新增、D、E、双渠道顺序。没有强制跟风阈值。decisions.csv 保存所有候选效用、各类观察量、概率、成本、选择。未新增不是永久放弃。

## 6. 客观处理

D 提交至少等待 domestic_delay=2 轮后进入全体共享队列，容量 domestic_capacity=12，按提交时间/员工号处理。回应时以 material_rate*(.5+.5*m) 形成材料，默认 material_rate=.7，然后送达企业。回应和材料均不是解决。

E 等待 customer_delay=2 轮后进入客户审核队列，容量 customer_capacity=12。p_review=clip(.2+.45*m+evidence_help*I_material,0,1)，默认 evidence_help=.3；只用本轮开始前存在的材料。拒绝结束此路径，不解决案件；接受后送达企业。

企业再等待 firm_delay=1，每企业每轮容量 firm_capacity=3 个独立案件。按最早到期时间/员工号处理，同案两条到期请求只占一个名额。p_firm,D=domestic_success*(.5+.5*m)，默认 .55；p_firm,E=clip(firm_customer_base+dependence_effect*d,0,1)，默认 .2+.65*d。分别抽样，每路径至多处理一次；任一成功结案，取消另一未结束路径。关闭企业响应时不会自动解决。

材料帮助与客户依赖的正向系数是预设探索机制；观察到正效应本身不是创新或经验发现。可设为零检查。

## 7. 每轮顺序、新争议与停止

1. t=0 抽取初始争议（概率 .65）；以后对没有未解决案件者按 new_dispute_rate 抽取新争议，默认 0。
2. 冻结之前轮次的共享行动、结果与材料信息，员工同步作选择。
3. 登记新增提交。追加不新建案件。
4. 国内队列回应、可能形成材料、送达企业。
5. 客户队列审核，只用第 2 步冻结的材料。
6. 企业处理，合并同案成功，单次补偿，取消其他在办路径。
7. 更新案件状态，记录流量与存量。

固定运行 steps 轮，不因暂时无投诉停止。可选 prevention 默认 0：企业曾解决案件后从下一轮起，未来争议率降为 lambda*(1-prevention)。这是简单可关闭机制，不是企业适应性决策，也不是已有未投诉案件普遍整改。

## 8. 指标与实验

new_complaints 是首次提交的独立案件数；追加不重复计算。各渠道 new_domestic/new_customer 包含首次和追加。同案可同时出现两条渠道记录。员工累计数跨案件去重。total_cases=unresolved+cumulative_resolved；unresolved=unsubmitted+pending+closed_unresolved。

customer_share_all_cases：用过客户路径案件/所有发生案件；customer_share_submitted：用过客户路径案件/至少提交过一个渠道案件。国内对应指标同理，可重叠。first_*_share 分母为首次投诉案件；no_submission_share 分母为所有案件。domestic_then_customer_share 分母为首先仅 D 的案件，含观察期内尚未追加者。

external_path_trigger_rate：客户路径触发企业成功案件/用过客户路径案件（含双路径同轮成功）。external_user_case_resolution_rate：客户使用者被任一路径解决案件/客户使用案件。customer_intervention_rate 分母为已完成客户审核案件，不把待审核计为拒绝。未解决比例和处理时间同时报告。已解决 processing_time=解决轮-首次提交轮，未解决留空；observed_wait 使用截至观察末轮的等待，不填 0。unresolved_case_rounds 是每轮末未解决数之和。

visible_domestic/customer 是本轮开始具有行动信息的全体去重员工数，尚未应用观察者共享过滤；不能解释为每个员工实际看到的数量。个人可见信息见 decisions.csv。pending/recent 是邻居比例，success/failure 是可见邻居人数。

分母为 0 返回 NaN。先逐运行计算比例，再按种子等权汇总；count 报告有效次数。图为均值 ±1 样本标准差，不是置信区间；paired_differences.csv 同时保存配对差均值、SD、SE 与 n，SE=SD/sqrt(n)。

主扫描 D 实际效果 .15/.35/.55/.75/.95 × 材料帮助开关。另比较无响应、无延迟、两种实际延迟、已知延迟、主观信号、只允许首次选择、近期可见性、结果更新、部分共享、主观路径重叠、依赖均值/离散、两倍观察期限、新争议与预防、ER/WS/BA。每情景 20 个连续相同种子；最终覆盖参数、源码哈希、运行次数见 batch/design.json。

固定案件检查移除选择、国内提前解决、队列和时间，只比较同一批案件客观材料帮助开/关；这是预设机制验证，不是筛选/互补百分比的因果分解。不能以两个差值相除声称三七分解。使用者材料质量与企业依赖均值同时输出用于检查构成。

## 9. 边界及后续研究

尚未模拟现实具体法律程序、累计经验学习、多个客户关系网络、企业自适应策略、独立合规状态、政策冲击和已有未投诉案件的普遍整改。信息可见性、更新强度、成本、等待预期、材料价值、客户介入与企业处理概率均待文献或现实证据支持。

v2 是正确性修复与探索假设扩展，不代表已确认论文创新。可见性、学习、企业分布等改变同时发生；解释作用应使用 v2 内的配对消融，不能把不同版本曲线差当成单一机制效果。
