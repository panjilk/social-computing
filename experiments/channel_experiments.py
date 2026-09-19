"""Run and plot the channel-choice research model, without altering A/B outputs."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import networkx as nx
from src.channel_choice import ChannelConfig,simulate_channels,world,review_probability,firm_customer_probability,ACTIVE


def audit(result):
    r=result['trajectory']; c=result['cases']; e=result['events']
    assert (r.total_cases==r.unresolved+r.cumulative_resolved).all()
    assert (r.unresolved==r.unsubmitted+r.pending+r.closed_unresolved).all()
    assert (r.new_disputes.cumsum()==r.total_cases).all()
    assert (r.new_resolved.cumsum()==r.cumulative_resolved).all()
    assert c.case_id.is_unique
    assert e[e.event=='case_resolved'].case_id.is_unique
    assert r.new_complaints.sum()==c.first_submitted.notna().sum()
    assert c.compensation.sum()==c.resolved.notna().sum()
    assert (c.compensation<=1).all()
    for key,event in [('d','domestic_received'),('e','customer_received')]:
        assert e[e.event==event].case_id.is_unique
        assert c[key+'_submitted'].notna().sum()==(e.event==event).sum()
    # A case resolved by one route cancels any unfinished other route.
    assert not c[c.resolved.notna()][['d_status','e_status']].isin(ACTIVE).any().any()


def endpoint(result):
    r=result['trajectory']; c=result['cases']
    def fraction(num,den): return float(num/den) if den else np.nan
    total=len(c); filed=c.first_submitted.notna(); ext=c.e_submitted.notna(); dom=c.d_submitted.notna()
    resolved=c.resolved.notna(); first_d=c.initial_choice=='d'
    ext_trigger=c.e_status=='successful'
    customer_reviewed=c.e_responded.notna()
    return dict(**r.iloc[-1].to_dict(),
        unique_submitted_cases=int(filed.sum()),unique_domestic_cases=int(dom.sum()),
        unique_customer_cases=int(ext.sum()),multi_cases=int((dom&ext).sum()),
        customer_share_all_cases=fraction(ext.sum(),total),
        customer_share_submitted=fraction(ext.sum(),filed.sum()),
        domestic_share_submitted=fraction(dom.sum(),filed.sum()),
        first_domestic_share=fraction(first_d.sum(),filed.sum()),
        first_customer_share=fraction((c.initial_choice=='e').sum(),filed.sum()),
        first_both_share=fraction((c.initial_choice=='d+e').sum(),filed.sum()),
        no_submission_share=fraction((~filed).sum(),total),
        domestic_then_customer_share=fraction((first_d&ext).sum(),first_d.sum()),
        mean_escalation_time=(c.loc[first_d&ext,'e_submitted']-c.loc[first_d&ext,'d_submitted']).mean(),
        external_path_trigger_rate=fraction(ext_trigger.sum(),ext.sum()),
        external_user_case_resolution_rate=fraction((ext&resolved).sum(),ext.sum()),
        customer_intervention_rate=fraction(c.customer_intervened_at.notna().sum(),customer_reviewed.sum()),
        material_at_customer_review_share=fraction((customer_reviewed&(c.material_at<c.e_responded)).sum(),customer_reviewed.sum()),
        external_user_mean_merit=c.loc[ext,'merit'].mean(),
        external_user_mean_dependence=c.loc[ext,'dependence'].mean(),
        unresolved_fraction=fraction((~resolved).sum(),total),
        submitted_unresolved_fraction=fraction((filed&~resolved).sum(),filed.sum()),
        external_user_unresolved_fraction=fraction((ext&~resolved).sum(),ext.sum()),
        mean_processing_time=c.loc[resolved,'processing_time'].mean(),
        median_processing_time=c.loc[resolved,'processing_time'].median(),
        unresolved_case_rounds=int(r.unresolved.sum()),pending_case_rounds=int(r.pending.sum()))


def metadata(c,seed):
    import matplotlib
    files=['src/channel_choice.py','experiments/channel_experiments.py']
    return dict(model='channel_choice_v3_institution_implementation',config=asdict(c),seed=seed,
        parameter_status='exploratory, not calibrated',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                      networkx=nx.__version__,matplotlib=matplotlib.__version__),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})


def save_demo(c,seed,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    result=simulate_channels(c,seed); audit(result)
    for name,data in result.items():
        if name=='graph': nx.write_edgelist(data,out/'network.edgelist',data=False)
        else: data.to_csv(out/f'{name}.csv',index=False)
    pd.DataFrame([endpoint(result)]).to_csv(out/'endpoint.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(metadata(c,seed),indent=2),encoding='utf-8')
    return result


def design(base):
    # Set named treatments explicitly; do not silently inherit contradicting switches.
    core=replace(base,domestic_enabled=True,customer_enabled=True,firm_response_enabled=True,
                 domestic_implementation_enabled=True,new_dispute_rate=0.,prevention=0.,domestic_signal=.55,
                 dependence_spread=.1,allow_escalation=True)
    scenarios={}
    for q in (.15,.35,.55,.75,.95):
        for help_on in (False,True):
            name=f'domestic_{q:.2f}_'+('help_on' if help_on else 'help_off')
            scenarios[name]=replace(core,domestic_success=q,evidence_help=.3 if help_on else 0.)
    central=replace(core,domestic_success=.55,evidence_help=.3)
    changes={
        'no_domestic':dict(domestic_enabled=False),
        'no_customer':dict(customer_enabled=False),
        'no_firm_response':dict(firm_response_enabled=False,domestic_implementation_enabled=False),
        'no_material':dict(material_rate=0.),
        'dependence_low':dict(dependence_mean=.15),
        'dependence_high':dict(dependence_mean=.85),
        'social_off':dict(social_weight=0.,outcome_learning_weight=0.),
        'social_high':dict(social_weight=.6),
        'domestic_slow':dict(domestic_delay=6),
        'customer_slow':dict(customer_delay=6),
        'no_delay':dict(domestic_delay=0,domestic_implementation_delay=0,customer_delay=0,firm_delay=0),
        'domestic_slow_known':dict(domestic_delay=6,expected_domestic_wait=7),
        'customer_slow_known':dict(customer_delay=6,expected_customer_wait=7),
        'domestic_expected_slow_only':dict(expected_domestic_wait=7),
        'signal_low_only':dict(domestic_signal=.15),
        'signal_high_only':dict(domestic_signal=.95),
        'objective_and_signal_low':dict(domestic_success=.15,domestic_signal=.15),
        'objective_and_signal_high':dict(domestic_success=.95,domestic_signal=.95),
        'first_choice_only':dict(allow_escalation=False),
        'recent_only':dict(observation_mode='recent'),
        'outcome_learning_off':dict(outcome_learning_weight=0.),
        'partial_visibility':dict(observation_probability=.5),
        'subjective_overlap_high':dict(subjective_route_overlap=.8),
        'dependence_spread_zero':dict(dependence_spread=0.),
        'long_horizon':dict(steps=2*central.steps),
        'new_on':dict(new_dispute_rate=.025),
        'new_on_prevention':dict(new_dispute_rate=.025,prevention=.5),
        'network_ws':dict(network='ws'),
        'network_ba':dict(network='ba'),
        'low_objective_fixed_belief':dict(domestic_success=.15,domestic_information=0.),
        'high_objective_fixed_belief':dict(domestic_success=.95,domestic_information=0.)}
    scenarios.update({name:replace(central,**change) for name,change in changes.items()})
    return scenarios


def fixed_cohort(c,seed):
    """Standardized external process probe, not a decomposition of aggregate changes.

    Force the same initially disputed, reachable people into an external review.
    Give the same subset domestic materials in both arms. No domestic resolutions,
    selection, timing or queues; only objective evidence-help coefficient differs.
    """
    w=world(c,seed); use=w['initial']&w['reachable']
    material=w['draws']['material'][0]<c.material_rate*(.5+.5*w['merit'])
    company=w['draws']['customer_outcome'][0]<firm_customer_probability(c,w['dependence'][w['firm']])
    off=w['draws']['review'][0]<review_probability(replace(c,evidence_help=0.),w['merit'],material)
    on=w['draws']['review'][0]<review_probability(replace(c,evidence_help=.3),w['merit'],material)
    rows=pd.DataFrame(dict(seed=seed,employee=np.arange(c.n),merit=w['merit'],
        dependence=w['dependence'][w['firm']],material=material,
        success_off=off&company,success_on=on&company))[use]
    return rows


def run_batch(base,repeats,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    scenarios=design(base); endpoints=[]; trajectories=[]; case_rows=[]
    for name,c in scenarios.items():
        for seed in range(repeats):
            result=simulate_channels(c,seed); audit(result)
            endpoints.append(dict(scenario=name,seed=seed,**endpoint(result)))
            for key,container in [('trajectory',trajectories),('cases',case_rows)]:
                container.append(result[key].assign(scenario=name,seed=seed))
        print(f'Completed {name}: {repeats} paired seeds',flush=True)
    e=pd.DataFrame(endpoints); e.to_csv(out/'endpoints.csv',index=False)
    pd.concat(trajectories,ignore_index=True).to_csv(out/'trajectories.csv',index=False)
    pd.concat(case_rows,ignore_index=True).to_csv(out/'cases.csv',index=False)
    metrics=[col for col in e.columns if col not in ('scenario','seed')]
    e.groupby('scenario')[metrics].agg(['mean','std','count']).to_csv(out/'summary.csv')
    pairs=[(f'domestic_{q:.2f}_help_on',f'domestic_{q:.2f}_help_off') for q in (.15,.35,.55,.75,.95)]
    pairs += [('new_on_prevention','new_on'),('high_objective_fixed_belief','low_objective_fixed_belief')]
    pairs += [(name,'domestic_0.55_help_on') for name in (
        'first_choice_only','recent_only','outcome_learning_off','partial_visibility',
        'domestic_slow','domestic_slow_known','domestic_expected_slow_only',
        'signal_low_only','signal_high_only','objective_and_signal_low','objective_and_signal_high',
        'subjective_overlap_high','long_horizon','no_delay')]
    pairs += [('domestic_slow_known','domestic_slow')]
    paired=[]
    for treatment,control in pairs:
        delta=e[e.scenario==treatment].set_index('seed')[metrics]-e[e.scenario==control].set_index('seed')[metrics]
        for col in metrics:
            paired.append(dict(treatment=treatment,control=control,metric=col,
                mean_difference=delta[col].mean(),sd_difference=delta[col].std(),n=delta[col].count(),
                se_difference=delta[col].std()/np.sqrt(delta[col].count()) if delta[col].count()>1 else np.nan))
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    probe=pd.concat([fixed_cohort(base,seed) for seed in range(repeats)],ignore_index=True)
    probe.to_csv(out/'fixed_cohort_cases.csv',index=False)
    fixed=probe.groupby('seed')[['success_off','success_on']].mean()
    fixed['paired_difference']=fixed.success_on-fixed.success_off
    fixed.to_csv(out/'fixed_cohort_summary.csv')
    info=metadata(base,None)
    info.update(seeds=list(range(repeats)),scenarios={k:asdict(v) for k,v in scenarios.items()},
        runs=len(e),all_run_accounting_passed=True,
        uncertainty='mean +/- 1 sample SD across seeds, not CI',
        fixed_cohort_warning='standardized process probe; not percentage selection/complementarity decomposition')
    (out/'design.json').write_text(json.dumps(info,indent=2),encoding='utf-8')


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); e=pd.read_csv(out/'endpoints.csv'); tr=pd.read_csv(out/'trajectories.csv')
    plt.rcParams.update({'font.size':9,'figure.dpi':140})
    central='domestic_0.55_help_on'
    def line(ax,names,metric,labels=None):
        for name in names:
            s=tr[tr.scenario==name].groupby('time')[metric].agg(['mean','std'])
            ax.plot(s.index,s['mean'],label=(labels or {}).get(name,name))
            ax.fill_between(s.index,s['mean']-s['std'],s['mean']+s['std'],alpha=.12)
        ax.set(xlabel='Round',ylabel=metric); ax.grid(alpha=.2)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,['new_customer','unresolved','pending','cumulative_resolved']):
        line(ax,[central,'domestic_0.55_help_off','no_domestic','no_customer'],metric)
    axes[0,0].legend(fontsize=7); fig.suptitle('Channel-choice dynamics | mean +/- 1 SD (not CI)')
    fig.tight_layout(); fig.savefig(out/'channel_dynamics.png'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,metric in zip(axes.flat,['customer_share_all_cases','external_path_trigger_rate',
                                 'external_user_mean_merit','material_at_customer_review_share']):
        for help_on in (False,True):
            suffix='on' if help_on else 'off'; q=np.array([.15,.35,.55,.75,.95])
            names=[f'domestic_{v:.2f}_help_{suffix}' for v in q]
            s=e.groupby('scenario')[metric].agg(['mean','std']).reindex(names)
            ax.errorbar(q,s['mean'],yerr=s['std'],marker='o',capsize=3,label=f'evidence help {suffix}')
        ax.set(xlabel='Domestic objective success parameter',ylabel=metric)
        ax.grid(alpha=.2)
    axes[0,0].legend(); fig.suptitle('Usage, outcomes and composition | mean +/- SD; initial beliefs fixed')
    fig.tight_layout(); fig.savefig(out/'substitution_complementarity.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=[central,'domestic_0.55_help_off','domestic_slow','customer_slow']
    columns=['first_domestic_share','first_customer_share','first_both_share']
    s=e.groupby('scenario')[columns].mean().reindex(names)
    sd=e.groupby('scenario')[columns].std().reindex(names)
    s.plot.bar(ax=axes[0],yerr=sd,rot=15,capsize=2)
    axes[0].set_ylabel('Share of unique first submissions')
    m=e.groupby('scenario').domestic_then_customer_share.agg(['mean','std']).reindex(names)
    axes[1].bar(range(len(names)),m['mean'],yerr=m['std'],capsize=3)
    axes[1].set_xticks(range(len(names)),names,rotation=15,fontsize=7)
    axes[1].set_ylabel('Later customer use / domestic-only first cases')
    fig.suptitle('First choice and later addition | mean +/- SD; incomplete follow-up retained')
    fig.tight_layout(); fig.savefig(out/'choice_paths.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    fixed=pd.read_csv(out/'fixed_cohort_summary.csv')
    cols=['success_off','success_on']; m=fixed[cols].mean(); sd=fixed[cols].std()
    axes[0].bar(cols,m,yerr=sd,capsize=3); axes[0].set_ylabel('External success in same fixed cohort')
    names=['dependence_low',central,'dependence_high']
    s=e.groupby('scenario').customer_share_all_cases.agg(['mean','std']).reindex(names)
    axes[1].errorbar([.15,.5,.85],s['mean'],yerr=s['std'],marker='o',capsize=3)
    axes[1].set(xlabel='Configured dependence center',ylabel='Customer users / all cases')
    fig.suptitle('Controlled mechanism probe and dependence sensitivity | mean +/- SD')
    fig.tight_layout(); fig.savefig(out/'mechanism_checks.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    line(axes[0],[central,'social_off','social_high'],'new_customer')
    line(axes[1],[central,'new_on','new_on_prevention'],'unresolved')
    for ax in axes: ax.legend(fontsize=7)
    fig.suptitle('Separate social-information and prevention checks | mean +/- SD; no imposed shock')
    fig.tight_layout(); fig.savefig(out/'feedback_checks.png'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    groups=[('domestic_then_customer_share',[central,'domestic_slow','domestic_slow_known','first_choice_only']),
            ('customer_share_all_cases',[central,'signal_high_only','high_objective_fixed_belief','objective_and_signal_high']),
            ('customer_share_all_cases',[central,'recent_only','outcome_learning_off','partial_visibility']),
            ('unresolved_fraction',[central,'subjective_overlap_high','long_horizon','no_delay'])]
    for ax,(metric,names) in zip(axes.flat,groups):
        values=e.groupby('scenario')[metric].agg(['mean','std']).reindex(names)
        ax.bar(range(len(names)),values['mean'],yerr=values['std'],capsize=3)
        ax.set_xticks(range(len(names)),[x.replace('domestic_0.55_help_on','baseline') for x in names],rotation=20,ha='right',fontsize=7)
        ax.set_ylabel(metric); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Separated mechanisms | mean +/- 1 sample SD, not CI')
    fig.tight_layout(); fig.savefig(out/'revision_checks.png'); plt.close(fig)


def report(out):
    out=Path(out); e=pd.read_csv(out/'endpoints.csv'); avg=e.groupby('scenario').mean(numeric_only=True)
    info=json.loads((out/'design.json').read_text(encoding='utf-8'))
    lines=['# 国内渠道与外部商业客户选择：实际结果','',
        f'已运行 {info["runs"]} 次仿真、{len(info["scenarios"])} 个情景；每个情景使用相同连续种子。每次运行均检查案件守恒、渠道去重及单次补偿。参数全部为探索值，没有现实校准。','',
        '## 国内处理效果与材料帮助','',
        '百分比先在每个运行内计算，再跨种子平均。客户使用比例分母为所有发生的独立争议案件；客户路径成功率分母为提交过客户渠道的独立案件，只计客户路径触发的解决，不把国内单独解决冒充客户成功。全部成功均指有限观察期内。','',
        '|国内效果参数|材料帮助|客户使用比例|客户路径成功率|客户使用者最终结案率（任一路径）|',
        '|---|---|---:|---:|---:|']
    for q in (.15,.35,.55,.75,.95):
        for toggle in ('off','on'):
            s=avg.loc[f'domestic_{q:.2f}_help_{toggle}']
            lines.append(f'|{q:.2f}|{toggle}|{s.customer_share_all_cases:.2%}|{s.external_path_trigger_rate:.2%}|{s.external_user_case_resolution_rate:.2%}|')
    lines += ['', '## 首选与追加','', '|情景|国内首选|客户首选|同时首选|国内首选后追加客户|',
              '|---|---:|---:|---:|---:|']
    for name in ['domestic_0.55_help_on','domestic_0.55_help_off','domestic_slow','customer_slow']:
        s=avg.loc[name]
        lines.append(f'|{name}|{s.first_domestic_share:.2%}|{s.first_customer_share:.2%}|{s.first_both_share:.2%}|{s.domestic_then_customer_share:.2%}|')
    fixed=pd.read_csv(out/'fixed_cohort_summary.csv')
    lines += ['', '前三列分母为发生首次提交的案件；追加比例分母为首次仅选国内的案件，包含观察期内尚未追加者。没有把“未继续行动”自动标为放弃。','',
        '## 固定案件检查','',
        f'在同一批初始有争议、客户可联系的员工上固定材料与随机抽样，客户路径成功率：帮助关闭 {fixed.success_off.mean():.2%}，帮助开启 {fixed.success_on.mean():.2%}；配对差均值 {fixed.paired_difference.mean():.2%}，差值样本 SD {fixed.paired_difference.std():.2%}。','',
        '这是移除渠道选择、国内提前解决、队列和时间差异后的标准化流程检查，只验证预设材料帮助机制的作用。不能用它除以主实验变化，宣称“多少百分比来自筛选、多少来自互补”。材料帮助为正是写入的假设，正向效果本身不是经验发现。','',
        '## 图片与本项目的关系','',
        '参考图片用于提出问题，没有把曲线形状、三七分解、特定企业情况、政策冲击或政治判断作为输入事实。此版本没有企业适应性博弈、没有第 150 轮制度冲击，也没有校准企业合规指标，因此不声称复现图片的倒 U 形或“没有级联”。','',
        'substitution_complementarity.png 分别显示使用、路径成功、使用者特征和材料可用情况；choice_paths.png 显示首选和追加；channel_dynamics.png 显示时序；mechanism_checks.png 显示固定案件检查和客户依赖敏感性；feedback_checks.png 分别检查同事信息和预防机制。所有图误差范围为运行间 ±1 样本标准差，不是置信区间。','',
        '## 限制','',
        'v2 主扫描固定初始信念：domestic_signal 独立于 domestic_success。额外情景分别改变客观效果、主观信号、实际延迟和预计等待。社会结果更新是有限窗口的探索性加权规则，不是经过校准的学习机制。国内渠道是聚合流程，未模拟具体法律制度。客户依赖不是境外收入占比，客户不是监管机构。','',
        '主实验同时包含选择、时序、队列、材料和竞争解决路径；关闭材料帮助后仍可能有路径竞争及选择变化，不能把所有剩余变化统称筛选效应。应结合 cases.csv 的案件特征、标准化检查和额外对照解释。','']
    lines += ['## v2 修正对照','',
        '以下相对基准的差值以百分点报告；SE 是配对均值差的标准误，不是置信区间。不同观察期限的比例不代表同一时间点的效果。','',
        '|对照情景|客户使用比例|相对基准差（百分点）|配对差 SE（百分点）|',
        '|---|---:|---:|---:|']
    central='domestic_0.55_help_on'
    for name in ['recent_only','outcome_learning_off','partial_visibility','first_choice_only',
                 'domestic_slow','domestic_slow_known','signal_high_only','high_objective_fixed_belief',
                 'objective_and_signal_high','subjective_overlap_high','long_horizon']:
        baseline=e[e.scenario==central].set_index('seed').customer_share_all_cases
        values=e[e.scenario==name].set_index('seed').customer_share_all_cases
        diff=values-baseline
        lines.append(f'|{name}|{values.mean():.2%}|{100*diff.mean():+.2f}|{100*diff.std()/np.sqrt(diff.count()):.2f}|')
    lines += ['', '修正后在办投诉持续可见；成功和失败按结束时间保留窗口，取消不计失败。'
              '新图 revision_checks.png 展示分离机制对照；此前 v1 结果保留于旧目录，不与 v2 混用。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['demo','batch','plot','report','all'],default='demo')
    p.add_argument('--config',default='configs/channel_choice.json'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--repeats',type=int,default=20); p.add_argument('--out',default='outputs/channel_choice_v3')
    a=p.parse_args(); c=ChannelConfig(**json.loads(Path(a.config).read_text(encoding='utf-8')))
    if a.repeats<2: p.error('repeats must be >=2')
    if a.mode in ('demo','all'): save_demo(c,a.seed,Path(a.out)/'demo')
    if a.mode in ('batch','all'): run_batch(c,a.repeats,Path(a.out)/'batch')
    if a.mode in ('plot','all'): plots(Path(a.out)/'batch')
    if a.mode in ('report','all'): report(Path(a.out)/'batch')
