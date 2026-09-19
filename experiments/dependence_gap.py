"""Image-middle dependence heterogeneity, fixed rank groups, no assumed inverted U."""
import argparse
import hashlib
import json
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.channel_choice import ChannelConfig,world,simulate_channels,firm_customer_probability
from src.cohort_trends import CohortSchedule
from src.customer_network import CustomerNetwork
from experiments.competition import design
from experiments.channel_experiments import audit

LABELS={'main':'两渠道均可达','no_perceived':'关闭依赖的直接预期作用','no_objective':'关闭依赖的客观作用',
        'narrow':'依赖差异减半','limited_access':'恢复部分客户不可达'}
METRICS=['domestic_use','customer_use','resolved','customer_success','first_customer','both_use','d_to_e']


def configurations(spec):
    old=json.loads(Path(spec['base_config']).read_text(encoding='utf-8'))
    data=json.loads(Path(old['base_config']).read_text(encoding='utf-8')); data.update(old['overrides'])
    # Materials disabled to isolate the middle-image mechanism, not jointly tune it.
    base=replace(design(ChannelConfig(**data))['both'],tie_break='random',
        customer_availability=1.,employee_reachability=1.)
    rows={}
    for name in spec['variants']:
        for mean in spec['means']:
            c=replace(base,dependence_mean=mean,dependence_spread=spec['spread'])
            if name=='no_perceived':c=replace(c,perceived_dependence=0.)
            if name=='no_objective':c=replace(c,dependence_effect=0.,firm_customer_base=.525)
            if name=='narrow':c=replace(c,dependence_spread=spec['narrow_spread'])
            if name=='limited_access':c=replace(c,customer_availability=.8,employee_reachability=.9)
            rows[(name,mean)]=c
    return rows,CohortSchedule(**old['schedule']),old['customer_network']


def rank_groups(firms):
    order=firms.sort_values(['dependence','firm']).firm.to_numpy()
    half=len(order)//2; tail=max(1,len(order)//3)
    return {'all':set(order),'low':set(order[:half]),'high':set(order[-half:]),
            'low_tail':set(order[:tail]),'high_tail':set(order[-tail:])}


def measure(cases,s,groups):
    rows=[]
    for period,cohorts in [('early',range(2)),('late',range(s.cohorts-2,s.cohorts))]:
        for group,ids in groups.items():
            g=cases[cases.cohort.isin(cohorts)&cases.firm.isin(ids)].copy(); n=len(g)
            end=g.cohort*s.spacing+s.followup
            d=g.d_submitted.lt(end); e=g.e_submitted.lt(end); solved=g.resolved.lt(end)
            success=solved&g.resolved_by.fillna('').isin(['e','d+e'])
            share=lambda x:float(x.sum()/n) if n else np.nan
            firm_rates=g.assign(solved=solved).groupby('firm').solved.mean()
            rows.append(dict(period=period,group=group,cases=n,customer_users=int(e.sum()),
                domestic_use=share(d),customer_use=share(e),resolved=share(solved),
                customer_success=float((success&e).sum()/e.sum()) if e.any() else np.nan,
                first_customer=share(e&(~d|g.e_submitted.lt(g.d_submitted))),
                both_use=share(d&e),d_to_e=share(d&e&g.d_submitted.lt(g.e_submitted)),
                firm_equal_resolved=float(firm_rates.mean()) if len(firm_rates) else np.nan,
                firms_observed=len(firm_rates)))
    return pd.DataFrame(rows)


def interval(values,samples):
    x=np.asarray(values); x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=np.nan,low=np.nan,high=np.nan)
    rng=np.random.default_rng(190903); boot=rng.choice(x,(samples,len(x)),replace=True).mean(axis=1)
    return dict(n=len(x),mean=x.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975))


def report(out,spec):
    data=pd.read_csv(out/'metrics.csv'); cols=METRICS+['firm_equal_resolved']; samples=spec['bootstrap_samples']
    gaps=[]; stats=[]
    for (name,mean,seed,period),g in data.groupby(['variant','dependence_mean','seed','period']):
        g=g.set_index('group')
        for grouping,hi,lo in [('halves','high','low'),('tails','high_tail','low_tail')]:
            gaps.append(dict(variant=name,dependence_mean=mean,seed=seed,period=period,grouping=grouping,
                **{k:g.loc[hi,k]-g.loc[lo,k] for k in cols}))
    gaps=pd.DataFrame(gaps); gaps.to_csv(out/'gaps_by_seed.csv',index=False)
    for (name,mean,group,period),g in data.groupby(['variant','dependence_mean','group','period']):
        for k in cols:stats.append(dict(variant=name,dependence_mean=mean,group=group,period=period,metric=k,
            **interval(g.sort_values('seed')[k],samples)))
    levels=pd.DataFrame(stats); levels.to_csv(out/'levels.csv',index=False)
    records=[]
    for (name,mean,grouping,period),g in gaps.groupby(['variant','dependence_mean','grouping','period']):
        for k in cols:records.append(dict(variant=name,dependence_mean=mean,grouping=grouping,period=period,metric=k,
            **interval(g.sort_values('seed')[k],samples)))
    gs=pd.DataFrame(records); gs.to_csv(out/'gap_intervals.csv',index=False)
    # Prespecified center=.5, endpoints=.1/.9: don't select observed peak after running.
    contrasts=[]
    for (name,grouping),g in gaps[gaps.period=='late'].groupby(['variant','grouping']):
        for k in cols:
            p=g.pivot(index='seed',columns='dependence_mean',values=k)
            for contrast,v in [('middle_minus_low',p[.5]-p[.1]),('middle_minus_high',p[.5]-p[.9]),
                               ('middle_minus_endpoint_average',p[.5]-(p[.1]+p[.9])/2)]:
                contrasts.append(dict(variant=name,grouping=grouping,metric=k,contrast=contrast,**interval(v,samples)))
    cs=pd.DataFrame(contrasts); cs.to_csv(out/'shape_contrasts.csv',index=False)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for name,label in LABELS.items():
        for ax,k in zip(axs[:2],['customer_use','customer_success']):
            g=levels[(levels.variant==name)&(levels.group=='all')&(levels.period=='late')&(levels.metric==k)].sort_values('dependence_mean')
            ax.plot(g.dependence_mean,g['mean'],marker='o',label=label); ax.fill_between(g.dependence_mean,g.low,g.high,alpha=.1)
        g=gs[(gs.variant==name)&(gs.grouping=='halves')&(gs.period=='late')&(gs.metric=='resolved')].sort_values('dependence_mean')
        axs[2].plot(g.dependence_mean,g['mean']*100,marker='o',label=label)
        axs[2].fill_between(g.dependence_mean,g.low*100,g.high*100,alpha=.1)
    for ax,label in zip(axs,['客户使用／争议案件','客户路径成功／客户使用案件','高减低依赖组解决比例（百分点）']):
        ax.set(xlabel='企业平均客户依赖参数（非境外收入占比）',ylabel=label); ax.grid(alpha=.2)
    axs[2].axhline(0,color='gray',ls='--'); h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',ncol=3); fig.suptitle('客户依赖与员工结果差距｜后两批｜逐项95%区间｜未校准',y=.87)
    fig.tight_layout(rect=[0,0,1,.82]); fig.savefig(out/'dependence_comparison.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for grouping,k,label in [('halves','resolved','三高减三低：案件加权'),('tails','resolved','两高减两低：案件加权'),
                             ('halves','firm_equal_resolved','三高减三低：企业等权')]:
        g=gs[(gs.variant=='main')&(gs.grouping==grouping)&(gs.period=='late')&(gs.metric==k)].sort_values('dependence_mean')
        ax.errorbar(g.dependence_mean,g['mean']*100,yerr=[(g['mean']-g.low)*100,(g.high-g['mean'])*100],marker='o',capsize=3,label=label)
    ax.axhline(0,color='gray',ls='--'); ax.set(xlabel='企业平均客户依赖参数',ylabel='高低依赖组解决差距（百分点）',title='分组与权重稳健性｜逐项95%区间')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'group_robustness.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    lines=['# 中图第一轮：客户依赖与案件结果差距','',f'完成{data.variant.nunique()*data.dependence_mean.nunique()*data.seed.nunique()}次运行，连续{data.seed.nunique()}种子。',
        '依赖均值0.1/0.3/0.5/0.7/0.9，主实验离散半幅0.1。每个种子以中间均值的企业排序固定高低组；组员不随均值变化。',
        '不是企业合规差距。依赖不是境外收入占比；材料关闭，未引入企业预防。后期=后两批，每批观察24轮。',
        '## 主实验：两条渠道均可达','|平均依赖|客户使用|客户路径成功比例|高依赖解决|低依赖解决|差距95%区间（百分点）|',
        '|---|---:|---:|---:|---:|---|']
    def val(mean,group,k):return levels[(levels.variant=='main')&(levels.dependence_mean==mean)&(levels.group==group)&(levels.period=='late')&(levels.metric==k)].iloc[0]
    for mean in spec['means']:
        z=gs[(gs.variant=='main')&(gs.dependence_mean==mean)&(gs.grouping=='halves')&(gs.period=='late')&(gs.metric=='resolved')].iloc[0]
        lines.append(f'|{mean}|'+ '|'.join(f'{val(mean,g,k)["mean"]:.2%}' for g,k in [('all','customer_use'),('all','customer_success'),('high','resolved'),('low','resolved')])+f'|{z["mean"]*100:.2f} [{z.low*100:.2f}, {z.high*100:.2f}]|')
    lines+=['','## 倒U形诊断','预先指定中间点0.5，分别比较0.1和0.9，不事后挑峰值。区间逐项，未多重比较校正。',
        '|情景|中间差距减低端差距|中间差距减高端差距|','|---|---|---|']
    for name,label in LABELS.items():
        cells=[]
        for contrast in ['middle_minus_low','middle_minus_high']:
            z=cs[(cs.variant==name)&(cs.grouping=='halves')&(cs.metric=='resolved')&(cs.contrast==contrast)].iloc[0]
            cells.append(f'{z["mean"]*100:.2f} [{z.low*100:.2f}, {z.high*100:.2f}]')
        lines.append(f'|{label}|'+ '|'.join(cells)+'|')
    lines+=['','## 限制',
        '只改变依赖均值不增加企业间离散程度。若曲线不出现山形，不能扩大中间组差异或裁剪端点来制造峰值。',
        '关闭直接预期作用后仍保留通过邻居处理结果的学习；关闭客观依赖作用时后续落实概率固定0.525，仍保留直接预期作用。',
        '客户成功比例受选择、取消与国内解决影响，不是客观概率本身。',
        '一共6家企业，高低组各3家，另检验两端各2家及企业等权；小企业组结果有明显随机不确定性。',
        '较高使用或成功率可能直接反映预设依赖公式，不能仅凭单调上升声称创新。需关注组间差距和机制对照。',
        '本轮未模拟企业违规/预防决策，因此不能声称合规差距，不能定位现实制造业处在曲线哪段。']
    (out/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')


def run(config,out,repeats):
    spec=json.loads(Path(config).read_text(encoding='utf-8')); variants,s,net=configurations(spec)
    out.mkdir(parents=True,exist_ok=True); rows=[]; firms=[]; clipping=[]
    for seed in range(repeats):
        reference=None; groups=None
        for (name,mean),c in variants.items():
            r=simulate_channels(c,seed,customer_network=CustomerNetwork(**net),cohort_schedule=replace(s)); audit(r)
            key=r['cases'][['case_id','cohort','merit']]
            if reference is None:
                reference=key; groups=rank_groups(r['firms'])
            else:pd.testing.assert_frame_equal(reference,key)
            assert groups==rank_groups(r['firms'])
            assert abs(r['firms'].dependence.mean()-mean)<1e-12
            rows.append(measure(r['cases'],s,groups).assign(variant=name,dependence_mean=mean,seed=seed))
            firms.append(r['firms'].assign(variant=name,dependence_mean=mean,seed=seed,
                high=lambda f:f.firm.isin(groups['high'])))
            p=firm_customer_probability(c,r['firms'].dependence.to_numpy())
            clipping.append(dict(variant=name,dependence_mean=mean,seed=seed,min_objective=p.min(),max_objective=p.max(),
                objective_at_boundary=int(((p<=0)|(p>=1)).sum())))
            if seed==0:
                folder=out/'demo'/f'{name}_{mean}'; folder.mkdir(parents=True,exist_ok=True)
                r['cases'].to_csv(folder/'cases.csv',index=False)
        print(f'Completed paired seed {seed+1}/{repeats}',flush=True)
    pd.concat(rows,ignore_index=True).to_csv(out/'metrics.csv',index=False)
    pd.concat(firms,ignore_index=True).to_csv(out/'firms.csv',index=False)
    pd.DataFrame(clipping).to_csv(out/'probability_bounds.csv',index=False); report(out,spec)
    paths=[config,__file__,'src/channel_choice.py','src/cohort_trends.py','src/customer_network.py']
    (out/'verification.json').write_text(json.dumps(dict(runs=len(variants)*repeats,seeds=list(range(repeats)),spec=spec,
        fixed_groups=True,paired_incident_attributes=True,schedule=asdict(s),network=net,
        scenarios={f'{name}_{mean}':asdict(c) for (name,mean),c in variants.items()},
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/dependence_gap.json')
    p.add_argument('--out',default='outputs/dependence_gap_v1');p.add_argument('--repeats',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text(encoding='utf-8'));n=a.repeats or spec['repeats']
    if n<2:p.error('At least two seeds required')
    run(a.config,Path(a.out),n)
