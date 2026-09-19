"""Predeclared mechanism checks; no data calibration and no winning-target search."""
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
from src.channel_choice import ChannelConfig,simulate_channels
from src.cohort_trends import CohortSchedule,cohort_measures
from src.customer_network import CustomerNetwork
from experiments.competition import design
from experiments.channel_experiments import audit


def scenarios(spec):
    old=json.loads(Path(spec['base_config']).read_text(encoding='utf-8'))
    data=json.loads(Path(old['base_config']).read_text(encoding='utf-8')); data.update(old['overrides'])
    legacy=design(ChannelConfig(**data))['both']; base=replace(legacy,tie_break='random')
    schedule=CohortSchedule(**old['schedule']); net=old['customer_network']
    rows={}
    def add(name,c=base,s=schedule,network=net,family='checks',label=None):
        rows[name]=(replace(c,steps=(s.cohorts-1)*s.spacing+s.followup),s,network,family,label or name)
    add('legacy',legacy,label='旧平局规则'); add('base',label='随机平局基准')
    learned=replace(base,wait_learning_weight=spec['wait_learning_weight'])
    add('wait_learning',learned,label='学习可见等待')
    add('wait_no_outcomes',replace(learned,outcome_learning_weight=0.),label='等待学习／无结果学习')
    for h in spec['followups']: add(f'followup_{h}',s=replace(schedule,followup=h),label=f'观察{h}轮')
    for h in spec['spacings']: add(f'spacing_{h}',s=replace(schedule,spacing=h),label=f'批次间隔{h}轮')
    for h in spec['memories']: add(f'memory_{h}',s=replace(schedule,outcome_window=h),label=f'结果记忆{h}轮')
    # Exact per-merit stage probability match; capacities made nonbinding.
    # D stage1=.6*(.5+.5m), E stage1=.3+.3m; stage2=.85 for both.
    sym=replace(base,customer_availability=1.,employee_reachability=1.,
        domestic_success=.6,customer_review_base=.3,customer_review_merit=.3,
        domestic_implementation_success=.85,firm_customer_base=.85,dependence_effect=0.,
        perceived_dependence=0.,belief_spread=0.,domestic_prior=.5,customer_prior=.5,
        domestic_capacity=base.n,customer_capacity=base.n,domestic_implementation_capacity=base.n,
        firm_capacity=base.n,domestic_delay=2,customer_delay=2,domestic_implementation_delay=1,
        firm_delay=1,social_weight=0.,outcome_learning_weight=0.)
    add('symmetric_random',sym,network=None,label='对称／随机平局')
    add('symmetric_legacy',replace(sym,tie_break='legacy'),network=None,label='对称／旧平局')
    for reach in spec['reachabilities']:
        for ratio in spec['customer_cost_ratios']:
            add(f'grid_r{reach}_c{ratio}',replace(base,customer_availability=1.,
                employee_reachability=reach,customer_cost=base.domestic_cost*ratio),
                family='grid',label=f'可达{reach}／成本比{ratio}')
    for p in spec['customer_final_probabilities']:
        add(f'objective_{p}',replace(base,firm_customer_base=p,dependence_effect=0.),
            family='objective',label=f'客户后续落实概率{p}')
    for p in spec['customer_priors']:
        add(f'prior_{p}',replace(base,customer_prior=p),family='prior',label=f'客户初始预期{p}')
    return rows


METRICS=['domestic_use','customer_use','customer_minus_domestic','first_d','first_e',
         'first_both','domestic_only','customer_only','both_use','no_submission','d_to_e','e_to_d','resolved']


def summarize(co,samples):
    records=[]
    for (name,seed,pop),g in co.groupby(['scenario','seed','population']):
        for period,sub in [('early',g[g.cohort<2]),('late',g[g.cohort>=g.cohort.max()-1])]:
            total=sub.cases.sum()
            records.append(dict(scenario=name,seed=seed,population=pop,period=period,
                **{m:(sub[m]*sub.cases).sum()/total if total else np.nan for m in METRICS}))
    per=pd.DataFrame(records); stats=[]; rng=np.random.default_rng(9182301)
    def save(name,pop,m,contrast,x):
        x=np.asarray(x); x=x[np.isfinite(x)]
        if not len(x): return
        boot=rng.choice(x,(samples,len(x)),replace=True).mean(axis=1)
        stats.append(dict(scenario=name,population=pop,metric=m,contrast=contrast,n=len(x),
            mean=x.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975)))
    for (name,pop),g in per.groupby(['scenario','population']):
        early=g[g.period=='early'].set_index('seed'); late=g[g.period=='late'].set_index('seed')
        ref=per[(per.scenario=='base')&(per.population==pop)&(per.period=='late')].set_index('seed')
        for m in METRICS:
            save(name,pop,m,'late',late[m]); save(name,pop,m,'change',late[m]-early[m])
            save(name,pop,m,'late_minus_base',late[m]-ref[m])
    return per,pd.DataFrame(stats)


def report(out,spec,variants):
    co=pd.read_csv(out/'cohorts.csv'); per,st=summarize(co,spec['bootstrap_samples'])
    per.to_csv(out/'periods.csv',index=False); st.to_csv(out/'intervals.csv',index=False)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
    def values(name,pop='all',metric='customer_minus_domestic',contrast='late'):
        return st[(st.scenario==name)&(st.population==pop)&(st.metric==metric)&(st.contrast==contrast)].iloc[0]
    checks=[k for k,v in variants.items() if v[3]=='checks']
    fig,axs=plt.subplots(1,2,figsize=(13,7))
    for ax,contrast,title in zip(axs,['late','change'],['后期客户减国内使用（百分点）','客户使用：后期减前期（百分点）']):
        metric='customer_minus_domestic' if contrast=='late' else 'customer_use'
        g=pd.DataFrame([values(k,metric=metric,contrast=contrast) for k in checks])
        ax.errorbar(g['mean']*100,np.arange(len(g)),xerr=[(g['mean']-g.low)*100,(g.high-g['mean'])*100],fmt='o',capsize=3)
        ax.set(yticks=range(len(g)),yticklabels=[variants[k][4] for k in checks],xlabel=title)
        ax.axvline(0,color='gray',ls='--'); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    fig.suptitle('诊断与稳健性｜种子 bootstrap 逐项95%区间（未校准）'); fig.tight_layout()
    fig.savefig(out/'robustness.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,pop,title in zip(axs,['all','both_reachable'],['全部争议员工','两渠道均可达员工']):
        arr=np.array([[values(f'grid_r{r}_c{c}',pop)['mean']*100 for c in spec['customer_cost_ratios']] for r in spec['reachabilities']])
        im=ax.imshow(arr,cmap='RdBu_r',vmin=-100,vmax=100)
        for i,r in enumerate(spec['reachabilities']):
            for j,c in enumerate(spec['customer_cost_ratios']):
                z=values(f'grid_r{r}_c{c}',pop); label='客户较多' if z.low>0 else '国内较多' if z.high<0 else '尚不明确'
                ax.text(j,i,f'{arr[i,j]:.1f}\n{label}',ha='center',va='center',fontsize=10)
        ax.set(xticks=range(len(spec['customer_cost_ratios'])),xticklabels=spec['customer_cost_ratios'],
            yticks=range(len(spec['reachabilities'])),yticklabels=spec['reachabilities'],
            xlabel='客户／国内基础成本比',ylabel='员工客户可达概率（企业均有客户）',title=title)
    fig.colorbar(im,ax=axs.tolist(),label='后期客户减国内使用（百分点）',shrink=.8)
    fig.suptitle('竞争格局｜颜色为均值，文字按逐项95%区间判定｜探索参数')
    fig.savefig(out/'competition_map.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    path=per[(per.period=='late')&(per.population=='all')].groupby('scenario')[['domestic_only','customer_only','both_use','no_submission']].mean().loc[checks]
    ax=path.plot.bar(stacked=True,figsize=(12,6),color=['#4477aa','#ee7733','#228833','#bbbbbb'])
    ax.set_xticklabels([variants[k][4] for k in checks],rotation=30,ha='right'); ax.set_ylabel('后期争议员工比例')
    ax.legend(['仅国内','仅客户','双渠道并用','未投诉'],ncol=4,loc='upper center',bbox_to_anchor=(.5,1.15))
    ax.set_ylim(0,1); ax.figure.tight_layout(); ax.figure.savefig(out/'exclusive_paths.png',dpi=160); plt.close(ax.figure)
    lines=['# 制度竞争模型优化实验','',f'完成{co.seed.nunique()}个连续配对种子 × {len(variants)}情景。未现实校准。',
        '前期=前两批；后期=后两批。种子内按案件数加权，跨种子等权。使用允许重叠；互斥路径合计100%。',
        '区间为种子bootstrap逐项95%区间，未经多重比较校正；不能将探索格点作为显著性筛选。','',
        '|情景|后期国内|后期客户|客户减国内95%区间（百分点）|客户早晚变化95%区间（百分点）|',
        '|---|---:|---:|---|---|']
    for name,v in variants.items():
        d=values(name,metric='domestic_use'); e=values(name,metric='customer_use'); z=values(name); t=values(name,metric='customer_use',contrast='change')
        lines.append(f'|{v[4]}|{d["mean"]:.2%}|{e["mean"]:.2%}|[{z.low*100:.2f}, {z.high*100:.2f}]|[{t.low*100:.2f}, {t.high*100:.2f}]|')
    lines+=['','## 解释限制','对称实验关闭社会学习，匹配各阶段概率和延迟，容量不约束；个体成本仍独立同分布。它检验渠道标签偏向，不要求单个种子恰好50/50。',
        '可达性网格固定企业有客户=1，员工可达概率与成本比联合变化；因此网格与原基准的差不能归因于单一变量。',
        '客观概率扫描只改变客户后续落实概率；主观扫描只改变初始客户预期，二者分开运行。',
        '等待学习使用自身和可见邻居的已结束时长、未结束案件年龄下界；不是无偏平均处理时长或生存估计。',
        '观察期限实验改变统计窗口和运行长度；间隔实验改变负荷与信息年龄；不把它们当成现实年份预测。',
        '改变处理时间会改变消耗的轮次事件随机数；配对相同种子不是保证所有案件遭遇相同结果。']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def run(config,out,repeats):
    spec=json.loads(Path(config).read_text(encoding='utf-8')); variants=scenarios(spec)
    out.mkdir(parents=True,exist_ok=True); co=[]; waits=[]
    for seed in range(repeats):
        reference=None
        for name,(c,s,net,family,label) in variants.items():
            r=simulate_channels(c,seed,customer_network=CustomerNetwork(**net) if net else None,cohort_schedule=replace(s)); audit(r)
            key=r['cases'][['employee','cohort','merit']].reset_index(drop=True)
            if reference is None: reference=key
            else: pd.testing.assert_frame_equal(reference,key)
            m=cohort_measures(r['cases'],s); valid=m.cases>0
            np.testing.assert_allclose(m.loc[valid,['domestic_only','customer_only','both_use','no_submission']].sum(axis=1),1.)
            co.append(m.assign(scenario=name,seed=seed))
            if len(r['decisions']):
                waits.append(r['decisions'].groupby('time')[['expected_wait_d','expected_wait_e']].mean().reset_index().assign(scenario=name,seed=seed))
            if seed==0:
                folder=out/'demo'/name; folder.mkdir(parents=True,exist_ok=True)
                r['cases'].to_csv(folder/'cases.csv',index=False)
                if name in ('base','wait_learning','symmetric_random'): r['decisions'].to_csv(folder/'decisions.csv',index=False)
        print(f'Completed paired seed {seed+1}/{repeats}',flush=True)
    pd.concat(co,ignore_index=True).to_csv(out/'cohorts.csv',index=False)
    pd.concat(waits,ignore_index=True).to_csv(out/'decision_waits.csv',index=False)
    report(out,spec,variants)
    paths=[config,'src/channel_choice.py','src/cohort_trends.py','src/wait_learning.py',__file__]
    meta=dict(seeds=list(range(repeats)),runs=len(variants)*repeats,spec=spec,
        scenarios={k:dict(config=asdict(v[0]),schedule=asdict(v[1]),network=v[2]) for k,v in variants.items()},
        paired_incident_identity=True,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    (out/'verification.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/competition_refinement.json')
    p.add_argument('--out',default='outputs/competition_refinement_v1'); p.add_argument('--repeats',type=int)
    a=p.parse_args(); spec=json.loads(Path(a.config).read_text(encoding='utf-8'))
    n=spec['repeats'] if a.repeats is None else a.repeats
    if n<2: p.error('At least two seeds required')
    run(a.config,Path(a.out),n)
