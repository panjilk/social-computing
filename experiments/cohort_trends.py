"""Conditional channel-use trends across comparable incident cohorts."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.channel_choice import ChannelConfig,simulate_channels
from src.customer_network import CustomerNetwork
from src.cohort_trends import CohortSchedule,cohort_measures,first_submission_series
from experiments.competition import design
from experiments.channel_experiments import audit

LABELS={'base':'基准经历传播','no_social':'关闭同事影响','no_outcomes':'关闭结果学习',
        'no_actions':'关闭行动观察效应','network_ws':'小世界网络','network_ba':'优先连接网络',
        'domestic_tight':'国内机构容量紧张','customer_tight':'客户审核容量紧张',
        'cost_low':'两侧成本低20%','cost_high':'两侧成本高20%'}
METRICS=['domestic_use','customer_use','first_d','first_e','first_both','no_submission',
         'd_to_e','e_to_d','resolved','customer_minus_domestic']


def scenarios(base):
    c=design(base)['both']
    return dict(base=c,no_social=replace(c,social_weight=0.,outcome_learning_weight=0.),
        no_outcomes=replace(c,outcome_learning_weight=0.),no_actions=replace(c,social_weight=0.),
        network_ws=replace(c,network='ws'),network_ba=replace(c,network='ba'),
        domestic_tight=replace(c,domestic_capacity=1),customer_tight=replace(c,customer_capacity=6),
        cost_low=replace(c,domestic_cost=c.domestic_cost*.8,customer_cost=c.customer_cost*.8),
        cost_high=replace(c,domestic_cost=c.domestic_cost*1.2,customer_cost=c.customer_cost*1.2))


def summarize(co,samples):
    seed_rows=[]
    for (scenario,seed,population),g in co.groupby(['scenario','seed','population']):
        for period,subset in [('early',g[g.cohort<2]),('late',g[g.cohort>=g.cohort.max()-1])]:
            total=subset.cases.sum()
            row=dict(scenario=scenario,seed=seed,population=population,period=period,cases=int(total))
            row.update({m:float((subset[m]*subset.cases).sum()/total) if total else np.nan for m in METRICS})
            seed_rows.append(row)
    per=pd.DataFrame(seed_rows); rows=[]; rng=np.random.default_rng(9181803)
    def interval(values):
        x=np.asarray(values); x=x[np.isfinite(x)]
        if len(x)==0:return dict(n=0,mean=np.nan,low=np.nan,high=np.nan)
        boot=rng.choice(x,(samples,len(x)),replace=True).mean(axis=1)
        return dict(n=len(x),mean=float(x.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)))
    for (scenario,population),g in per.groupby(['scenario','population']):
        early=g[g.period=='early'].set_index('seed'); late=g[g.period=='late'].set_index('seed')
        for m in METRICS:
            for comparison,x in [('early_level',early[m]),('late_level',late[m]),('late_minus_early',late[m]-early[m])]:
                rows.append(dict(scenario=scenario,population=population,metric=m,comparison=comparison,**interval(x)))
    # The change caused by social mechanisms beyond random cohort composition/load changes.
    for population in per.population.unique():
        delta={}
        for name in ('base','no_social'):
            g=per[(per.population==population)&(per.scenario==name)]
            delta[name]=g[g.period=='late'].set_index('seed')[METRICS]-g[g.period=='early'].set_index('seed')[METRICS]
        for m in METRICS:
            rows.append(dict(scenario='base',population=population,metric=m,
                comparison='trend_base_minus_no_social',**interval(delta['base'][m]-delta['no_social'][m])))
    return per,pd.DataFrame(rows)


def report(out,samples):
    co=pd.read_csv(out/'cohorts.csv'); per,stats=summarize(co,samples)
    per.to_csv(out/'early_late_by_seed.csv',index=False); stats.to_csv(out/'trend_intervals.csv',index=False)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(metric,label) in zip(axes.flat,[('domestic_use','国内使用比例'),('customer_use','客户使用比例'),
                                           ('first_e','首次仅客户比例'),('no_submission','观察期内未投诉比例')]):
        for name in ('base','no_social'):
            g=co[(co.scenario==name)&(co.population=='all')].groupby('cohort')[metric].agg(['mean','std'])
            x=g.index.to_numpy()+1; y=g['mean'].to_numpy(); sd=g['std'].to_numpy()
            ax.plot(x,y,marker='o',label=LABELS[name]); ax.fill_between(x,y-sd,y+sd,alpha=.15)
        ax.set(xlabel='新争议批次（每批均观察24轮）',ylabel=label,xticks=range(1,9)); ax.grid(alpha=.2)
    h,l=axes.flat[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=2)
    fig.suptitle('制度渠道使用趋势｜均值 ±1 标准差｜分母：各批次争议员工',y=.94)
    fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(out/'cohort_trends.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(metric,label) in zip(axes,[('customer_use','客户使用：后两批减前两批（百分点）'),
                                      ('customer_minus_domestic','后两批客户使用减国内使用（百分点）')]):
        comparison='late_minus_early' if metric=='customer_use' else 'late_level'
        s=stats[(stats.population=='all')&(stats.metric==metric)&(stats.comparison==comparison)].set_index('scenario').loc[list(LABELS)]
        y=np.arange(len(s)); ax.errorbar(s['mean']*100,y,xerr=[(s['mean']-s.low)*100,(s.high-s['mean'])*100],fmt='o',capsize=3)
        ax.axvline(0,color='gray',ls='--'); ax.set(yticks=y,yticklabels=[LABELS[x] for x in s.index],xlabel=label)
        ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    fig.suptitle('条件趋势与使用优势｜配对种子bootstrap逐项95%区间')
    fig.tight_layout(); fig.savefig(out/'scenario_trends.png',dpi=160); plt.close(fig)
    def stat(name,m,comparison,pop='all'):
        return stats[(stats.scenario==name)&(stats.metric==m)&(stats.comparison==comparison)&(stats.population==pop)].iloc[0]
    lines=['# 新争议批次下的制度竞争趋势','',
        f'实际完成{co.scenario.nunique()*co.seed.nunique()}次仿真；{co.seed.nunique()}个连续配对种子。每次8批次，每批观察24轮。所有参数未经现实校准。',
        '前期=前2批，后期=后2批；先在各种子内按案件数合并两批，再跨种子等权平均。客户与国内使用允许重叠。','',
        '|情景|前期国内使用|后期国内使用|前期客户使用|后期客户使用|客户变化95%区间（百分点）|',
        '|---|---:|---:|---:|---:|---|']
    for name in LABELS:
        d0=stat(name,'domestic_use','early_level'); d1=stat(name,'domestic_use','late_level')
        e0=stat(name,'customer_use','early_level'); e1=stat(name,'customer_use','late_level'); change=stat(name,'customer_use','late_minus_early')
        lines.append(f'|{LABELS[name]}|{d0["mean"]:.2%}|{d1["mean"]:.2%}|{e0["mean"]:.2%}|{e1["mean"]:.2%}|[{change.low*100:.2f}, {change.high*100:.2f}]|')
    lines+=['','## 基准详细比较','', '|指标|前期|后期|后期减前期95%区间（百分点）|','|---|---:|---:|---|']
    names=dict(first_d='首次仅国内',first_e='首次仅客户',first_both='首次同时',no_submission='未投诉',d_to_e='国内→客户追加',e_to_d='客户→国内追加',resolved='已解决')
    for m,label in names.items():
        a=stat('base',m,'early_level'); b=stat('base',m,'late_level'); z=stat('base',m,'late_minus_early')
        lines.append(f'|{label}|{a["mean"]:.2%}|{b["mean"]:.2%}|[{z.low*100:.2f}, {z.high*100:.2f}]|')
    for m,label in [('customer_use','客户使用'),('domestic_use','国内使用')]:
        z=stat('base',m,'trend_base_minus_no_social')
        lines.append(f'\n{label}的趋势差（基准的后减前，减去关闭同事影响的后减前）：{z["mean"]*100:.2f}个百分点，95%区间[{z.low*100:.2f}, {z.high*100:.2f}]。')
    lines+=['','## 判断边界',
        '这是固定情景下后续争议批次的条件预测，不是对现实未来月份/年份的已校准预测。tick为抽象轮次。没有模拟新闻事件的确定成功率。',
        '国内/客户覆盖以各批次全部争议员工为分母；另有both_reachable人群统计。首次三类加未投诉=100%。相同观察期限排除新批次随访不足。',
        '客户使用增长与超过国内分别判断；区间包含零不等于证明不变。区间逐项，未多重比较调整。',
        '关闭同事影响的对照保留自身等待/追加；早晚差仍可来自随机批次组成和处理负荷。社会机制贡献以配对趋势差诊断，不由一条上升曲线认定。',
        '后续批次员工在模拟开始就存在于员工网络，只是争议发生更晚；每人至多一次。固定企业规模，无人口增长或新争议预防。',
        '结果学习使用最近18轮的邻居唯一员工结果，从固定先验重新计算，不是无限累积记忆。未受理、正在处理、取消不计成功或失败；已结束失败与实际补救成功分开。',
        '成本与先验相对历史模型有明确修改，用于允许不行动，不是数据校准；成本上下20%情景一并报告。',
        '客户有无与员工可达性沿用随机异质性，未假设人人可以联系客户。容量和网络情景在整个运行内固定，无政策改善冲击。']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def run(config,out,repeats):
    spec=json.loads(Path(config).read_text(encoding='utf-8'))
    data=json.loads(Path(spec['base_config']).read_text(encoding='utf-8')); data.update(spec['overrides'])
    base=ChannelConfig(**data); specs={k:v for k,v in scenarios(base).items() if k in spec['scenarios']}
    if set(specs)!=set(LABELS): raise ValueError('This report requires the complete predefined scenario set')
    out.mkdir(parents=True,exist_ok=True); cohorts=[]; cases=[]; trajectories=[]; beliefs=[]
    for seed in range(repeats):
        ref=None
        for name,c in specs.items():
            schedule=CohortSchedule(**spec['schedule']); net=CustomerNetwork(**spec['customer_network'])
            r=simulate_channels(c,seed,customer_network=net,cohort_schedule=schedule); audit(r)
            ca=r['cases']; assert ca.employee.is_unique
            key=ca[['case_id','employee','created','cohort','merit','reachable','dependence']]
            if ref is None: ref=key.copy()
            else: pd.testing.assert_frame_equal(ref,key)
            m=cohort_measures(ca,schedule)
            valid=m.cases>0
            np.testing.assert_allclose(m.loc[valid,['first_d','first_e','first_both','no_submission']].sum(axis=1),1.)
            assert (m.loc[valid,'both_use']<=m.loc[valid,['domestic_use','customer_use']].min(axis=1)+1e-12).all()
            np.testing.assert_allclose(m.loc[valid,'resolved']+m.loc[valid,'unresolved'],1.)
            tags=dict(scenario=name,seed=seed)
            cohorts.append(m.assign(**tags)); cases.append(ca.assign(**tags))
            tr=r['trajectory'].merge(first_submission_series(ca,c.steps),on='time')
            # Each employee has at most one case, so these are per-round unique people.
            tr['rolling_domestic_users']=tr.new_domestic.rolling(6,min_periods=1).sum()
            tr['rolling_customer_users']=tr.new_customer.rolling(6,min_periods=1).sum()
            trajectories.append(tr.assign(**tags)); beliefs.append(r['belief_history'].assign(**tags))
            if seed==0:
                folder=out/'demo'/name; folder.mkdir(parents=True,exist_ok=True)
                for item in ('events','decisions','employees','firms','customer_edges','customer_queues'):
                    r[item].to_csv(folder/f'{item}.csv',index=False)
        print(f'Completed paired seed {seed+1}/{repeats}',flush=True)
    for name,frames in [('cohorts',cohorts),('cases',cases),('trajectories',trajectories),('beliefs',beliefs)]:
        pd.concat(frames,ignore_index=True).to_csv(out/f'{name}.csv',index=False)
    report(out,spec['bootstrap_samples'])
    with (out/'tests.txt').open('w',encoding='utf-8') as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
    if not tests.wasSuccessful(): raise RuntimeError('Tests failed')
    paths=['src/channel_choice.py','src/cohort_trends.py','src/customer_network.py','experiments/cohort_trends.py',config,spec['base_config']]
    meta=dict(spec=spec,scenarios={k:asdict(v) for k,v in specs.items()},seeds=list(range(repeats)),
        runs=len(specs)*repeats,tests=tests.testsRun,passed=True,paired_incidents=True,equal_followup=True,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    (out/'verification.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/cohort_trends.json')
    p.add_argument('--out',default='outputs/cohort_trends_v1'); p.add_argument('--repeats',type=int)
    a=p.parse_args(); spec=json.loads(Path(a.config).read_text(encoding='utf-8'))
    count=spec['repeats'] if a.repeats is None else a.repeats
    if count<2: p.error('At least two paired seeds required')
    run(a.config,Path(a.out),count)
