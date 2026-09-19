"""Image-right experiment: separate information and objective shocks, feedback factorial."""
import argparse
import json
import hashlib
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.channel_choice import ChannelConfig,simulate_channels
from src.customer_network import CustomerNetwork
from src.continuous_competition import ContinuousCompetition,window_metrics,audit_continuous
from experiments.competition import design


def configurations(spec):
    old=json.loads(Path(spec['base_config']).read_text(encoding='utf-8'))
    data=json.loads(Path(old['base_config']).read_text(encoding='utf-8'));data.update(old['overrides'])
    c=replace(design(ChannelConfig(**data))['both'],n=spec['n'],steps=spec['steps'],
        initial_dispute=0.,new_dispute_rate=0.,tie_break='random',domestic_capacity=3,
        customer_capacity=6,domestic_implementation_capacity=6,firm_capacity=2)
    variants={}
    for shock in spec['shocks']:
        for rho in spec['rhos']:
            for feedback in [False,True]:
                name=f'{shock}_r{rho}_f{int(feedback)}'
                # rho=0 removes neighbor outcome/action observation as well as entry diffusion.
                cc=replace(c,observation_probability=0. if rho==0 else 1.)
                variants[name]=(cc,ContinuousCompetition(**spec['dynamics'],rho=rho,shock=shock,feedback=feedback))
    for horizon in spec['extra_horizons']:
        for feedback in [False,True]:
            name=f'objective_r0.6_f{int(feedback)}_h{horizon}'
            variants[name]=(c,ContinuousCompetition(**{**spec['dynamics'],'active_horizon':horizon},
                rho=.6,shock='objective',feedback=feedback))
    return variants,old['customer_network']


def interval(x,samples):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=np.nan,low=np.nan,high=np.nan)
    rng=np.random.default_rng(290103);boot=rng.choice(x,(samples,len(x)),replace=True).mean(axis=1)
    return dict(n=len(x),mean=x.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975))


METRICS=['cases','customer_users','customer_employees','domestic_use','customer_use','both_use',
         'first_customer','d_to_e','resolved','unresolved','potential','blocked','prevented','mean_prevention','knowledge_share']


def report(out,spec):
    m=pd.read_csv(out/'windows.csv');tr=pd.read_csv(out/'trajectories.csv');samples=spec['bootstrap_samples']
    summaries=[];contrasts=[]
    def record(name,contrast,x,metric):contrasts.append(dict(scenario=name,contrast=contrast,metric=metric,**interval(x,samples)))
    for (name,period),g in m.groupby(['scenario','period']):
        for metric in METRICS:summaries.append(dict(scenario=name,period=period,metric=metric,**interval(g.sort_values('seed')[metric],samples)))
    for name,g in m.groupby('scenario'):
        periods={period:v.set_index('seed') for period,v in g.groupby('period')}
        for metric in METRICS:
            record(name,'late_minus_before',periods['late_after'][metric]-periods['before'][metric],metric)
            record(name,'late_minus_early_after',periods['late_after'][metric]-periods['early_after'][metric],metric)
        if '_h' not in name:
            row=g.iloc[0];ref=f'none_r{row.rho}_f{int(row.feedback)}'
            control=m[(m.scenario==ref)]
            cp={period:v.set_index('seed') for period,v in control.groupby('period')}
            for metric in METRICS:
                record(name,'shock_difference_in_changes',
                    (periods['late_after'][metric]-periods['before'][metric])-(cp['late_after'][metric]-cp['before'][metric]),metric)
    for name,g in m[(m.period=='late_after')&m.feedback.astype(bool)].groupby('scenario'):
        control=m[(m.scenario==name.replace('_f1','_f0'))&(m.period=='late_after')].set_index('seed')
        a=g.set_index('seed')
        for metric in METRICS:record(name,'feedback_minus_off_late',a[metric]-control[metric],metric)
    st=pd.DataFrame(summaries);ct=pd.DataFrame(contrasts)
    st.to_csv(out/'intervals.csv',index=False);ct.to_csv(out/'contrasts.csv',index=False)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    for shock,title in [('objective','客观客户审核条件变化'),('information','一次性入口信息告知'),('none','无外生冲击')]:
        fig,axs=plt.subplots(2,2,figsize=(12,8))
        for rho in spec['rhos']:
            for feedback in [0,1]:
                name=f'{shock}_r{rho}_f{feedback}';label=f'{"无同事传播" if rho==0 else "有同事传播"}／{"有预防" if feedback else "无预防"}'
                g=tr[tr.scenario==name]
                for ax,key,ylabel in zip(axs.flat,['new_disputes','new_customer','knowledge_share','mean_prevention'],
                    ['每轮新争议案件（10轮平滑）','每轮客户提交案件（10轮平滑）','知道客户入口的员工比例','企业预防投入指数']):
                    panel=g.pivot(index='time',columns='seed',values=key)
                    if key in ['new_disputes','new_customer']:panel=panel.rolling(10,min_periods=10).mean()
                    y=panel.mean(axis=1);sd=panel.std(axis=1)
                    ax.plot(panel.index,y,label=label);ax.fill_between(panel.index,y-sd,y+sd,alpha=.1)
                    ax.set(xlabel='轮次',ylabel=ylabel);ax.grid(alpha=.2)
                    ax.axvline(spec['dynamics']['shock_time'],color='gray',ls='--')
        h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center')
        fig.suptitle(title+'｜均值±1标准差｜灰线为预定冲击时点',y=.92)
        fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(out/f'{shock}_dynamics.png',dpi=130,bbox_inches='tight');plt.close(fig)
    # Equal followup use proportions: a different denominator from calendar counts.
    fig,axs=plt.subplots(1,3,figsize=(14,5))
    for ax,(shock,title) in zip(axs,[('none','无冲击'),('objective','客观变化'),('information','信息告知')]):
        for rho in spec['rhos']:
            for feedback in [0,1]:
                name=f'{shock}_r{rho}_f{feedback}'
                g=st[(st.scenario==name)&(st.metric=='customer_use')].set_index('period').loc[['before','early_after','late_after']]
                ax.errorbar(range(3),g['mean']*100,yerr=[(g['mean']-g.low)*100,(g.high-g['mean'])*100],marker='o',capsize=2,
                    label=f'{"无传播" if rho==0 else "有传播"}／{"有预防" if feedback else "无预防"}')
        ax.set(title=title,xticks=range(3),xticklabels=['前期','冲击后早期','后期'],ylabel='24轮内客户使用／实际新争议案件（%）');ax.grid(alpha=.2)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=4)
    fig.suptitle('同龄案件的客户采用趋势｜逐项95%区间｜不同于投诉总人数',y=.9)
    fig.tight_layout(rect=[0,0,1,.85]);fig.savefig(out/'cohort_usage.png',dpi=130,bbox_inches='tight');plt.close(fig)
    def v(name,period,metric):return st[(st.scenario==name)&(st.period==period)&(st.metric==metric)].iloc[0]
    def z(name,contrast,metric):return ct[(ct.scenario==name)&(ct.contrast==contrast)&(ct.metric==metric)].iloc[0]
    def ci(row,scale=100):return f'{row["mean"]*scale:.2f} [{row.low*scale:.2f}, {row.high*scale:.2f}]'
    lines=['# 右图第一轮：持续争议、传播与企业预防','',f'完成{m.scenario.nunique()*m.seed.nunique()}次仿真，{m.seed.nunique()}个连续种子。未现实校准。',
        '窗口：前期10—39、冲击后早期60—89、后期110—139轮；每案观察24轮。冲击第50轮。',
        '人数与案件数分开记录；同一员工可有多个不同争议，渠道使用比例以案件为分母，去重人数另列。',
        '## 使用水平与后续趋势','|情景|后期新争议案数|后期客户去重员工数|后期国内使用|后期客户使用|后期减冲击后早期客户使用（百分点及95%区间）|',
        '|---|---:|---:|---:|---:|---|']
    for name in m.scenario.unique():
        lines.append(f'|{name}|{v(name,"late_after","cases")["mean"]:.1f}|{v(name,"late_after","customer_employees")["mean"]:.1f}|'+
            f'{v(name,"late_after","domestic_use")["mean"]:.2%}|{v(name,"late_after","customer_use")["mean"]:.2%}|{ci(z(name,"late_minus_early_after","customer_use"))}|')
    lines+=['','## 企业预防的配对作用（开启减关闭，后期）',
        '|情景|新争议案件数差|客户去重员工数差|客户使用比例差（百分点）|','|---|---|---|---|']
    for name in m[m.feedback.astype(bool)].scenario.unique():
        lines.append(f'|{name}|{ci(z(name,"feedback_minus_off_late","cases"),1)}|{ci(z(name,"feedback_minus_off_late","customer_employees"),1)}|{ci(z(name,"feedback_minus_off_late","customer_use"))}|')
    lines+=['','## 不能作出的推断',
        '传播开启同时允许入口知识传递和邻居信息观察；关闭传播不等于关闭本人案件处理。其总效应不是纯知识传播的独立效应。',
        '预防规则预设投入抑制争议机会，其作用方向部分来自结构假设。是否改变渠道比例、是否抵消传播须看对照，不能由争议人数下降认定。',
        '客观冲击只增加客户审核概率，信息冲击只增加入口知晓，不直接提高员工成功预期；两者不能混作一个新闻事件。',
        '行动期满仍未解决的旧案保留未解决状态，但停止后续处理并允许新案；48与96轮行动期对照用于诊断退出规则。',
        '未看到持续增长不等于证明长期稳定；164轮和两个后期窗口不能证明无限期均衡。',
        '本轮每次使用者人数可能受重复案件影响，明确给出去重人数。所有区间逐项未多重比较调整，图中时间轨迹阴影为标准差而非置信区间。']
    (out/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')


def run(config,out,repeats):
    spec=json.loads(Path(config).read_text(encoding='utf-8'));variants,net=configurations(spec)
    out.mkdir(parents=True,exist_ok=True);windows=[];trajectories=[];cases=[]
    for seed in range(repeats):
        potential=None
        for name,(c,dyn) in variants.items():
            hook=replace(dyn);r=simulate_channels(c,seed,customer_network=CustomerNetwork(**net),dynamics=hook);audit_continuous(r)
            if potential is None:potential=hook.potential.copy()
            else:np.testing.assert_array_equal(potential,hook.potential)
            for period,(start,end) in spec['windows'].items():
                assert end+spec['followup']<=c.steps
                row=window_metrics(r['cases'],start,end,spec['followup']);g=r['dynamics'].query('@start<=time<@end')
                row.update({k:int(g[k].sum()) for k in ['potential','blocked','prevented']})
                row.update({k:float(g[k].mean()) for k in ['mean_prevention','knowledge_share']})
                windows.append(dict(scenario=name,seed=seed,period=period,rho=dyn.rho,feedback=dyn.feedback,**row))
            tr=r['trajectory'].merge(r['dynamics'].drop(columns='new_disputes'),on='time')
            trajectories.append(tr.assign(scenario=name,seed=seed));cases.append(r['cases'].assign(scenario=name,seed=seed))
            if seed==0:
                folder=out/'demo'/name;folder.mkdir(parents=True,exist_ok=True)
                for item in ['events','retirements','firm_dynamics']:r[item].to_csv(folder/f'{item}.csv',index=False)
        print(f'Completed paired seed {seed+1}/{repeats}',flush=True)
    pd.DataFrame(windows).to_csv(out/'windows.csv',index=False)
    pd.concat(trajectories,ignore_index=True).to_csv(out/'trajectories.csv',index=False)
    pd.concat(cases,ignore_index=True).to_csv(out/'cases.csv',index=False)
    report(out,spec)
    paths=[config,__file__,'src/channel_choice.py','src/continuous_competition.py']
    (out/'verification.json').write_text(json.dumps(dict(runs=len(variants)*repeats,seeds=list(range(repeats)),spec=spec,
        potential_opportunities_paired=True,scenarios={k:dict(config=asdict(c),dynamics=asdict(d)) for k,(c,d) in variants.items()},
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/continuous_competition.json')
    p.add_argument('--out',default='outputs/continuous_competition_v1');p.add_argument('--repeats',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text(encoding='utf-8'));n=a.repeats or spec['repeats']
    if n<2:p.error('At least two seeds')
    run(a.config,Path(a.out),n)
