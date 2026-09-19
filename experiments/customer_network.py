"""Paired concentration experiment. No policy-improvement treatments."""
import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.channel_choice import ChannelConfig,simulate_channels
from src.customer_network import CustomerNetwork
from experiments.competition import design,measures
from experiments.channel_experiments import audit


def run(config,out,repeats):
    spec=json.loads(Path(config).read_text(encoding='utf-8'))
    c=ChannelConfig(**json.loads(Path(spec['base_config']).read_text(encoding='utf-8')))
    c=replace(c,customer_availability=spec['customer_availability'],
              employee_reachability=spec['employee_reachability'])
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    endpoints=[]; cases=[]; trajectories=[]; queues=[]; edges=[]
    for seed in range(repeats):
        reference={}
        people=None
        for pooled in (False,True):
            for k in spec['active_customers']:
                for arm,cc in design(c).items():
                    net=CustomerNetwork(spec['customers'],k,pooled)
                    r=simulate_channels(cc,seed,customer_network=net); audit(r)
                    if people is None: people=r['employees']
                    else: pd.testing.assert_frame_equal(people,r['employees'])
                    key=(pooled,arm)
                    if pooled or arm=='domestic_only':
                        if key in reference: pd.testing.assert_frame_equal(reference[key],r['trajectory'])
                        else: reference[key]=r['trajectory']
                    tag=dict(seed=seed,pooled=pooled,active_customers=k,arm=arm)
                    m,tr=measures(r)
                    q=r['customer_queues']; ca=r['cases']; reviewed=ca.e_responded.notna()
                    m.update(review_wait=(ca.loc[reviewed,'e_responded']-ca.loc[reviewed,'e_submitted']).mean(),
                        eligible_queue_rounds=int(q.eligible_left.sum()),
                        review_utilization=float(q.reviewed.sum()/(cc.steps*cc.customer_capacity)),
                        unreviewed_submitted=int((ca.e_submitted.notna()&~reviewed).sum()))
                    endpoints.append(dict(**tag,**m))
                    cases.append(ca.assign(**tag)); trajectories.append(tr.assign(**tag))
                    queues.append(q.assign(**tag)); edges.append(r['customer_edges'].assign(**tag))
                    if seed==0:
                        folder=out/'demo'/f'{pooled}_{k}_{arm}'; folder.mkdir(parents=True,exist_ok=True)
                        r['decisions'].to_csv(folder/'decisions.csv',index=False)
                        r['events'].to_csv(folder/'events.csv',index=False)
        print(f'Completed seed {seed+1}/{repeats}',flush=True)
    ep=pd.DataFrame(endpoints); ep.to_csv(out/'endpoints.csv',index=False)
    for name,frames in [('cases',cases),('trajectories',trajectories),('customer_queues',queues),('customer_edges',edges)]:
        pd.concat(frames,ignore_index=True).to_csv(out/f'{name}.csv',index=False)
    metrics=['customer_use','resolved_share','d_to_e_all','e_to_d_all','review_wait','eligible_queue_rounds']
    paired=[]; rng=np.random.default_rng(917701)
    for pooled in (False,True):
        for k in spec['active_customers']:
            subset=ep[(ep.pooled==pooled)&(ep.active_customers==k)]
            b=subset[subset.arm=='both'].set_index('seed')
            s=subset[subset.arm=='customer_only'].set_index('seed')
            for metric in ['customer_use','resolved_share']:
                d=(b[metric]-s[metric]).to_numpy()
                boot=rng.choice(d,(2000,len(d)),replace=True).mean(axis=1)
                paired.append(dict(pooled=pooled,active_customers=k,metric=metric,
                    comparison='both - customer_only',mean=float(d.mean()),
                    low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975))))
    pd.DataFrame(paired).to_csv(out/'paired_intervals.csv',index=False)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric,label in zip(axes.flat,metrics[:3]+['eligible_queue_rounds'],
             ['客户使用比例','争议解决比例','国内→客户追加比例','累计可审核积压（案件×轮）']):
        for pooled,label2 in [(False,'客户独立容量'),(True,'容量共享诊断')]:
            data=ep[(ep.arm=='both')&(ep.pooled==pooled)].groupby('active_customers')[metric].agg(['mean','std']).sort_index()
            ax.errorbar(data.index,data['mean'],yerr=data['std'],marker='o',capsize=3,label=label2)
        ax.set(xlabel='承接企业连接的客户数（越少越集中）',ylabel=label,xticks=spec['active_customers'])
        ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2)
    fig.suptitle('企业—客户网络集中度：双渠道共存｜均值 ±1 标准差',y=.94)
    fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(out/'customer_network.png',dpi=160); plt.close(fig)
    rows=['# 企业—客户网络首轮实验','',f'实际运行 {len(ep)} 次；{repeats} 个连续配对种子。全部参数为探索值。','',
        '以下为双渠道共存组均值，使用/解决/追加的分母均为全部初始争议案件。',
        '|容量规则|连接客户数|客户使用|解决|国内→客户|客户→国内|累计可审核积压|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for (pooled,k),g in ep[ep.arm=='both'].groupby(['pooled','active_customers']):
        rows.append(f'|{"共享" if pooled else "独立"}|{k}|{g.customer_use.mean():.2%}|{g.resolved_share.mean():.2%}|{g.d_to_e_all.mean():.2%}|{g.e_to_d_all.mean():.2%}|{g.eligible_queue_rounds.mean():.2f}|')
    rows+=['','容量共享时不同连接结构的逐轮结果完全一致，已逐种子断言核对。仅国内组也不受连接结构影响。',
        '独立容量时，每客户每轮2个审核名额，总计12；集中使部分容量无法被案件使用。这是连接与容量分布不匹配的联合机制，不是脱离容量约束的纯拓扑效应。',
        '未审核者可能仍等待，也可能已被国内解决而取消，不能全部算审核失败。review_wait仅统计已审核者。',
        '员工初始不知道连接结构，因此不应解释成网络结构直接改变首次偏好。随机事件按轮次和员工配对；审核时间改变时实现的抽签也会改变。',
        '图中误差棒是重复间标准差；paired_intervals.csv为2000次种子bootstrap的逐项95%区间，未做多重比较调整。',
        '当前是单主要客户的二部网络，不是完整供应链，不代表已确认的论文创新。']
    (out/'RESULTS.md').write_text('\n'.join(rows),encoding='utf-8')
    paths=['src/channel_choice.py','src/customer_network.py','experiments/customer_network.py',config]
    meta=dict(config=spec,base=asdict(c),seeds=list(range(repeats)),runs=len(ep),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        audits_passed=True,pooled_topology_invariance=True,domestic_only_invariance=True)
    (out/'design.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/customer_network.json')
    p.add_argument('--out',default='outputs/customer_network_v1'); p.add_argument('--repeats',type=int)
    a=p.parse_args(); spec=json.loads(Path(a.config).read_text(encoding='utf-8'))
    repeats=spec['repeats'] if a.repeats is None else a.repeats
    if repeats<2: p.error('At least two repeats required')
    run(a.config,a.out,repeats)
