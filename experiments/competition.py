"""Fixed-condition paired comparison of one versus two available channels."""
import argparse
import json
import hashlib
from pathlib import Path
from dataclasses import replace, asdict
import unittest
import numpy as np
import pandas as pd
from src.channel_choice import ChannelConfig, simulate_channels, world
from experiments.channel_experiments import audit, endpoint, metadata

LABELS={'domestic_only':'仅国内渠道','customer_only':'仅客户渠道','both':'双渠道共存'}


def design(base):
    # Fixed cohort avoids treatment-dependent arrival opportunities after resolution.
    core=replace(base,new_dispute_rate=0.,prevention=0.,material_rate=0.,evidence_help=0.,
        perceived_evidence_bonus=0.,domestic_information=0.)
    return {name:replace(core,domestic_enabled=d,customer_enabled=e)
        for name,d,e in [('domestic_only',True,False),('customer_only',False,True),('both',True,True)]}


def measures(result):
    ca=result['cases']; tr=result['trajectory'].copy(); n=len(ca)
    def share(k,den=n): return float(k/den) if den else np.nan
    d=ca.d_submitted.notna(); e=ca.e_submitted.notna()
    de=d&e&(ca.d_submitted<ca.e_submitted)
    ed=d&e&(ca.e_submitted<ca.d_submitted)
    same=d&e&(ca.d_submitted==ca.e_submitted)
    first_d=ca.initial_choice=='d'; first_e=ca.initial_choice=='e'
    m=endpoint(result)
    m.update(domestic_use=share(d.sum()),customer_use=share(e.sum()),
        first_d_all=share(first_d.sum()),first_e_all=share(first_e.sum()),
        first_both_all=share(same.sum()),never_submitted=share(ca.first_submitted.isna().sum()),
        d_to_e_count=int(de.sum()),e_to_d_count=int(ed.sum()),
        d_to_e_all=share(de.sum()),e_to_d_all=share(ed.sum()),
        d_to_e_conditional=share(de.sum(),first_d.sum()),
        e_to_d_conditional=share(ed.sum(),first_e.sum()),
        both_use=share((d&e).sum()),resolved_share=share(ca.resolved.notna().sum()))
    for t in tr.time:
        tr.loc[tr.time==t,'cumulative_domestic_share']=share((ca.d_submitted<=t).sum())
        tr.loc[tr.time==t,'cumulative_customer_share']=share((ca.e_submitted<=t).sum())
        tr.loc[tr.time==t,'new_d_to_e']=int((de&(ca.e_submitted==t)).sum())
        tr.loc[tr.time==t,'new_e_to_d']=int((ed&(ca.d_submitted==t)).sum())
    return m,tr


def run(base,repeats,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    specs=design(base); endpoints=[]; trajectories=[]; cases=[]; events=[]
    for seed in range(repeats):
        reference=None
        for name,c in specs.items():
            w=world(c,seed)
            if reference is None: reference=w
            else:
                for key in ('firm','initial','has_customer','dependence','reachable','merit','beliefs','costs','sharing'):
                    np.testing.assert_array_equal(w[key],reference[key])
                assert set(w['graph'].edges)==set(reference['graph'].edges)
                for key in w['draws']: np.testing.assert_array_equal(w['draws'][key],reference['draws'][key])
            r=simulate_channels(c,seed); audit(r)
            if not c.domestic_enabled: assert r['cases'].d_submitted.isna().all()
            if not c.customer_enabled: assert r['cases'].e_submitted.isna().all()
            assert r['cases'].material_at.isna().all()
            m,tr=measures(r); endpoints.append(dict(scenario=name,seed=seed,**m))
            trajectories.append(tr.assign(scenario=name,seed=seed))
            cases.append(r['cases'].assign(scenario=name,seed=seed))
            events.append(r['events'].assign(scenario=name,seed=seed))
            if seed==0:
                folder=out/'demo'/name; folder.mkdir(parents=True,exist_ok=True)
                for key,value in r.items():
                    if key!='graph': value.to_csv(folder/f'{key}.csv',index=False)
        if (seed+1)%10==0: print(f'Completed {seed+1} paired seeds',flush=True)
    ep=pd.DataFrame(endpoints); ep.to_csv(out/'endpoints.csv',index=False)
    for name,items in [('trajectories',trajectories),('cases',cases),('events',events)]:
        pd.concat(items,ignore_index=True).to_csv(out/f'{name}.csv',index=False)
    ep.groupby('scenario').agg({k:['mean','std'] for k in
        ['domestic_use','customer_use','both_use','resolved_share','unresolved_fraction',
         'first_d_all','first_e_all','first_both_all','never_submitted',
         'd_to_e_all','e_to_d_all','d_to_e_conditional','e_to_d_conditional']}).to_csv(out/'summary.csv')
    paired=[]
    both=ep[ep.scenario=='both'].set_index('seed')
    for single,metrics in [('domestic_only',['domestic_use','resolved_share','unresolved_case_rounds']),
                           ('customer_only',['customer_use','resolved_share','unresolved_case_rounds'])]:
        ref=ep[ep.scenario==single].set_index('seed')
        for metric in metrics:
            diff=both[metric]-ref[metric]
            paired.append(dict(comparison='both - '+single,metric=metric,n=diff.count(),
                mean_difference=diff.mean(),sd_difference=diff.std(),se_difference=diff.std()/np.sqrt(diff.count())))
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    meta=metadata(base,None); meta.update(model='channel_competition_v1',runs=len(ep),
        seeds=list(range(repeats)),scenarios={name:asdict(c) for name,c in specs.items()},
        fixed_cohort=True,information_module=False,uncertainty='mean +/- sample SD, not CI')
    for path in ['experiments/competition.py','configs/competition.json']:
        meta['source_sha256'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (out/'design.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    with (out/'tests.txt').open('w',encoding='utf-8') as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
    if not tests.wasSuccessful(): raise RuntimeError('Tests failed')
    (out/'verification.json').write_text(json.dumps(dict(tests=tests.testsRun,passed=True,
        runs=len(ep),world_pairing_passed=True,accounting_passed=True),indent=2),encoding='utf-8')
    plot_report(out)


def plot_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    installed={f.name for f in fontManager.ttflist}
    plt.rcParams['font.family']=next((n for n in ['Microsoft YaHei','Noto Sans CJK SC','SimHei'] if n in installed),'sans-serif')
    plt.rcParams['axes.unicode_minus']=False
    out=Path(out); ep=pd.read_csv(out/'endpoints.csv'); tr=pd.read_csv(out/'trajectories.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=150)
    metrics={'cumulative_domestic_share':'累计使用国内渠道的案件比例',
        'cumulative_customer_share':'累计使用客户渠道的案件比例',
        'new_additional_channel':'每轮追加另一渠道的案件数','unresolved':'未解决争议案件数'}
    for ax,(metric,label) in zip(axes.flat,metrics.items()):
        for name in LABELS:
            s=tr[tr.scenario==name].groupby('time')[metric].agg(['mean','std'])
            ax.plot(s.index,s['mean'],label=LABELS[name])
            ax.fill_between(s.index,s['mean']-s['std'],s['mean']+s['std'],alpha=.12)
        ax.set(xlabel='轮次',ylabel=label); ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3)
    fig.suptitle('固定条件下的渠道竞争｜均值 ±1 样本标准差')
    fig.tight_layout(rect=(0,0,1,.89)); fig.savefig(out/'competition_dynamics.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=150)
    first=['first_d_all','first_e_all','first_both_all','never_submitted']
    bottom=np.zeros(3)
    for key,label in zip(first,['首次仅国内','首次仅客户','首次同时使用','从未提交']):
        values=ep.groupby('scenario')[key].mean().reindex(LABELS).values
        axes[0].bar(range(3),values,bottom=bottom,label=label); bottom+=values
    axes[0].set_xticks(range(3),list(LABELS.values())); axes[0].set_ylabel('全部初始争议案件中的比例')
    axes[0].legend(fontsize=8); axes[0].set_title('首次选择与未投诉（跨种子均值）')
    keys=['d_to_e_all','e_to_d_all']; s=ep[ep.scenario=='both'][keys]
    axes[1].bar(range(2),s.mean(),yerr=s.std(),capsize=4)
    axes[1].set_xticks(range(2),['国内 → 客户','客户 → 国内'])
    axes[1].set_ylabel('全部初始争议案件中的比例'); axes[1].set_title('共存组的后续追加｜均值 ±1 标准差')
    fig.tight_layout(); fig.savefig(out/'competition_choices.png'); plt.close(fig)
    avg=ep.groupby('scenario').mean(numeric_only=True)
    lines=['# 固定条件下的渠道竞争：实际结果','',f'共 {len(ep)} 次运行，三情景使用相同连续种子。无国内改善、公告、材料帮助或预防机制；关闭新争议，固定初始争议队列。',
        '所有参数为探索值。渠道效果、成本和时间不相同但固定，三组只切换可用渠道；因此结果不能视为某类制度天然更优。','',
        '|情景|国内使用|客户使用|同时或先后使用两者|最终解决|未解决|','|---|---:|---:|---:|---:|---:|']
    for name in LABELS:
        s=avg.loc[name]
        lines.append(f'|{LABELS[name]}|{s.domestic_use:.2%}|{s.customer_use:.2%}|{s.both_use:.2%}|{s.resolved_share:.2%}|{s.unresolved_fraction:.2%}|')
    s=avg.loc['both']
    lines += ['',f'共存组：国内→客户占全部案件 {s.d_to_e_all:.2%}，客户→国内占全部案件 {s.e_to_d_all:.2%}。',
        f'条件追加比例：国内→客户 / 首次仅国内 = {s.d_to_e_conditional:.2%}；客户→国内 / 首次仅客户 = {s.e_to_d_conditional:.2%}。',
        '条件比例含尚未追加和已提前解决者，不是仍在办者追加风险；分母为零的运行记 NA，summary.csv 的均值忽略 NA，逐种子分母可从 cases.csv 核对。','',
        '所有使用和解决比例分母为该次运行全部初始争议案件，跨种子等权平均。两渠道使用可重叠，不应相加当作市场份额。首次分类与从未提交则互斥且和为 1。',
        'paired_differences.csv 给出共存减单渠道的配对差、样本 SD 与 SE；SE 不是置信区间。使用量变化同时包含选择变化、提前解决和后续追加，不能全部解释为直接争夺。',
        '本模型为固定规则下的渠道选择竞争，没有机构主动调整政策的策略博弈。双渠道共同成功不等于已识别制度互补。',
        '固定人数、无新增争议、每案每渠道仅可提交一次，会使曲线趋于平台；这不能证明现实长期均衡或扩散被制度抑制。',
        '材料帮助和公告被关闭，同事行动及结果观察保留。客户可达性及依赖异质性沿用固定默认分布，不做参数扫描。',
        '图 competition_dynamics.png 展示使用、追加和未解决随时间变化；competition_choices.png 展示首次路径与双向追加。阴影或误差棒为跨种子 ±1 样本标准差，不是置信区间。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/competition.json')
    p.add_argument('--repeats',type=int,default=30); p.add_argument('--out',default='outputs/competition_v1')
    p.add_argument('--small',action='store_true'); args=p.parse_args()
    if args.repeats<2: p.error('repeats must be >=2')
    c=ChannelConfig(**json.loads(Path(args.config).read_text(encoding='utf-8')))
    if args.small: c=replace(c,n=24,firms=3,degree=4,steps=15)
    run(c,args.repeats,args.out)
