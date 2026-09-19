"""Image-left hypotheses: substitution and material complementarity, not assumed results."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.channel_choice import ChannelConfig,simulate_channels,review_probability,firm_customer_probability
from src.cohort_trends import CohortSchedule
from src.customer_network import CustomerNetwork
from experiments.competition import design
from experiments.channel_experiments import audit

MODES={'disabled':'材料无作用','objective':'仅客观材料作用','full':'客观与主观材料作用'}
AXES={'support':('support_probabilities','domestic_success','国内支持救济概率参数（非最终解决率）'),
      'delay':('domestic_delays','domestic_delay','国内初次响应延迟（轮）'),
      'implementation':('implementation_delays','domestic_implementation_delay','国内支持后落实延迟（轮）')}


def configurations(spec):
    old=json.loads(Path(spec['base_config']).read_text(encoding='utf-8'))
    data=json.loads(Path(old['base_config']).read_text(encoding='utf-8')); data.update(old['overrides'])
    base=replace(design(ChannelConfig(**data))['both'],tie_break='random',
        material_requires_support=True,material_rate=spec['material_rate'])
    variants={}
    for axis in spec.get('axes',['support','delay']):
        key,field,_=AXES[axis]; levels=spec[key]
        for level in levels:
            for mode in MODES:
                c=replace(base,**{field:level},evidence_help=0. if mode=='disabled' else spec['objective_material_bonus'],
                    perceived_evidence_bonus=spec['subjective_material_bonus'] if mode=='full' else 0.)
                variants[f'{axis}_{level}_{mode}']=(c,axis,level,mode)
    return variants,CohortSchedule(**old['schedule']),old['customer_network']


def measures(cases,schedule,c):
    rows=[]
    for period,cohorts in [('early',range(2)),('late',range(schedule.cohorts-2,schedule.cohorts))]:
        selected=cases[cases.cohort.isin(cohorts)]
        for population,g in [('all',selected),('both_reachable',selected[selected.reachable.astype(bool)])]:
            end=g.cohort*schedule.spacing+schedule.followup
            d=g.d_submitted.lt(end); e=g.e_submitted.lt(end); n=len(g); ne=int(e.sum())
            solved=g.resolved.lt(end); resolved_by=g.resolved_by.fillna('')
            e_success=solved&resolved_by.isin(['e','d+e'])
            e_only_success=solved&resolved_by.eq('e'); joint=solved&resolved_by.eq('d+e')
            reviewed=e&g.e_responded.lt(end)
            material=reviewed&g.material_at.lt(g.e_responded)
            def mean(mask):return float(mask.sum()/n) if n else np.nan
            def conditional(mask):return float((mask&e).sum()/ne) if ne else np.nan
            # Fixed standard score among selected E users; composition diagnostic only.
            q0=review_probability(replace(c,evidence_help=0.),g.merit.to_numpy(),False)*firm_customer_probability(c,g.dependence.to_numpy())
            qr=review_probability(c,g.merit.to_numpy(),material.to_numpy())*firm_customer_probability(c,g.dependence.to_numpy())
            rows.append(dict(period=period,population=population,cases=n,customer_users=ne,
                domestic_use=mean(d),customer_use=mean(e),first_d=mean(d&(~e|g.d_submitted.lt(g.e_submitted))),
                first_e=mean(e&(~d|g.e_submitted.lt(g.d_submitted))),first_both=mean(d&e&g.d_submitted.eq(g.e_submitted)),
                neither=mean(~(d|e)),d_to_e=mean(d&e&g.d_submitted.lt(g.e_submitted)),
                e_to_d=mean(d&e&g.e_submitted.lt(g.d_submitted)),both_use=mean(d&e),
                resolved=mean(solved),unresolved=mean(~solved),
                customer_route_success=conditional(e_success),customer_only_resolution=conditional(e_only_success),
                joint_resolution=conditional(joint),customer_user_any_resolution=conditional(solved),
                customer_unresolved=conditional(~solved),material_at_review=conditional(material),
                standardized_no_material_score=float(q0[e].mean()) if ne else np.nan,
                mechanical_material_increment=float((qr[e]-q0[e]).mean()) if ne else np.nan,
                mean_customer_merit=float(g.loc[e,'merit'].mean()) if ne else np.nan,
                mean_customer_dependence=float(g.loc[e,'dependence'].mean()) if ne else np.nan))
    return pd.DataFrame(rows)


def report(out,spec):
    data=pd.read_csv(out/'metrics.csv'); cols=[x for x in data.columns if x not in
        ['period','population','cases','customer_users','scenario','axis','level','mode','seed']]
    records=[]
    # Same resampled seed indices for every statistic of a given size.
    def summarize(x):
        x=np.asarray(x); x=x[np.isfinite(x)]
        if not len(x):return dict(mean=np.nan,low=np.nan,high=np.nan,n=0)
        rng=np.random.default_rng(190901)
        boot=rng.choice(x,(spec['bootstrap_samples'],len(x)),replace=True).mean(axis=1)
        return dict(mean=x.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975),n=len(x))
    keys=['axis','level','mode','period','population']
    for group,g in data.groupby(keys):
        for m in cols:records.append(dict(zip(keys,group),metric=m,**summarize(g.sort_values('seed')[m])))
    stats=pd.DataFrame(records); stats.to_csv(out/'intervals.csv',index=False)
    contrasts=[]
    for (axis,mode,pop),g in data[data.period=='late'].groupby(['axis','mode','population']):
        low=g[g.level==g.level.min()].set_index('seed'); high=g[g.level==g.level.max()].set_index('seed')
        for m in cols:contrasts.append(dict(axis=axis,mode=mode,population=pop,level=np.nan,
            contrast='highest_minus_lowest',metric=m,**summarize(high[m]-low[m])))
    for (axis,level,pop),g in data[data.period=='late'].groupby(['axis','level','population']):
        for a,b in [('objective','disabled'),('full','objective'),('full','disabled')]:
            ga=g[g['mode']==a].set_index('seed'); gb=g[g['mode']==b].set_index('seed')
            for m in cols:contrasts.append(dict(axis=axis,mode=a,population=pop,level=level,
                contrast=f'{a}_minus_{b}',metric=m,**summarize(ga[m]-gb[m])))
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
    for axis in data.axis.unique():
        xlabel=AXES[axis][2]
        fig,axs=plt.subplots(2,2,figsize=(12,8))
        for ax,(m,label) in zip(axs.flat,[('customer_use','客户渠道使用比例'),('customer_route_success','客户路径成功／客户使用案件'),
            ('standardized_no_material_score','客户使用者无材料标准得分'),('material_at_review','审核时已有材料／客户使用案件')]):
            for mode,title in MODES.items():
                g=stats[(stats.axis==axis)&(stats['mode']==mode)&(stats.period=='late')&(stats.population=='all')&(stats.metric==m)].sort_values('level')
                ax.plot(g.level,g['mean'],marker='o',label=title); ax.fill_between(g.level,g.low,g.high,alpha=.15)
            ax.set(xlabel=xlabel,ylabel=label); ax.grid(alpha=.2)
        h,l=axs.flat[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3)
        fig.suptitle('使用竞争与材料互补检验｜后两批等长观察｜逐项95%区间｜未校准',y=.94)
        fig.tight_layout(rect=[0,0,1,.89]); fig.savefig(out/f'{axis}_comparison.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    lines=['# 左图研究：实际实验结果','',f'{data.seed.nunique()}个连续配对种子，{data.scenario.nunique()}组情景；参数未经现实校准。',
        '后期=最后两批合并，先种子内按案件计数，后跨种子等权。客户使用分母为全部争议案件；客户成功分母为客户使用案件。',
        '客户路径成功含同时由两条路径成功的案件，各案件在该指标只计一次。另报仅客户解决与联合解决；不能将联合案件分别相加作总解决数。',
        '所有指标只使用各批次相同24轮随访内的行动和结果。区间为逐项95% bootstrap，未作多重比较修正。','',
        '|扫描|参数|材料模式|客户使用|客户路径成功比例|全部案件解决|客户使用者未解决|无材料标准得分|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for (axis,level,mode),g in stats[(stats.period=='late')&(stats.population=='all')].groupby(['axis','level','mode']):
        vals=g.set_index('metric')['mean']
        lines.append(f'|{axis}|{level}|{MODES[mode]}|'+ '|'.join(f'{vals[m]:.2%}' for m in
            ['customer_use','customer_route_success','resolved','customer_unresolved','standardized_no_material_score'])+'|')
    lines+=['','## 最高参数减最低参数的配对变化','|扫描|材料模式|客户使用变化及95%区间（百分点）|客户成功比例变化及区间|','|---|---|---|---|']
    ct=pd.DataFrame(contrasts)
    for axis in data.axis.unique():
        for mode in MODES:
            g=ct[(ct.axis==axis)&(ct['mode']==mode)&(ct.population=='all')&(ct.contrast=='highest_minus_lowest')].set_index('metric')
            texts=[]
            for m in ['customer_use','customer_route_success']:
                z=g.loc[m]; texts.append(f'{z["mean"]*100:.2f} [{z.low*100:.2f}, {z.high*100:.2f}]')
            lines.append(f'|{axis}|{MODES[mode]}|'+ '|'.join(texts)+'|')
    lines+=['','## 识别边界',
        '无材料标准得分：在实际选择客户的案件上，计算固定无材料审核概率乘企业后续落实概率。变化描述案件构成，不是观察成功率的因果分解。',
        '机械材料增量：冻结实际使用者及审核时材料状态，计算有/无材料的路径概率差；忽略排队取消和国内竞争解决，不能叫材料贡献占比。',
        '材料模式配对差是该模型中材料机制的总影响，会反馈到选择、排队与信息；不是保持实际使用者不变的直接效应。',
        '本轮材料仅在国内支持救济后按概率形成；不等于所有行政文书，也未声称来自某真实机关。材料不等于已解决。',
        '各扫描不同时改变速度、解决效果、初始信念。客观支持改变后，员工仅通过原有结果学习得知，未假设即时全知。',
        '本轮未实现企业预防、合规差距或右图持续争议反馈，不据此宣称传播不会失控。']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def run(config,out,repeats):
    spec=json.loads(Path(config).read_text(encoding='utf-8')); variants,s,net=configurations(spec)
    out.mkdir(parents=True,exist_ok=True); rows=[]
    for seed in range(repeats):
        ref=None
        for name,(c,axis,level,mode) in variants.items():
            r=simulate_channels(c,seed,customer_network=CustomerNetwork(**net),cohort_schedule=replace(s)); audit(r)
            g=r['cases']; key=g[['case_id','cohort','merit','dependence','reachable']]
            if ref is None: ref=key
            else: pd.testing.assert_frame_equal(ref,key)
            assert g.loc[g.material_at.notna(),'d_institution_outcome'].eq('remedy_supported').all()
            m=measures(g,s,c); np.testing.assert_allclose(m[['first_d','first_e','first_both','neither']].sum(axis=1),1.)
            np.testing.assert_allclose(m.resolved+m.unresolved,1.)
            rows.append(m.assign(seed=seed,scenario=name,axis=axis,level=level,mode=mode))
            if seed==0:
                folder=out/'demo'/name; folder.mkdir(parents=True,exist_ok=True)
                for item in ['cases','events','decisions']:r[item].to_csv(folder/f'{item}.csv',index=False)
        print(f'Completed paired seed {seed+1}/{repeats}',flush=True)
    pd.concat(rows,ignore_index=True).to_csv(out/'metrics.csv',index=False); report(out,spec)
    paths=[config,__file__,'src/channel_choice.py','src/cohort_trends.py','src/customer_network.py']
    (out/'verification.json').write_text(json.dumps(dict(runs=len(variants)*repeats,seeds=list(range(repeats)),
        spec=spec,schedule=asdict(s),network=net,scenarios={k:asdict(v[0]) for k,v in variants.items()},
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/substitution_complementarity.json')
    p.add_argument('--out',default='outputs/substitution_complementarity_v1'); p.add_argument('--repeats',type=int)
    a=p.parse_args(); n=a.repeats or json.loads(Path(a.config).read_text(encoding='utf-8'))['repeats']
    if n<2:p.error('At least two seeds')
    run(a.config,Path(a.out),n)
