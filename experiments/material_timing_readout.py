"""Readout of material timing experiment; no parameter selection from results."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(out):
    st=pd.read_csv(out/'intervals.csv'); ct=pd.read_csv(out/'paired_contrasts.csv'); m=pd.read_csv(out/'metrics.csv')
    def value(level,mode,metric):
        return st[(st.level==level)&(st['mode']==mode)&(st.metric==metric)&(st.period=='late')&(st.population=='all')].iloc[0]
    def effect(level,metric,contrast='objective_minus_disabled'):
        return ct[(ct.level==level)&(ct.metric==metric)&(ct.contrast==contrast)&(ct.population=='all')].iloc[0]
    def ci(z):return f'{z["mean"]*100:.2f} [{z.low*100:.2f}, {z.high*100:.2f}]'
    levels=sorted(st.level.unique())
    lines=['# 材料互补是否取决于国内落实时序？','',
        f'实际完成{m.scenario.nunique()}情景×{m.seed.nunique()}连续种子。国内支持参数固定0.55，初次响应延迟固定2轮，落实概率固定0.85。',
        '只改变支持后落实延迟，并独立对照材料无作用、仅客观作用、客观+主观作用。全部探索参数，非现实校准。',
        '后期=后两批案件；每批观察24轮。先按种子内案件统计，再跨种子等权。',
        '## 完整材料模式的水平','|落实延迟|客户使用|国内使用|双渠道使用|国内→客户追加|客户路径成功／客户使用|全部案件解决|审核时有材料／客户使用|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for level in levels:
        lines.append(f'|{level:g}|'+ '|'.join(f'{value(level,"full",k)["mean"]:.2%}' for k in
            ['customer_use','domestic_use','both_use','d_to_e','customer_route_success','resolved','material_at_review'])+'|')
    lines+=['','## 相同延迟下，开启客观材料作用的配对变化',
        '单位：百分点，括号为逐项bootstrap95%区间；未做多重比较校正。',
        '|落实延迟|客户使用变化|全部案件解决变化|客户条件成功比例变化|','|---|---|---|---|']
    for level in levels:
        lines.append(f'|{level:g}|'+ '|'.join(ci(effect(level,k)) for k in ['customer_use','resolved','customer_route_success'])+'|')
    lines+=['','## 增加主观材料预期的配对变化（完整减仅客观）',
        '|落实延迟|客户使用变化|全部案件解决变化|','|---|---|---|']
    for level in levels:
        lines.append(f'|{level:g}|'+ '|'.join(ci(effect(level,k,'full_minus_objective')) for k in ['customer_use','resolved'])+'|')
    # Paired difference-in-differences estimates interaction, not a direct mediation effect.
    rows=[]
    g=m[(m.period=='late')&(m.population=='all')]
    for metric in ['resolved','customer_use','customer_route_success','material_at_review']:
        diffs={}
        for level in [min(levels),max(levels)]:
            subset=g[g.level==level]
            diffs[level]=subset[subset['mode']=='objective'].set_index('seed')[metric]-subset[subset['mode']=='disabled'].set_index('seed')[metric]
        x=(diffs[max(levels)]-diffs[min(levels)]).dropna().to_numpy()
        rng=np.random.default_rng(190902); boot=rng.choice(x,(2000,len(x)),replace=True).mean(axis=1)
        rows.append(dict(metric=metric,mean=x.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975)))
    interaction=pd.DataFrame(rows); interaction.to_csv(out/'timing_interactions.csv',index=False)
    lines+=['','## 材料作用是否随延迟改变',
        '计算：(最大延迟的客观材料开关差)−(最小延迟的客观材料开关差)。案件来源和种子配对，行为与事件过程可不同。']
    for _,row in interaction.iterrows():lines.append(f'- {row.metric}：{ci(row)}个百分点。')
    lines+=['','## 解释边界',
        '客户使用增加不等于总解决改善；客户单一路径成功与联合成功不能重复加到案件总数。',
        '即使材料在更长落实延迟下更有作用，也不能推导出拖延国内落实更好；须比较全部解决、未解决和等待代价。',
        '本次在固定24轮窗口统计，延迟会改变办结与取消机会；材料开关配对效应包含社会学习、选择和排队反馈，不是固定人群的纯直接效应。',
        '主观材料加成是未经校准的假设，不应因其增加使用就称之为真实员工学习。',
        '本轮检验左图时序机制，未实现中图企业合规差距或右图持续争议反馈。']
    (out/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    for ax,(metric,title) in zip(axs,[('customer_use','客户使用变化'),('resolved','全部案件解决变化')]):
        for contrast,label in [('objective_minus_disabled','客观材料作用'),('full_minus_objective','额外主观预期作用')]:
            r=pd.DataFrame([effect(level,metric,contrast) for level in levels])
            ax.errorbar(levels,r['mean']*100,yerr=[(r['mean']-r.low)*100,(r.high-r['mean'])*100],marker='o',capsize=3,label=label)
        ax.axhline(0,color='gray',ls='--'); ax.set(xlabel='国内支持后落实延迟（轮）',ylabel=title+'（百分点）'); ax.grid(alpha=.2)
    h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,ncol=2,loc='upper center')
    fig.suptitle('相同延迟下材料开关的配对差｜后期争议案件｜逐项95%区间',y=.91)
    fig.tight_layout(rect=[0,0,1,.86]); fig.savefig(out/'material_effects.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    old=Path('outputs/substitution_complementarity_v1/metrics.csv')
    checks={}
    if old.exists():
        previous=pd.read_csv(old); previous=previous[(previous.axis=='support')&(previous.level==.55)&previous.seed.isin(m.seed.unique())]
        current=m[m.level==1]; cols=[k for k in current.columns if k not in ['axis','level','scenario']]
        keys=['seed','mode','period','population']
        pd.testing.assert_frame_equal(current[cols].sort_values(keys).reset_index(drop=True),previous[cols].sort_values(keys).reset_index(drop=True))
        checks['delay1_matches_previous_center']=True
    checks['runs']=int(m.scenario.nunique()*m.seed.nunique())
    (out/'additional_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/material_timing_v1')
    run(Path(p.parse_args().out))
