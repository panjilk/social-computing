"""Paired endpoints and plain-language conclusions for the dependence scan."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.dependence_gap import interval


def run(out):
    meta=json.loads((out/'verification.json').read_text(encoding='utf-8')); samples=meta['spec']['bootstrap_samples']
    m=pd.read_csv(out/'metrics.csv'); shape=pd.read_csv(out/'shape_contrasts.csv')
    levels=pd.read_csv(out/'levels.csv'); gaps=pd.read_csv(out/'gap_intervals.csv'); bounds=pd.read_csv(out/'probability_bounds.csv')
    allcases=m[(m.group=='all')&(m.period=='late')]; rows=[]
    for name,g in allcases.groupby('variant'):
        a=g[g.dependence_mean==.1].set_index('seed'); b=g[g.dependence_mean==.9].set_index('seed')
        for metric in ['customer_use','domestic_use','resolved','customer_success','both_use','first_customer','d_to_e']:
            rows.append(dict(variant=name,metric=metric,**interval(b[metric]-a[metric],samples)))
    endpoints=pd.DataFrame(rows); endpoints.to_csv(out/'endpoint_contrasts.csv',index=False)
    def ci(z):return f'{z["mean"]*100:.2f}个百分点，95%区间[{z.low*100:.2f}, {z.high*100:.2f}]'
    lines=['# 中图研究：目前能得出什么结论','',
        '本轮讨论企业客户依赖与员工案件结果，不是企业合规行为，也不是未来年份预测。',
        '## 1. 平均依赖变化与客户使用','主实验两侧入口均可达，固定企业离散幅度和企业身份。依赖均值0.1→0.9：']
    for metric,label in [('customer_use','客户使用'),('customer_success','客户使用案件中的客户路径成功'),('resolved','全部争议案件解决')]:
        row=endpoints[(endpoints.variant=='main')&(endpoints.metric==metric)].iloc[0]
        lines.append(f'- {label}的配对变化：{ci(row)}。')
    lines+=['','依赖进入客观与主观公式，因此以上单调变化本身不构成新发现；不能把模型依赖0.9解释为境外收入90%。',
        '## 2. 企业差距是不是中间最大？','预先指定0.5为中间点，比较高依赖组减低依赖组的案件解决比例差：']
    sg=shape[(shape.variant=='main')&(shape.grouping=='halves')&(shape.metric=='resolved')]
    for _,row in sg.iterrows():lines.append(f'- {row.contrast}：{ci(row)}。')
    pair=sg[sg.contrast.isin(['middle_minus_low','middle_minus_high'])]
    verdict='中间大于两个端点有逐项区间支持，但仍不能证明完整倒U或精确峰值。' if (pair.low>0).all() else '没有获得中间差距同时大于两端的明确证据，不能声称复现了图片的倒U形。'
    lines+=['',verdict,'','## 3. 分组与权重稳健性','|比较方式|中间减低端|中间减高端|','|---|---|---|']
    for grouping,metric,label in [('halves','resolved','高低各3家，案件加权'),('tails','resolved','高低各2家，案件加权'),
                                  ('halves','firm_equal_resolved','高低各3家，企业等权')]:
        cells=[]
        for contrast in ['middle_minus_low','middle_minus_high']:
            z=shape[(shape.variant=='main')&(shape.grouping==grouping)&(shape.metric==metric)&(shape.contrast==contrast)].iloc[0]
            cells.append(ci(z))
        lines.append('|'+label+'|'+'|'.join(cells)+'|')
    lines+=['','## 4. 主客观作用的对照','|情景|客户使用：高均值减低均值|全部解决：高均值减低均值|','|---|---|---|']
    for name in ['main','no_perceived','no_objective','narrow','limited_access']:
        cells=[ci(endpoints[(endpoints.variant==name)&(endpoints.metric==k)].iloc[0]) for k in ['customer_use','resolved']]
        lines.append('|'+name+'|'+'|'.join(cells)+'|')
    lines+=['','no_perceived只关闭依赖的直接先验作用，仍学习实际结果。no_objective将后续落实概率固定0.525，仍保留主观先验差异。',
        '跨情景比较须使用配对差进一步检验，不能只凭两个区间一个含零、一个不含零就断言两效应不同。',
        '## 5. 当前证据边界',
        '全部客观后续落实概率严格在0和1之间，没有端点裁剪。保持固定离散幅度，避免把企业分布收缩误判为保护差距收缩。',
        '每个种子仅6家企业、每批案件数量有限，高低组差距有较大不确定性。',
        '固定分组排除了追着结果挑企业，但没有排除企业的其他随机特征影响，因此使用跨种子区间。',
        '这些是不同参数情景的比较，不是同一世界中未来必然增长。右图的持续争议和企业预防反馈仍未实现。']
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    assert bounds.objective_at_boundary.eq(0).all()
    assert (bounds.min_objective>0).all() and (bounds.max_objective<1).all()
    checks=dict(no_objective_probability_clipping=True,all_runs=meta['runs'])
    # Check full access main center against previous no-material center.
    # Prior full-access grid r1 cost1 has same baseline core, but no material changes; do not assume files exist.
    for name,g in m[m.group=='all'].groupby('variant'):
        assert (g.customer_use<=1).all() and (g.resolved<=1).all()
        assert (g.both_use<=g[['domestic_use','customer_use']].min(axis=1)+1e-12).all()
    checks['overlap_denominators_valid']=True
    (out/'additional_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/dependence_gap_v1')
    run(Path(p.parse_args().out))
