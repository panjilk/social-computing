"""Post-run interpretation and consistency audit, without outcome-driven retuning."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def run(out):
    m=pd.read_csv(out/'metrics.csv'); st=pd.read_csv(out/'intervals.csv'); ct=pd.read_csv(out/'paired_contrasts.csv')
    def val(axis,level,mode,metric):
        return st[(st.axis==axis)&(st.level==level)&(st['mode']==mode)&(st.metric==metric)&
            (st.period=='late')&(st.population=='all')].iloc[0]
    def change(axis,mode,metric):
        return ct[(ct.axis==axis)&(ct['mode']==mode)&(ct.metric==metric)&
            (ct.contrast=='highest_minus_lowest')&(ct.population=='all')].iloc[0]
    def describe(z):return f'{z["mean"]*100:.2f}个百分点，95%区间[{z.low*100:.2f}, {z.high*100:.2f}]'
    lines=['# 第一轮研究结论：图片左图','',
        '本轮只检验使用竞争与材料互补，未完成中图和右图研究。参数是探索参数，不是现实预测。',
        '## 国内支持救济概率变化','支持参数从0.2提高到0.9，国内延迟、落实概率及初始主观预期保持不变：','',
        '|材料模式|客户使用：低→高|客户路径成功：低→高|客户使用变化95%区间（百分点）|成功比例变化95%区间|',
        '|---|---|---|---|---|']
    for mode,label in [('disabled','材料无作用'),('objective','仅客观作用'),('full','客观+主观作用')]:
        texts=[]
        for metric in ['customer_use','customer_route_success']:
            a=val('support',.2,mode,metric); b=val('support',.9,mode,metric)
            texts.append(f'{a["mean"]:.2%}→{b["mean"]:.2%}')
        lines.append(f'|{label}|'+ '|'.join(texts+[describe(change('support',mode,k)) for k in
            ['customer_use','customer_route_success']])+'|')
    z=change('support','full','customer_route_success')
    conclusion='支持上升' if z.low>0 else '支持下降' if z.high<0 else '没有明确上升或下降证据'
    lines+=['',f'完整材料模式的客户条件成功比例：{conclusion}。不能把图片“使用更少但更容易成功”当成已经复现。',
        '## 材料是否真正赶上客户审核','|支持参数|审核时有材料／客户使用案件|材料在固定使用者上的机械概率增量|','|---|---:|---:|']
    for level in [.2,.55,.9]:
        a=val('support',level,'full','material_at_review'); b=val('support',level,'full','mechanical_material_increment')
        lines.append(f'|{level}|{a["mean"]:.2%}|{b["mean"]:.2%}|')
    lines+=['','材料形成后必须在客户审核前可见才能影响审核；若国内更早解决，客户路径被取消。',
        '当前支持性材料规则意味着材料只来自获得国内救济支持的案件，而这些案件又有机会迅速由国内解决。',
        '这构成流程上的竞争，不应为得到互补曲线而取消国内解决或延长流程。后续可以预先设计落实延迟对照来单独检验时序。',
        '## 响应速度变化','国内初次响应延迟从0到6轮，完整材料模式：',
        '- 客户使用变化：'+describe(change('delay','full','customer_use')),
        '- 客户路径成功比例变化：'+describe(change('delay','full','customer_route_success')),
        '延迟越大表示响应越慢。该扫描不同时改变支持与落实概率，不能与支持概率扫描混作一种改善。',
        '## 结论边界','无材料标准得分描述使用者构成；机械材料增量描述冻结案件上的概率变化；材料模式配对差描述总机制作用。三者不是可相加的因果贡献。',
        '客户成功率只在使用客户的案件中定义，可能受选择、国内提前解决、取消和随访窗口影响。另报全部解决与未解决比例。',
        '每人一个案件，每批统一24轮随访；未检验重复争议、企业预防或长期扩散。',
        '没有文献或数据依据证明材料增益0.3或主观加成0.18，不能把这些设定称为现实规律。']
    (out/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    a=m[(m.axis=='support')&(m.level==.55)].drop(columns=['axis','level','scenario']).sort_values(['seed','mode','period','population']).reset_index(drop=True)
    b=m[(m.axis=='delay')&(m.level==2)].drop(columns=['axis','level','scenario']).sort_values(['seed','mode','period','population']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    np.testing.assert_allclose(m.customer_route_success,m.customer_only_resolution+m.joint_resolution,equal_nan=True)
    diagnostics=[]
    for folder in (out/'demo').iterdir():
        ca=pd.read_csv(folder/'cases.csv'); mat=ca.material_at.notna()
        diagnostics.append(dict(scenario=folder.name,material_cases=int(mat.sum()),
            material_before_customer_review=int((mat&ca.material_at.lt(ca.e_responded)).sum()),
            material_cases_domestic_only_resolved=int((mat&ca.resolved_by.eq('d')).sum())))
    pd.DataFrame(diagnostics).to_csv(out/'material_timing_seed0.csv',index=False)
    (out/'additional_checks.json').write_text(json.dumps(dict(center_scan_identical=True,
        exclusive_and_joint_customer_success_identity=True,runs=int(m.scenario.nunique()*m.seed.nunique()),
        unique_configurations_per_seed=15),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/substitution_complementarity_v1')
    run(Path(p.parse_args().out))
