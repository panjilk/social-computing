"""Post-run integrity checks and beginner-facing findings; no simulation tuning."""
import json
import hashlib
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(out):
    st=pd.read_csv(out/'intervals.csv'); co=pd.read_csv(out/'cohorts.csv')
    def value(name,metric='customer_minus_domestic',contrast='late',pop='all'):
        return st[(st.scenario==name)&(st.metric==metric)&(st.contrast==contrast)&(st.population==pop)].iloc[0]
    def ci(z): return f'{z["mean"]*100:.2f}个百分点，95%区间[{z.low*100:.2f}, {z.high*100:.2f}]'
    def line(name,title):
        d=value(name,'domestic_use'); e=value(name,'customer_use'); z=value(name); t=value(name,'customer_use','change')
        verdict='客户使用较多' if z.low>0 else '国内使用较多' if z.high<0 else '使用优势尚不明确'
        trend='上升' if t.low>0 else '下降' if t.high<0 else '没有明确上升或下降证据'
        return f'- {title}：国内{d["mean"]:.2%}，客户{e["mean"]:.2%}；{verdict}。客户早晚变化{ci(t)}，{trend}。'
    lines=['# 优化版结果怎么理解','',
        '以下均为探索参数下的条件结果，不是现实未来预测。前期=前两批，后期=后两批；使用比例分母是相应争议员工。',
        '## 基准与实现偏向',line('base','随机平局基准'),line('symmetric_random','匹配客观与主观条件的对称诊断'),
        f'\n旧规则减随机平局基准的后期客户使用变化：{ci(value("legacy","customer_use","late_minus_base"))}。',
        '对称条件下20个独立抽样世界仍有差距，不能把渠道标签对等误解成有限样本均值必然相等。详见额外的symmetry_mirror.json：交换个体两侧成本、先验及事件随机流，核查提交/结束/解决时刻是否逐案互换。精确平局另有针对性测试。',
        '## 等待学习',line('wait_learning','加入可见等待学习'),
        f'\n相对固定等待基准，后期客户使用配对变化：{ci(value("wait_learning","customer_use","late_minus_base"))}。',
        '如变化不明显，不代表排队永远不重要，也不应继续调参制造效果；目前只是这组负荷、折扣与学习强度下的结果。',
        '## 观察窗口、负荷和记忆',
        *[line(name,title) for name,title in [('followup_12','观察12轮'),('followup_48','观察48轮'),
            ('spacing_3','每3轮一批争议'),('spacing_12','每12轮一批争议'),('memory_6','结果记忆6轮'),('memory_36','结果记忆36轮')]],
        '## 可达性与成本：完整九格对照',
        *[line(name,name.removeprefix('grid_').replace('r','可达概率').replace('_c','／成本比')) for name in co.scenario.unique() if name.startswith('grid_')],
        '即使全部争议员工中国内使用较多，可达子群仍可能有不同格局；两者是不同分母，不可混用。',
        '## 偏好与效果分开',
        *[line(name,name) for name in co.scenario.unique() if name.startswith(('objective_','prior_'))],
        'objective表示客户后续落实概率，不是全流程成功率；prior表示客户初始主观预期。两类实验独立，不把相信有效等同实际有效。',
        '## 首次选择与并用', '|基准后期指标|均值|','|---|---:|']
    for m,label in [('first_d','首次仅国内'),('first_e','首次仅客户'),('first_both','首次同时'),
                    ('domestic_only','观察期内仅国内'),('customer_only','观察期内仅客户'),
                    ('both_use','观察期内双渠道'),('no_submission','未投诉'),('d_to_e','国内→客户'),('e_to_d','客户→国内')]:
        lines.append(f'|{label}|{value("base",m)["mean"]:.2%}|')
    lines+=['','首次三类加未投诉=100%；仅国内、仅客户、双渠道、未投诉也合计100%，两套分类不能混加。',
        '国内→客户与客户→国内均以全部争议员工为分母，不是首次选择对应渠道者中的条件追加率。',
        '## 限制','20个种子和有限参数网格不能证明全参数空间规律。区间逐项，未做多重比较校正。',
        '等待学习是含删失下界的主观启发式，不是现实校准或无偏生存估计。',
        '国内与客户多阶段机制及可达结构仍有设定依赖。没有真实行为数据，不能从这些数值推断现实未来份额。']
    (out/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    # Historical regression: all old metrics for all matching seeds/cohorts.
    oldpath=Path('outputs/cohort_trends_v1/cohorts.csv')
    checks={}
    if oldpath.exists():
        old=pd.read_csv(oldpath); old=old[(old.scenario=='base')&old.seed.isin(co.seed.unique())]
        new=co[co.scenario=='legacy']; keys=['seed','population','cohort']
        columns=[k for k in old.columns if k in new.columns and k!='scenario']
        pd.testing.assert_frame_equal(old.sort_values(keys)[columns].reset_index(drop=True),
            new.sort_values(keys)[columns].reset_index(drop=True),check_dtype=False)
        checks['historical_base_metrics_identical']=True
    for cols in [['domestic_only','customer_only','both_use','no_submission'],['first_d','first_e','first_both','no_submission']]:
        np.testing.assert_allclose(co.loc[co.cases>0,cols].sum(axis=1),1.)
    checks['exclusive_paths_and_first_choices_sum_to_one']=True
    a=co[co.scenario=='base'].set_index(['seed','population','cohort'])
    b=co[co.scenario=='legacy'].set_index(['seed','population','cohort'])
    checks['random_ties_changed_any_baseline_usage']=bool((a[['domestic_use','customer_use']]!=b[['domestic_use','customer_use']]).any().any())
    checks['runs']=int(co.scenario.nunique()*co.seed.nunique())
    fixed=pd.read_csv(out/'demo/base/decisions.csv')
    learned=pd.read_csv(out/'demo/wait_learning/decisions.csv')
    common=fixed.merge(learned,on=['time','case_id'],suffixes=('_fixed','_learned'))
    checks['seed0_shared_decisions']=len(common)
    checks['seed0_changed_choices_on_shared_decisions']=int((common.selected_fixed!=common.selected_learned).sum())
    checks['seed0_learned_wait_range']={k:[float(learned[k].min()),float(learned[k].max())]
        for k in ['expected_wait_d','expected_wait_e']}
    (out/'additional_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (out/'readout_verification.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks':checks},indent=2),encoding='utf-8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    for ax,prefix,title in zip(axs,['objective_','prior_'],['客户后续落实概率（客观）','客户初始成功预期（主观）']):
        names=[s for s in co.scenario.unique() if s.startswith(prefix)]; x=[float(s[len(prefix):]) for s in names]
        for m,label in [('domestic_use','国内正式渠道'),('customer_use','外部商业客户渠道')]:
            rows=pd.DataFrame([value(s,m) for s in names])
            ax.errorbar(x,rows['mean']*100,yerr=[(rows['mean']-rows.low)*100,(rows.high-rows['mean'])*100],marker='o',capsize=4,label=label)
        ax.set(xlabel=title,ylabel='后期使用比例（%）',ylim=(0,100)); ax.grid(alpha=.2)
    h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=2)
    fig.suptitle('主观判断与客观效果分别改变｜逐项95%区间｜分母：后期争议员工',y=.91)
    fig.tight_layout(rect=[0,0,1,.88]); fig.savefig(out/'belief_vs_effect.png',dpi=160); plt.close(fig)
    checks_names=['legacy','base','wait_learning','wait_no_outcomes','followup_12','followup_48',
        'spacing_3','spacing_12','memory_6','memory_36','symmetric_random','symmetric_legacy']
    labels=['旧平局规则','随机平局基准','学习可见等待','等待学习／无结果学习','观察12轮','观察48轮',
        '间隔3轮','间隔12轮','记忆6轮','记忆36轮','对称／随机平局','对称／旧平局']
    p=pd.read_csv(out/'periods.csv'); paths=p[(p.population=='all')&(p.period=='late')].groupby('scenario')[
        ['domestic_only','customer_only','both_use','no_submission']].mean().loc[checks_names]
    ax=paths.plot.bar(stacked=True,figsize=(12,6),color=['#4477aa','#ee7733','#228833','#bbbbbb'])
    ax.set_xticklabels(labels,rotation=30,ha='right'); ax.set(xlabel='实验情景',ylabel='后期争议员工比例（均值）',ylim=(0,1))
    ax.legend(['仅国内','仅客户','双渠道并用','未投诉'],ncol=4,loc='upper center',bbox_to_anchor=(.5,1.15))
    ax.figure.tight_layout(); ax.figure.savefig(out/'exclusive_paths.png',dpi=160); plt.close(ax.figure)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/competition_refinement_v1')
    run(Path(p.parse_args().out))
