"""Optional continuing disputes, knowledge diffusion and lagged firm prevention.

Episode retirement is withdrawal/censoring, NEVER resolution. Unresolved ledger
retains retired episodes. No empirical calibration or stationary guarantee.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.channel_choice import ACTIVE


@dataclass
class ContinuousCompetition:
    arrival_rate: float = .02
    initial_knowledge: float = .15
    rho: float = .3
    feedback: bool = True
    shock: str = 'none'
    shock_time: int = 80
    review_bonus: float = .25
    announcement_fraction: float = .2
    active_horizon: int = 48
    outcome_window: int = 18
    pressure_window: int = 24
    prevention_strength: float = .8
    pressure_scale: float = 20.
    prevention_cost: float = .08
    adaptation: float = .2

    def start(self,c,w,seed):
        if c.initial_dispute or c.new_dispute_rate or c.prevention:
            raise ValueError('Continuous hook owns all arrivals and prevention')
        for key in ['arrival_rate','initial_knowledge','rho','review_bonus','announcement_fraction',
                    'prevention_strength','adaptation']:
            if not 0<=getattr(self,key)<=1:raise ValueError(key)
        if self.shock not in ['none','objective','information']:raise ValueError('shock')
        for key in ['shock_time','active_horizon','outcome_window','pressure_window']:
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(key)
        if self.pressure_scale<=0 or self.prevention_cost<0:raise ValueError('prevention parameters')
        def rng(k):return np.random.default_rng(np.random.SeedSequence([seed,2901,k]))
        self.potential=rng(0).random((c.steps,c.n))<self.arrival_rate
        self.prevention_draw=rng(1).random((c.steps,c.n))
        self.diffusion=rng(2).random((c.steps,c.n))
        self.known=rng(3).random(c.n)<self.initial_knowledge
        self.announcement=rng(4).random(c.n)<self.announcement_fraction
        self.effort=np.zeros(c.firms); self.base_review=c.customer_review_base
        self.records=[];self.firm_records=[];self.retirements=[]
        w['customer_known']=self.known

    def advance(self,t,c,w,cases,current,neighbors):
        # Retiring an episode releases employee availability, not its unresolved liability.
        for i,case in list(current.items()):
            if t-case['created']>=self.active_horizon:
                for k in ['d','e']:
                    if case[k+'_status'] in ACTIVE:
                        case[k+'_status']='observation_expired';case[k+'_finished']=t
                case['status']='archived_unresolved'
                self.retirements.append(dict(time=t,case_id=case['case_id'],employee=i))
                del current[i]
        past=self.known.copy()
        for i,nb in enumerate(neighbors):
            informed=sum(past[j] and w['sharing'][i,j] for j in nb)
            if self.diffusion[t,i]<1-(1-self.rho)**informed:self.known[i]=True
        if self.shock=='information' and t==self.shock_time:self.known|=self.announcement
        c.customer_review_base=self.base_review+(self.review_bonus if self.shock=='objective' and t>=self.shock_time else 0.)
        pressure=np.zeros(c.firms)
        for case in cases:
            # Only actual commercial intervention, visible before this round.
            at=case['customer_intervened_at']
            if at is not None and 1<=t-at<=self.pressure_window:pressure[case['firm']]+=1
        size=np.bincount(w['firm'],minlength=c.firms)
        score=self.pressure_scale*(pressure/size)*w['dependence']-self.prevention_cost
        target=np.clip(score,0,1) if self.feedback else np.zeros(c.firms)
        self.effort=(1-self.adaptation)*self.effort+self.adaptation*target
        suppression=self.prevention_strength*self.effort
        potential=np.flatnonzero(self.potential[t]); born=[];blocked=0;prevented=0
        for i in potential:
            if int(i) in current:blocked+=1
            elif self.prevention_draw[t,i]<suppression[w['firm'][i]]:prevented+=1
            else:born.append(int(i))
        self.records.append(dict(time=t,potential=len(potential),blocked=blocked,prevented=prevented,
            new_disputes=len(born),knowledge_share=self.known.mean(),mean_prevention=self.effort.mean(),
            occupied=len(current),archived_this_round=sum(r['time']==t for r in self.retirements)))
        for f in range(c.firms):self.firm_records.append(dict(time=t,firm=f,pressure=pressure[f],
            prevention=self.effort[f],suppression=suppression[f],dependence=w['dependence'][f]))
        return born

    def attach(self,result):
        result['dynamics']=pd.DataFrame(self.records)
        result['firm_dynamics']=pd.DataFrame(self.firm_records)
        result['retirements']=pd.DataFrame(self.retirements,columns=['time','case_id','employee'])
        result['cases']['retired_at']=result['cases'].case_id.map({r['case_id']:r['time'] for r in self.retirements})


def window_metrics(cases,start,end,followup):
    """Incident cohorts with identical followup; may contain repeated employees."""
    g=cases[cases.created.ge(start)&cases.created.lt(end)]; n=len(g)
    deadline=g.created+followup
    d=g.d_submitted.lt(deadline);e=g.e_submitted.lt(deadline);solved=g.resolved.lt(deadline)
    def share(x):return float(x.sum()/n) if n else np.nan
    return dict(cases=n,employees=g.employee.nunique(),domestic_users=int(d.sum()),customer_users=int(e.sum()),
        customer_employees=g.loc[e,'employee'].nunique(),domestic_employees=g.loc[d,'employee'].nunique(),
        domestic_use=share(d),customer_use=share(e),both_use=share(d&e),
        first_customer=share(e&(~d|g.e_submitted.lt(g.d_submitted))),
        d_to_e=share(d&e&g.d_submitted.lt(g.e_submitted)),resolved=share(solved),unresolved=share(~solved))


def audit_continuous(r):
    ca=r['cases'];tr=r['trajectory'];dy=r['dynamics']
    assert ca.case_id.is_unique
    np.testing.assert_array_equal(dy.potential,dy.blocked+dy.prevented+dy.new_disputes)
    np.testing.assert_array_equal(tr.total_cases,tr.unresolved+tr.cumulative_resolved)
    np.testing.assert_array_equal(tr.new_disputes,dy.new_disputes)
    assert ca.loc[ca.retired_at.notna(),'resolved'].isna().all()
    assert ca.loc[ca.resolved.isna(),'compensation'].eq(0).all()
    assert ca.loc[ca.resolved.notna(),'compensation'].eq(1).all()
    ev=r['events'];assert ev[ev.event=='case_resolved'].case_id.is_unique
    for k in ['d','e']:
        assert ca.loc[ca[k+'_submitted'].notna(),k+'_status'].ne('unused').all()
