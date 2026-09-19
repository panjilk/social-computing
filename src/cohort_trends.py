"""Exogenous, paired incident cohorts. Not calibrated calendar-time forecasts."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.channel_choice import perceived_probability


@dataclass
class CohortSchedule:
    cohorts: int = 8
    spacing: int = 6
    followup: int = 24
    preparation_max: int = 3
    outcome_window: int = 18

    def start(self,c,w,seed):
        for name in ('cohorts','spacing','followup','outcome_window'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.preparation_max) is not int or self.preparation_max<0:
            raise ValueError('preparation_max must be nonnegative')
        if self.cohorts>c.n or (self.cohorts-1)*self.spacing+self.followup>c.steps:
            raise ValueError('Each cohort needs a full observation window')
        if c.new_dispute_rate or c.prevention:
            raise ValueError('Scheduled cohorts exclude endogenous new disputes and prevention')
        rng=np.random.default_rng(np.random.SeedSequence([seed,1801]))
        self.cohort=np.empty(c.n,dtype=int)
        # Random assignment, balanced in total and within each firm where feasible.
        offset=0
        for f in range(c.firms):
            ids=rng.permutation(np.flatnonzero(w['firm']==f))
            self.cohort[ids]=(np.arange(len(ids))+offset)%self.cohorts
            offset=(offset+len(ids))%self.cohorts
        prep=np.random.default_rng(np.random.SeedSequence([seed,1802]))
        self.preparation=prep.integers(0,self.preparation_max+1,c.n)
        self.incident=w['initial'].copy()
        self.onset=self.cohort*self.spacing
        self.rows=[]

    def at(self,t):
        return np.flatnonzero(self.incident&(self.onset==t))

    def record(self,t,c,w,signals):
        beliefs=np.array([perceived_probability(c,w,i,signals['success'][i],signals['failure'][i])
                          for i in range(c.n)])
        arriving=self.incident&(self.onset==t)
        self.rows.append(dict(time=t,mean_belief_d=beliefs[:,0].mean(),mean_belief_e=beliefs[:,1].mean(),
            arriving_belief_d=beliefs[arriving,0].mean() if arriving.any() else np.nan,
            arriving_belief_e=beliefs[arriving,1].mean() if arriving.any() else np.nan))

    def attach(self,result):
        result['employees']['incident_cohort']=self.cohort
        result['employees']['scheduled_incident']=np.where(self.incident,self.onset,-1)
        result['employees']['preparation_rounds']=self.preparation
        # In this variant initial_dispute means incident eligibility, not all occurring at t=0.
        result['employees']=result['employees'].rename(columns={'initial_dispute':'incident_eligible'})
        result['cases']['cohort']=result['cases'].employee.map(lambda i:int(self.cohort[i]))
        result['belief_history']=pd.DataFrame(self.rows)


def cohort_measures(cases,schedule):
    """All cohorts evaluated at the same case age [0, H); never use later outcomes."""
    rows=[]
    for cohort in range(schedule.cohorts):
        group=cases[cases.cohort==cohort]; end=cohort*schedule.spacing+schedule.followup
        for population,g in [('all',group),('both_reachable',group[group.reachable.astype(bool)])]:
            n=len(g)
            def share(mask): return float(mask.sum()/n) if n else np.nan
            d=g.d_submitted.lt(end); e=g.e_submitted.lt(end)
            fd=d&(~e|g.d_submitted.lt(g.e_submitted))
            fe=e&(~d|g.e_submitted.lt(g.d_submitted))
            same=d&e&g.d_submitted.eq(g.e_submitted)
            none=~(d|e)
            rows.append(dict(cohort=cohort,onset=cohort*schedule.spacing,followup=schedule.followup,
                population=population,cases=n,domestic_users=int(d.sum()),customer_users=int(e.sum()),
                first_d=share(fd),first_e=share(fe),first_both=share(same),no_submission=share(none),
                domestic_use=share(d),customer_use=share(e),both_use=share(d&e),
                domestic_only=share(d&~e),customer_only=share(e&~d),
                d_to_e=share(d&e&g.d_submitted.lt(g.e_submitted)),
                e_to_d=share(d&e&g.e_submitted.lt(g.d_submitted)),
                resolved=share(g.resolved.lt(end)),unresolved=share(~g.resolved.lt(end)),
                customer_minus_domestic=share(e)-share(d)))
    return pd.DataFrame(rows)


def first_submission_series(cases,steps):
    rows=[]
    for t in range(steps):
        g=cases[cases.first_submitted==t]; n=len(g)
        counts=[int((g.initial_choice==key).sum()) for key in ('d','e','d+e')]
        rows.append(dict(time=t,first_submissions=n,first_d_count=counts[0],first_e_count=counts[1],
            first_both_count=counts[2],**{name:count/n if n else np.nan for name,count in
            zip(('first_d_share','first_e_share','first_both_share'),counts)}))
    return pd.DataFrame(rows)
