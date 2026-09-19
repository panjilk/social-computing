"""Case-based domestic / commercial-customer choice model.

Exploratory mechanism model, separate from the A/B threshold baseline.
No legal institution or real firm is represented by calibrated parameters.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class ChannelConfig:
    n: int = 120
    firms: int = 6
    steps: int = 45
    degree: int = 6
    network: str = 'er'
    initial_dispute: float = .65
    new_dispute_rate: float = 0.
    domestic_enabled: bool = True
    customer_enabled: bool = True
    firm_response_enabled: bool = True
    domestic_delay: int = 2
    domestic_capacity: int = 12
    domestic_success: float = .55
    # domestic_success is institutional remedy-support probability, not fulfillment.
    domestic_implementation_enabled: bool = True
    domestic_implementation_delay: int = 1
    domestic_implementation_capacity: int = 12
    domestic_implementation_success: float = .85
    material_rate: float = .7
    material_requires_support: bool = False
    customer_delay: int = 2
    customer_capacity: int = 12
    customer_review_base: float = .2
    customer_review_merit: float = .45
    evidence_help: float = .3
    firm_delay: int = 1
    firm_capacity: int = 3
    firm_customer_base: float = .2
    dependence_effect: float = .65
    customer_availability: float = .8
    employee_reachability: float = .9
    dependence_mean: float = .5
    dependence_spread: float = .1
    domestic_prior: float = .55
    customer_prior: float = .45
    domestic_information: float = .7
    domestic_signal: float = .55
    perceived_dependence: float = .25
    perceived_evidence_bonus: float = .18
    belief_spread: float = .2
    domestic_cost: float = .18
    customer_cost: float = .28
    multi_extra_cost: float = .02
    benefit: float = 1.
    wait_discount: float = .035
    overdue_decay: float = .12
    expected_domestic_wait: int = 3
    expected_customer_wait: int = 3
    review_interval: int = 1
    escalation_wait: int = 3
    social_weight: float = .2
    visibility_window: int = 3
    observation_mode: str = 'pending_or_recent'
    observation_probability: float = 1.
    outcome_learning_weight: float = .3
    outcome_prior_strength: float = 2.
    allow_escalation: bool = True
    subjective_route_overlap: float = 0.
    prevention: float = 0.
    tie_break: str = 'legacy'
    wait_learning_weight: float = 0.
    wait_prior_strength: float = 2.

    def __post_init__(self):
        if self.tie_break not in ('legacy','random'):
            raise ValueError('Invalid tie_break')
        if not 0 <= self.wait_learning_weight <= 1 or not np.isfinite(self.wait_learning_weight):
            raise ValueError('Invalid wait_learning_weight')
        if self.wait_prior_strength <= 0 or not np.isfinite(self.wait_prior_strength):
            raise ValueError('Invalid wait_prior_strength')
        ints=('n','firms','steps','degree','domestic_delay','domestic_capacity',
              'customer_delay','customer_capacity','firm_delay','firm_capacity',
              'expected_domestic_wait','expected_customer_wait','review_interval',
              'escalation_wait','visibility_window','domestic_implementation_delay',
              'domestic_implementation_capacity')
        for name in ints:
            v=getattr(self,name)
            if type(v) is not int or v<0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not (self.n>=4 and 1<=self.firms<=self.n and self.steps>=1):
            raise ValueError('Invalid population or horizon')
        if not (2<=self.degree<self.n and self.degree%2==0):
            raise ValueError('degree must be even and in [2,n)')
        if min(self.review_interval,self.visibility_window,self.expected_domestic_wait,
               self.expected_customer_wait)<1 or self.network not in ('er','ws','ba'):
            raise ValueError('Invalid timing or network')
        if self.observation_mode not in ('recent', 'pending_or_recent'):
            raise ValueError('Invalid observation_mode')
        if self.dependence_spread > min(self.dependence_mean, 1-self.dependence_mean)+1e-12:
            raise ValueError('dependence_spread must fit within [0,1] without clipping')
        for name in ('domestic_enabled','customer_enabled','firm_response_enabled','allow_escalation',
                     'domestic_implementation_enabled','material_requires_support'):
            if type(getattr(self,name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        nonnegative=('domestic_cost','customer_cost','multi_extra_cost','benefit',
                     'wait_discount','overdue_decay','outcome_prior_strength','wait_prior_strength')
        for name,value in asdict(self).items():
            if isinstance(value,float):
                if not np.isfinite(value) or value<0 or (name not in nonnegative and value>1):
                    raise ValueError(f'Invalid {name}')
        # JSON can supply integer-valued probabilities and costs as well.
        for name in ('initial_dispute','new_dispute_rate','domestic_success','material_rate',
                     'domestic_implementation_success',
                     'customer_review_base','customer_review_merit','evidence_help',
                     'firm_customer_base','dependence_effect','customer_availability',
                     'employee_reachability','dependence_mean','dependence_spread',
                     'domestic_prior','customer_prior','domestic_information',
                     'perceived_dependence','perceived_evidence_bonus','belief_spread',
                     'social_weight','prevention','domestic_signal','observation_probability',
                     'outcome_learning_weight','subjective_route_overlap'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1:
                raise ValueError(f'{name} must be a probability or bounded coefficient')
        for name in nonnegative:
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.outcome_prior_strength <= 0:
            raise ValueError('outcome_prior_strength must be positive')


def world(c, seed):
    def rng(key):
        return np.random.default_rng(np.random.SeedSequence([seed,100,key]))
    net=rng(0); ini=rng(1); traits=rng(2); firms_rng=rng(3)
    netseed=int(net.integers(0,2**32-1))
    if c.network=='er':
        g=nx.erdos_renyi_graph(c.n,c.degree/(c.n-1),seed=netseed)
    elif c.network=='ws':
        g=nx.watts_strogatz_graph(c.n,c.degree,.15,seed=netseed)
    else:
        g=nx.barabasi_albert_graph(c.n,c.degree//2,seed=netseed)
    firm=ini.permutation(np.arange(c.n)%c.firms)
    initial=ini.random(c.n)<c.initial_dispute
    has_customer=firms_rng.random(c.firms)<c.customer_availability
    offsets=firms_rng.uniform(-1,1,c.firms)
    offsets-=offsets.mean()
    scale=np.max(np.abs(offsets))
    offsets=offsets/scale if scale>0 else np.zeros(c.firms)
    dependence=c.dependence_mean+c.dependence_spread*offsets
    reachable=has_customer[firm] & (traits.random(c.n)<c.employee_reachability)
    merit=traits.uniform(.2,1,c.n)
    beliefs=np.clip(np.column_stack((
        np.full(c.n,c.domestic_prior+c.domestic_information*(c.domestic_signal-.55)),
        c.customer_prior+c.perceived_dependence*(dependence[firm]-.5)))+
        traits.uniform(-c.belief_spread,c.belief_spread,(c.n,2)),0,1)
    costs=traits.uniform(.5,1.5,(c.n,2))*[c.domestic_cost,c.customer_cost]
    # Independent process streams: increasing steps preserves every process prefix.
    draws={name:rng(key).random((c.steps,c.n)) for key,name in enumerate(
        ('arrivals','material','review','domestic_outcome','customer_outcome','domestic_implementation'),10)}
    # Fixed directed sharing links, independent of outcomes and network generation.
    sharing=rng(4).random((c.n,c.n))<c.observation_probability
    return dict(tie_draws=rng(31).random((c.steps,c.n)),graph=g,sharing=sharing,firm=firm,initial=initial,has_customer=has_customer,
                dependence=dependence,reachable=reachable,merit=merit,
                beliefs=beliefs,costs=costs,draws=draws)


def review_probability(c, merit, material):
    return np.clip(c.customer_review_base+c.customer_review_merit*merit+
                   c.evidence_help*material,0,1)


def firm_customer_probability(c, dependence):
    return np.clip(c.firm_customer_base+c.dependence_effect*dependence,0,1)


ACTIVE=('submitted','forwarded','awaiting_implementation')
CASE_COLUMNS=['case_id','employee','firm','created','first_submitted','d_submitted',
    'e_submitted','d_responded','e_responded','material_at','customer_intervened_at',
    'd_due','e_due','d_status','e_status','resolved','resolved_by','compensation',
    'initial_choice','status','merit','reachable','dependence','d_finished','e_finished',
    'd_institution_outcome','d_implementation_at']


def observe(c, cases, t, neighbors, sharing, outcome_window=None):
    """Freeze past information; pending is not a success/failure observation.

    Action fractions use ALL neighbors as denominator. Outcome counts use unique
    visible employees per route and outcome in the trailing window. A person can
    have different outcomes on different cases; cancellation is never failure.
    """
    flags={name:np.zeros((c.n,2),dtype=bool)
           for name in ('recent','pending','success','failure')}
    for case in cases:
        i=case['employee']
        for k,key in enumerate(('d','e')):
            submitted=case[key+'_submitted']; finished=case[key+'_finished']
            status=case[key+'_status']
            if submitted is not None and 1<=t-submitted<=c.visibility_window:
                flags['recent'][i,k]=True
            if submitted is not None and submitted<t and status in ACTIVE:
                flags['pending'][i,k]=True
            if finished is not None and 1<=t-finished<=(c.visibility_window if outcome_window is None else outcome_window):
                label={'successful':'success','closed_unresolved':'failure'}.get(status)
                if label: flags[label][i,k]=True
    flags['activity']=flags['recent'] | (flags['pending'] if
        c.observation_mode=='pending_or_recent' else False)
    signals={name:np.zeros((c.n,2)) for name in flags}
    for i,nb in enumerate(neighbors):
        if not nb: continue
        visible=[j for j in nb if sharing[i,j]]
        for name,values in flags.items():
            counts=values[visible].sum(axis=0)
            signals[name][i]=counts if name in ('success','failure') else counts/len(nb)
    return signals,flags


def joint_success(probabilities, overlap=0.):
    """At most two subjective opportunities: independence to fully nested success."""
    if not probabilities: return 0.
    if len(probabilities)==1: return probabilities[0]
    low,high=sorted(probabilities)
    return high+(1-overlap)*low*(1-high)


def perceived_probability(c, w, i, successes=None, failures=None, material=False):
    """Current channel belief, before case-specific waiting discounts."""
    probability=w['beliefs'][i].copy()
    if successes is not None:
        # Recompute from fixed prior and current window; do not count the same
        # observed outcome repeatedly as new evidence every round.
        posterior=(c.outcome_prior_strength*probability+successes)/(
            c.outcome_prior_strength+successes+failures)
        probability=(1-c.outcome_learning_weight)*probability+c.outcome_learning_weight*posterior
    if material:
        probability[1]=min(1.,probability[1]+c.perceived_evidence_bonus)
    return probability


def choose_additions(c, w, case, t, social, successes=None, failures=None):
    """Compare incremental utility; paid costs are sunk, existing routes stay open."""
    i=case['employee']
    probability=perceived_probability(c,w,i,successes,failures,
        case['material_at'] is not None and case['material_at']<t)
    expected=(w['expected_wait'][i] if 'expected_wait' in w else
              (c.expected_domestic_wait,c.expected_customer_wait))
    available=(c.domestic_enabled,c.customer_enabled and case['reachable'] and
               ('customer_known' not in w or w['customer_known'][i]))
    base=[]; options=[]
    for k,key in enumerate(('d','e')):
        status=case[key+'_status']
        if status in ACTIVE:
            age=t-case[key+'_submitted']
            p=probability[k]*np.exp(-c.overdue_decay*max(0,age-expected[k]))
            base.append(p*np.exp(-c.wait_discount*max(1,expected[k]-age)))
        if status=='unused' and available[k]:
            options.append(k)
    costs=w['costs'][i]*(1-c.social_weight*np.asarray(social))
    candidates=[()]+[(k,) for k in options]
    if len(options)==2:
        candidates.append((0,1))
    utilities=[]
    for addition in candidates:
        ps=base+[probability[k]*np.exp(-c.wait_discount*expected[k]) for k in addition]
        benefit=c.benefit*joint_success(ps,c.subjective_route_overlap)
        # Extra coordination cost only when an addition creates a two-route case.
        existing=sum(case[k+'_submitted'] is not None for k in ('d','e'))
        extra=c.multi_extra_cost if addition and existing+len(addition)==2 else 0.
        utilities.append(float(benefit-sum(costs[k] for k in addition)-extra))
    best=int(np.argmax(utilities))
    tied=np.flatnonzero(np.asarray(utilities)==utilities[best])
    if c.tie_break=='random' and len(tied)>1:
        best=int(tied[min(int(w['tie_draws'][t,i]*len(tied)),len(tied)-1)])
    selected=candidates[best]
    return selected,dict(p_d=float(probability[0]),p_e=float(probability[1]),
        expected_wait_d=float(expected[0]),expected_wait_e=float(expected[1]),
        cost_d=float(costs[0]),cost_e=float(costs[1]),
        utilities=';'.join(f'{"+".join("de"[k] for k in a) or "wait"}:{u:.6f}'
                           for a,u in zip(candidates,utilities)))


def simulate_channels(c, seed=0, information=None, customer_network=None, cohort_schedule=None, dynamics=None):
    if dynamics is not None:
        from dataclasses import replace
        c=replace(c)
        if information is not None or cohort_schedule is not None:
            raise ValueError('Continuous dynamics cannot combine with other time hooks')
    w=world(c,seed); cases=[]; current={}; events=[]; decisions=[]; rows=[]
    if dynamics is not None: dynamics.start(c,w,seed)
    if cohort_schedule is not None:
        cohort_schedule.start(c,w,seed)
    if customer_network is not None:
        customer_network.start(c,w,seed)
    initial_beliefs=w['beliefs'].copy()
    if information is not None:
        information.start(c,w,seed)
    protected=np.zeros(c.firms,dtype=bool)
    neighbors=[list(w['graph'].neighbors(i)) for i in range(c.n)]

    def log(t,case,event):
        if event in ('domestic_implementation_success','domestic_implementation_failure',
                     'domestic_no_remedy','d_cancelled_by_resolution'):
            case['d_finished']=t
        if event in ('customer_firm_success','customer_firm_failure','customer_declined','e_cancelled_by_resolution'):
            case['e_finished']=t
        events.append(dict(time=t,case_id=case['case_id'],employee=case['employee'],event=event))

    def create(i,t):
        f=int(w['firm'][i])
        case={key:None for key in CASE_COLUMNS}
        case.update(case_id=f'{i}:{t}',employee=i,firm=f,created=t,
            d_status='unused',e_status='unused',compensation=0.,initial_choice='',
            status='unsubmitted',merit=float(w['merit'][i]),reachable=bool(w['reachable'][i]),
            dependence=float(w['dependence'][f]))
        cases.append(case); current[i]=case; log(t,case,'dispute_created')

    if cohort_schedule is None:
        for i in np.flatnonzero(w['initial']):
            create(int(i),0)
    for t in range(c.steps):
        new=0
        if dynamics is not None:
            for i in dynamics.advance(t,c,w,cases,current,neighbors):
                create(int(i),t); new+=1
        elif cohort_schedule is not None:
            for i in cohort_schedule.at(t):
                create(int(i),t); new+=1
        else:
            new=int(w['initial'].sum()) if t==0 else 0
        if t>0 and cohort_schedule is None and dynamics is None:
            for i in range(c.n):
                rate=c.new_dispute_rate*(1-c.prevention*protected[w['firm'][i]])
                if i not in current and w['draws']['arrivals'][t,i]<rate:
                    create(i,t); new+=1
        signals,flags=observe(c,cases,t,neighbors,w['sharing'],
            dynamics.outcome_window if dynamics is not None else
            None if cohort_schedule is None else cohort_schedule.outcome_window)
        if c.wait_learning_weight:
            from src.wait_learning import expected_waits
            w['expected_wait']=expected_waits(c,cases,t,neighbors,w['sharing'],
                c.visibility_window if cohort_schedule is None else cohort_schedule.outcome_window)
        if cohort_schedule is not None:
            cohort_schedule.record(t,c,w,signals)
        objective=c.domestic_success
        if information is not None:
            objective=information.advance(t,w,neighbors)
            information.record(t,c,w,signals,current)
        seen=flags['activity']; social=signals['activity']
        first=[]; d_new=[]; e_new=[]; additions=[]
        # Freeze evidence availability before domestic processing in this round.
        materials={i:case['material_at'] is not None and case['material_at']<t
                   for i,case in current.items()}
        for i,case in sorted(current.items()):
            age=t-case['created']
            if cohort_schedule is not None and age<cohort_schedule.preparation[i]:
                continue
            if age%c.review_interval:
                continue
            if case['first_submitted'] is not None and not c.allow_escalation:
                continue
            if case['first_submitted'] is not None and t-case['first_submitted']<c.escalation_wait:
                continue
            choice,details=choose_additions(c,w,case,t,social[i],signals['success'][i],signals['failure'][i])
            decisions.append(dict(time=t,case_id=case['case_id'],employee=i,
                selected='+'.join('de'[k] for k in choice) or 'no_addition',
                material_known=materials[i],social_d=float(social[i,0]),
                social_e=float(social[i,1]),
                **{f'{name}_{key}':float(signals[name][i,k]) for name in
                   ('pending','recent','success','failure') for k,key in enumerate(('d','e'))},**details))
            if not choice:
                continue
            is_first=case['first_submitted'] is None
            if is_first:
                case['first_submitted']=t
                case['initial_choice']='+'.join('de'[k] for k in choice)
                first.append(case)
            else:
                additions.append(case)
            for k in choice:
                key='de'[k]; case[key+'_submitted']=t; case[key+'_status']='submitted'
                (d_new if k==0 else e_new).append(case)
                log(t,case,'domestic_received' if k==0 else 'customer_received')
        d_queue=sorted([v for v in current.values() if v['d_status']=='submitted' and
                        t-v['d_submitted']>=c.domestic_delay],key=lambda v:(v['d_submitted'],v['employee']))
        for case in d_queue[:c.domestic_capacity]:
            i=case['employee']; case['d_responded']=t
            log(t,case,'domestic_responded')
            supported=w['draws']['domestic_outcome'][t,i]<objective*(.5+.5*case['merit'])
            if (not c.material_requires_support or supported) and w['draws']['material'][t,i]<c.material_rate*(.5+.5*case['merit']):
                case['material_at']=t; log(t,case,'domestic_material_issued')
            else:
                log(t,case,'domestic_material_not_issued')
            case['d_institution_outcome']='remedy_supported' if supported else 'no_remedy'
            if supported:
                case['d_status']='awaiting_implementation'
                case['d_due']=t+c.domestic_implementation_delay
                log(t,case,'domestic_remedy_supported')
            else:
                case['d_status']='closed_unresolved'
                log(t,case,'domestic_no_remedy')
        e_queue=sorted([v for v in current.values() if v['e_status']=='submitted' and
                        t-v['e_submitted']>=c.customer_delay],key=lambda v:(v['e_submitted'],v['employee']))
        reviewed = (e_queue[:c.customer_capacity] if customer_network is None else
                    customer_network.select(e_queue,t))
        for case in reviewed:
            i=case['employee']; case['e_responded']=t; log(t,case,'customer_reviewed')
            p=review_probability(c,case['merit'],materials[i])
            if w['draws']['review'][t,i]<p:
                case['customer_intervened_at']=t; case['e_status']='forwarded'
                case['e_due']=t+c.firm_delay; log(t,case,'customer_intervened')
            else:
                case['e_status']='closed_unresolved'; log(t,case,'customer_declined')
        # Evaluate the two routes before committing case resolution, so same-round
        # successes remain joint outcomes with one compensation and one case count.
        successes={}
        if c.domestic_implementation_enabled:
            queue=sorted([v for v in current.values() if
                v['d_status']=='awaiting_implementation' and v['d_due']<=t],
                key=lambda v:(v['d_due'],v['employee']))
            for case in queue[:c.domestic_implementation_capacity]:
                i=case['employee']; case['d_implementation_at']=t
                success=w['draws']['domestic_implementation'][t,i]<c.domestic_implementation_success
                case['d_status']='successful' if success else 'closed_unresolved'
                log(t,case,'domestic_implementation_success' if success else 'domestic_implementation_failure')
                if success: successes.setdefault(i,[]).append('d')
        if c.firm_response_enabled:
            for f in range(c.firms):
                queue=[]
                for case in current.values():
                    due=[case[k+'_due'] for k in ('e',) if case[k+'_status']=='forwarded'
                         and case[k+'_due']<=t]
                    if case['firm']==f and due:
                        queue.append((min(due),case['employee'],case))
                for _,i,case in sorted(queue,key=lambda x:x[:2])[:c.firm_capacity]:
                    log(t,case,'firm_case_processed')  # commercial-customer route only
                    for key in ('e',):
                        if case[key+'_status']!='forwarded' or case[key+'_due']>t:
                            continue
                        p=firm_customer_probability(c,case['dependence'])
                        draw=w['draws']['customer_outcome'][t,i]
                        success=draw<p
                        case[key+'_status']='successful' if success else 'closed_unresolved'
                        log(t,case,('domestic' if key=='d' else 'customer')+
                            ('_firm_success' if success else '_firm_failure'))
                        if success:
                            successes.setdefault(i,[]).append(key)
        solved=[]
        for i,successful in sorted(successes.items()):
            case=current[i]
            case.update(resolved=t,resolved_by='+'.join(successful),compensation=1.,status='resolved')
            for key in ('d','e'):
                if case[key+'_status'] in ACTIVE:
                    case[key+'_status']='cancelled_by_resolution'
                    log(t,case,key+'_cancelled_by_resolution')
            solved.append(case); log(t,case,'case_resolved'); protected[case['firm']]=True
        for case in solved:
            del current[case['employee']]
        for case in current.values():
            case['status']=('unsubmitted' if case['first_submitted'] is None else
                'pending' if any(case[k+'_status'] in ACTIVE for k in ('d','e')) else 'closed_unresolved')
        rows.append(dict(time=t,new_disputes=new,new_complaints=len(first),
            new_complainants=len({v['employee'] for v in first}),
            new_domestic=len(d_new),new_customer=len(e_new),new_additional_channel=len(additions),
            unresolved=(sum(v['resolved'] is None for v in cases) if dynamics is not None else len(current)),pending=sum(v['status']=='pending' for v in current.values()),
            unsubmitted=sum(v['status']=='unsubmitted' for v in current.values()),
            closed_unresolved=sum(v['status']=='closed_unresolved' for v in current.values()),
            pending_domestic=sum(v['d_status'] in ACTIVE for v in current.values()),
            pending_domestic_institution=sum(v['d_status']=='submitted' for v in current.values()),
            pending_domestic_implementation=sum(v['d_status']=='awaiting_implementation' for v in current.values()),
            new_domestic_supported=sum(v['d_responded']==t and v['d_institution_outcome']=='remedy_supported' for v in cases),
            new_domestic_implemented=sum(v['d_implementation_at']==t and v['d_status']=='successful' for v in cases),
            pending_customer=sum(v['e_status'] in ACTIVE for v in current.values()),
            new_resolved=len(solved),cumulative_resolved=sum(v['resolved'] is not None for v in cases),
            cumulative_complainants=len({v['employee'] for v in cases if v['first_submitted'] is not None}),
            total_cases=len(cases),visible_domestic=int(seen[:,0].sum()),visible_customer=int(seen[:,1].sum())))
    case_df=pd.DataFrame(cases,columns=CASE_COLUMNS)
    for col in ('created','first_submitted','d_submitted','e_submitted','d_responded',
                'e_responded','material_at','customer_intervened_at','d_due','e_due','resolved','d_finished','e_finished',
                'd_implementation_at'):
        case_df[col]=pd.to_numeric(case_df[col])
    case_df['processing_time']=case_df.resolved-case_df.first_submitted
    case_df['observed_wait']=case_df.resolved.fillna(c.steps-1)-case_df.first_submitted
    case_df['dispute_age_observed']=case_df.resolved.fillna(c.steps-1)-case_df.created
    people=pd.DataFrame(dict(employee=range(c.n),firm=w['firm'],initial_dispute=w['initial'],
        reachable=w['reachable'],merit=w['merit'],belief_d=initial_beliefs[:,0],belief_e=initial_beliefs[:,1],
        cost_d=w['costs'][:,0],cost_e=w['costs'][:,1]))
    firms=pd.DataFrame(dict(firm=range(c.firms),has_customer=w['has_customer'],dependence=w['dependence']))
    result=dict(trajectory=pd.DataFrame(rows),cases=case_df,
        events=pd.DataFrame(events,columns=['time','case_id','employee','event']),
        decisions=pd.DataFrame(decisions,columns=['time','case_id','employee','selected',
            'material_known','social_d','social_e',
            *[f'{name}_{key}' for name in ('pending','recent','success','failure') for key in ('d','e')],
            'p_d','p_e','cost_d','cost_e','expected_wait_d','expected_wait_e','utilities']),
        employees=people,firms=firms,graph=w['graph'])
    if information is not None:
        result['belief_panel']=pd.DataFrame(information.panel)
        result['information_events']=pd.DataFrame(information.events,
            columns=['time','employee','source'])
    if customer_network is not None:
        customer_network.attach(result)
    if cohort_schedule is not None:
        cohort_schedule.attach(result)
    if dynamics is not None:
        dynamics.attach(result)
    return result
