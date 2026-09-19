"""Optional firm-customer bipartite network with customer-specific review queues."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CustomerNetwork:
    customers: int = 6
    active_customers: int = 6
    pooled: bool = False
    allocation: str = 'equal'

    def start(self,c,w,seed):
        if (type(self.customers) is not int or type(self.active_customers) is not int
                or not 1 <= self.active_customers <= min(self.customers,c.firms)
                or type(self.pooled) is not bool
                or self.allocation not in ('equal','degree')):
            raise ValueError('Invalid customer network configuration')
        if self.allocation=='equal' and c.customer_capacity % self.customers:
            raise ValueError('Total review capacity must divide equally across customers')
        rng=np.random.default_rng(np.random.SeedSequence([seed,701]))
        order=rng.permutation(c.firms)
        self.assignment=np.full(c.firms,-1,dtype=int)
        self.assignment[order]=np.arange(c.firms)%self.active_customers
        self.assignment[~w['has_customer']]=-1
        self.capacity=c.customer_capacity//self.customers
        self.total_capacity=c.customer_capacity
        self.capacities=np.full(self.customers,self.capacity,dtype=int)
        if self.allocation=='degree':
            degree=np.bincount(self.assignment[self.assignment>=0],minlength=self.customers)
            # No connections: equal reserve allocation; no case can access it.
            weights=degree.astype(float) if degree.sum() else np.ones(self.customers)
            quota=c.customer_capacity*weights/weights.sum()
            self.capacities=np.floor(quota).astype(int)
            leftover=c.customer_capacity-int(self.capacities.sum())
            order=np.argsort(-(quota-self.capacities),kind='stable')
            self.capacities[order[:leftover]]+=1
        self.rows=[]
        self.edges=pd.DataFrame([dict(firm=f,customer=int(k),dependence=float(w['dependence'][f]))
            for f,k in enumerate(self.assignment) if k>=0],columns=['firm','customer','dependence'])

    def select(self,queue,t):
        # Input is global FIFO (submission time, employee id); preserve order.
        groups=[[] for _ in range(self.customers)]
        for case in queue:
            k=int(self.assignment[case['firm']])
            if k<0: raise AssertionError('Unreachable firm submitted to customer')
            groups[k].append(case)
        chosen=(queue[:self.total_capacity] if self.pooled else
                [case for k,group in enumerate(groups) for case in group[:self.capacities[k]]])
        ids={case['case_id'] for case in chosen}
        for k,group in enumerate(groups):
            served=sum(case['case_id'] in ids for case in group)
            self.rows.append(dict(time=t,customer=k,eligible=len(group),reviewed=served,
                eligible_left=len(group)-served,nominal_capacity=int(self.capacities[k]),
                oldest_eligible_age=max([t-case['e_submitted'] for case in group],default=0)))
        return [case for case in queue if case['case_id'] in ids]

    def attach(self,result):
        result['customer_edges']=self.edges
        result['customer_queues']=pd.DataFrame(self.rows)
        result['cases']['customer']=result['cases'].firm.map(lambda f:int(self.assignment[f]))
