"""Local waiting heuristic using completed durations AND pending lower bounds.

Not a survival estimator: pending ages are censored lower bounds, not completion
times. Cancellation by the other route is excluded. Recomputed from fixed prior.
"""
import numpy as np


def expected_waits(c,cases,t,neighbors,sharing,window):
    prior=np.array([c.expected_domestic_wait,c.expected_customer_wait],dtype=float)
    sums=np.zeros((c.n,2)); counts=np.zeros((c.n,2))
    for case in cases:
        owner=case['employee']
        viewers=[i for i,nb in enumerate(neighbors) if i==owner or (owner in nb and sharing[i,owner])]
        for k,key in enumerate(('d','e')):
            start=case[key+'_submitted']; end=case[key+'_finished']; status=case[key+'_status']
            if start is None or start>=t: continue
            if status in ('submitted','forwarded','awaiting_implementation'):
                duration=t-start
            elif status in ('successful','closed_unresolved') and end is not None and 1<=t-end<=window:
                duration=end-start
            else: continue
            sums[viewers,k]+=duration; counts[viewers,k]+=1
    posterior=(c.wait_prior_strength*prior+sums)/(c.wait_prior_strength+counts)
    return (1-c.wait_learning_weight)*prior+c.wait_learning_weight*posterior
