"""Exact label-exchange diagnostic with matched mirrored stochastic worlds."""
import json
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
import src.channel_choice as model
from src.cohort_trends import cohort_measures
from experiments.competition_refinement import scenarios
from experiments.channel_experiments import audit


def run():
    spec=json.loads(Path('configs/competition_refinement.json').read_text(encoding='utf-8'))
    c,s,_,_,_=scenarios(spec)['symmetric_random']; original=model.world
    out=Path('outputs/competition_refinement_v1'); rows=[]
    for seed in range(spec['repeats']):
        def mirrored(config,seed):
            w=original(config,seed)
            w['beliefs']=w['beliefs'][:,::-1].copy(); w['costs']=w['costs'][:,::-1].copy()
            d=w['draws']; d['domestic_outcome'],d['review']=d['review'],d['domestic_outcome']
            d['domestic_implementation'],d['customer_outcome']=d['customer_outcome'],d['domestic_implementation']
            return w
        a=model.simulate_channels(c,seed,cohort_schedule=replace(s))
        with patch.object(model,'world',mirrored): b=model.simulate_channels(c,seed,cohort_schedule=replace(s))
        audit(a); audit(b)
        for left,right in [('d_submitted','e_submitted'),('e_submitted','d_submitted'),
                           ('d_finished','e_finished'),('e_finished','d_finished'),('resolved','resolved')]:
            np.testing.assert_allclose(a['cases'][left],b['cases'][right],equal_nan=True)
        for tag,r in [('original',a),('mirror',b)]:
            m=cohort_measures(r['cases'],s); g=m[(m.population=='all')&(m.cohort>=6)]
            rows.append(dict(seed=seed,world=tag,domestic_use=(g.domestic_use*g.cases).sum()/g.cases.sum(),
                customer_use=(g.customer_use*g.cases).sum()/g.cases.sum()))
    df=pd.DataFrame(rows); df.to_csv(out/'symmetry_mirror.csv',index=False)
    result=dict(paired_seeds=spec['repeats'],additional_runs=len(rows),
        exact_channel_exchange_of_submissions_finishes_resolution=True,
        balanced_domestic_use=df.domestic_use.mean(),balanced_customer_use=df.customer_use.mean())
    (out/'symmetry_mirror.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': run()
