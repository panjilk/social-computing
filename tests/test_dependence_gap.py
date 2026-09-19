import unittest
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from src.channel_choice import ChannelConfig,world,firm_customer_probability
from experiments.dependence_gap import configurations,rank_groups,measure
from src.cohort_trends import CohortSchedule


class DependenceTests(unittest.TestCase):
    def test_boundary_precision_not_real_overflow(self):
        c=ChannelConfig(dependence_mean=.9,dependence_spread=.1)
        w=world(c,0)
        self.assertLessEqual(w['dependence'].max(),1.+1e-12)
        with self.assertRaises(ValueError):replace(c,dependence_spread=.1001)

    def test_design_rank_mean_and_nonclipping(self):
        spec=json.loads(Path('configs/dependence_gap.json').read_text(encoding='utf-8'))
        variants,_,_=configurations(spec)
        for seed in range(5):
            ref=None
            for c in variants.values():
                w=world(c,seed); g=rank_groups(pd.DataFrame({'firm':range(c.firms),'dependence':w['dependence']}))
                if ref is None:ref=g
                self.assertEqual(ref,g)
                self.assertAlmostEqual(w['dependence'].mean(),c.dependence_mean)
                p=firm_customer_probability(c,w['dependence'])
                self.assertTrue(((p>0)&(p<1)).all())

    def test_no_objective_removes_dependence_probability(self):
        c=ChannelConfig(dependence_effect=0.,firm_customer_base=.525)
        np.testing.assert_allclose(firm_customer_probability(c,np.array([0.,.5,1.])),.525)

    def test_group_denominators_and_joint_success(self):
        ca=pd.DataFrame(dict(cohort=[0,0,0,0],firm=[0,0,1,1],
            d_submitted=[0,np.nan,0,np.nan],e_submitted=[1,0,0,np.nan],
            resolved=[2,2,2,np.nan],resolved_by=['d','e','d+e','']))
        m=measure(ca,CohortSchedule(cohorts=4,spacing=3,followup=10),{'all':{0,1},'high':{0},'low':{1}})
        g=m[(m.period=='early')&(m.group=='all')].iloc[0]
        self.assertEqual(g.customer_use,.75)
        self.assertAlmostEqual(g.customer_success,2/3)
        self.assertEqual(g.resolved,.75)
        self.assertTrue(m[m.period=='late'].resolved.isna().all())


if __name__=='__main__':unittest.main()
