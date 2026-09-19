import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from src.channel_choice import ChannelConfig,simulate_channels,world,observe
from src.cohort_trends import CohortSchedule,cohort_measures,first_submission_series
from experiments.channel_experiments import audit


class CohortTests(unittest.TestCase):
    def config(self):
        return ChannelConfig(n=24,firms=3,degree=4,steps=16,new_dispute_rate=0.,prevention=0.)

    def schedule(self): return CohortSchedule(3,4,8,2,8)

    def test_single_cohort_reproduces_legacy(self):
        c=self.config(); a=simulate_channels(c,1)
        b=simulate_channels(c,1,cohort_schedule=CohortSchedule(1,1,16,0,c.visibility_window))
        pd.testing.assert_frame_equal(a['cases'],b['cases'].drop(columns='cohort'))
        pd.testing.assert_frame_equal(a['trajectory'],b['trajectory'])

    def test_paired_disjoint_arrivals_and_reproducibility(self):
        c=self.config(); a=simulate_channels(c,2,cohort_schedule=self.schedule())
        b=simulate_channels(replace(c,domestic_success=0.,firm_response_enabled=False),2,cohort_schedule=self.schedule())
        audit(a); audit(b)
        pd.testing.assert_frame_equal(a['cases'][['case_id','employee','created']],b['cases'][['case_id','employee','created']])
        self.assertTrue(a['cases'].employee.is_unique)
        d=simulate_channels(c,2,cohort_schedule=self.schedule())
        pd.testing.assert_frame_equal(a['cases'],d['cases'])

    def test_preparation_respected(self):
        c=replace(self.config(),initial_dispute=1.)
        r=simulate_channels(c,3,cohort_schedule=self.schedule())
        ca=r['cases']; prep=r['employees'].set_index('employee').preparation_rounds
        self.assertTrue((ca.first_submitted.dropna()-ca.loc[ca.first_submitted.notna(),'created']>=
                         ca.loc[ca.first_submitted.notna(),'employee'].map(prep)).all())

    def test_no_disputes_and_no_division_by_zero(self):
        c=replace(self.config(),initial_dispute=0.)
        s=self.schedule(); r=simulate_channels(c,0,cohort_schedule=s); audit(r)
        m=cohort_measures(r['cases'],s)
        self.assertTrue(m.customer_use.isna().all())
        self.assertEqual(r['trajectory'].new_complaints.sum(),0)

    def test_waiting_possible(self):
        c=replace(self.config(),initial_dispute=1.,domestic_cost=10.,customer_cost=10.)
        r=simulate_channels(c,0,cohort_schedule=self.schedule())
        self.assertTrue(r['cases'].first_submitted.isna().all())

    def test_equal_followup_ignores_later_additions_and_resolution(self):
        ca=pd.DataFrame(dict(cohort=[0,1],created=[0,4],reachable=[True,True],
            d_submitted=[0,4],e_submitted=[8,11],resolved=[8,12]))
        m=cohort_measures(ca,CohortSchedule(2,4,8,0,8))
        m=m[m.population=='all'].set_index('cohort')
        self.assertEqual(m.loc[0,'customer_use'],0.)
        self.assertEqual(m.loc[1,'customer_use'],1.)
        self.assertEqual(m.resolved.sum(),0.)
        np.testing.assert_allclose(m[['first_d','first_e','first_both','no_submission']].sum(axis=1),1.)

    def test_outcome_window_separate_and_no_future_leak(self):
        c=self.config()
        case=dict(employee=0,d_submitted=0,e_submitted=None,d_finished=1,e_finished=None,
                  d_status='successful',e_status='unused')
        nb=[[0] for _ in range(c.n)]; share=np.ones((c.n,c.n),dtype=bool)
        s,_=observe(c,[case],1,nb,share,8)
        self.assertEqual(s['success'][1,0],0.)
        s,_=observe(c,[case],6,nb,share,8)
        self.assertEqual(s['success'][1,0],1.)
        self.assertEqual(s['activity'][1,0],0.)

    def test_bad_horizon_or_endogenous_arrival_rejected(self):
        with self.assertRaises(ValueError): simulate_channels(self.config(),cohort_schedule=CohortSchedule())
        with self.assertRaises(ValueError): simulate_channels(replace(self.config(),new_dispute_rate=.1),cohort_schedule=self.schedule())

    def test_first_choice_denominator(self):
        ca=pd.DataFrame(dict(first_submitted=[0,0,1,np.nan],initial_choice=['d','e','d+e','']))
        tr=first_submission_series(ca,3)
        self.assertEqual(tr.loc[0,'first_d_share'],.5)
        self.assertEqual(tr.loc[1,'first_both_share'],1.)
        self.assertTrue(np.isnan(tr.loc[2,'first_e_share']))
