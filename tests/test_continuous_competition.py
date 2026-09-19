import unittest
from dataclasses import replace
import pandas as pd
import numpy as np
from src.channel_choice import ChannelConfig,simulate_channels,world
from src.continuous_competition import ContinuousCompetition,audit_continuous,window_metrics


class ContinuousTests(unittest.TestCase):
    def config(self):return ChannelConfig(n=12,firms=2,degree=4,steps=30,initial_dispute=0.,new_dispute_rate=0.)

    def test_no_arrival_no_complaint(self):
        r=simulate_channels(self.config(),dynamics=ContinuousCompetition(arrival_rate=0.))
        audit_continuous(r);self.assertEqual(len(r['cases']),0)

    def test_retirement_not_resolution_and_new_ids(self):
        c=replace(self.config(),domestic_success=0.,customer_review_base=0.,customer_review_merit=0.,evidence_help=0.)
        r=simulate_channels(c,2,dynamics=ContinuousCompetition(arrival_rate=1.,active_horizon=5,feedback=False))
        audit_continuous(r);self.assertTrue(r['cases'].resolved.isna().all())
        self.assertGreater(len(r['cases']),c.n);self.assertTrue(r['cases'].case_id.is_unique)
        self.assertGreater(len(r['retirements']),0)

    def test_unknown_cannot_use_customer(self):
        r=simulate_channels(self.config(),dynamics=ContinuousCompetition(arrival_rate=.5,initial_knowledge=0.,rho=0.))
        self.assertTrue(r['cases'].e_submitted.isna().all())

    def test_feedback_off_and_reproducibility(self):
        c=self.config();d=ContinuousCompetition(arrival_rate=.2,feedback=False)
        a=simulate_channels(c,4,dynamics=replace(d));b=simulate_channels(c,4,dynamics=replace(d))
        audit_continuous(a);pd.testing.assert_frame_equal(a['cases'],b['cases'])
        self.assertTrue(a['dynamics'].prevented.eq(0).all())
        self.assertTrue(a['dynamics'].mean_prevention.eq(0).all())

    def test_potential_stream_independent_of_feedback(self):
        c=self.config();a=ContinuousCompetition(feedback=False);b=ContinuousCompetition(feedback=True)
        a.start(c,world(c,0),0);b.start(c,world(c,0),0)
        np.testing.assert_array_equal(a.potential,b.potential)

    def test_shock_does_not_mutate_input(self):
        c=self.config();p=c.customer_review_base
        simulate_channels(c,0,dynamics=ContinuousCompetition(shock='objective',shock_time=5))
        self.assertEqual(c.customer_review_base,p)

    def test_knowledge_no_same_round_cascade(self):
        c=self.config();w=world(c,0);d=ContinuousCompetition(rho=1.,feedback=False,arrival_rate=0.)
        d.start(c,w,0);d.known[:]=False;d.known[0]=True
        nb=[[] for _ in range(c.n)];nb[1]=[0];nb[2]=[1];w['sharing'][:]=True
        d.advance(0,c,w,[],{},nb)
        self.assertTrue(d.known[1]);self.assertFalse(d.known[2])

    def test_only_past_intervention_drives_prevention(self):
        c=self.config();w=world(c,0);d=ContinuousCompetition(arrival_rate=0.,prevention_cost=0.)
        d.start(c,w,0);nb=[[] for _ in range(c.n)]
        cases=[dict(customer_intervened_at=0,firm=0)]
        d.advance(0,c,w,cases,{},nb);self.assertEqual(d.effort[0],0.)
        d.advance(1,c,w,cases,{},nb);self.assertGreater(d.effort[0],0.)

    def test_window_censors_late_additions(self):
        ca=pd.DataFrame(dict(employee=[0,0],created=[0,10],d_submitted=[0,10],e_submitted=[5,12],resolved=[6,13]))
        m=window_metrics(ca,0,20,4)
        self.assertEqual(m['cases'],2);self.assertEqual(m['employees'],1)
        self.assertEqual(m['customer_use'],.5);self.assertEqual(m['resolved'],.5)


if __name__=='__main__':unittest.main()
