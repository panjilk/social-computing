import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.channel_choice import ChannelConfig,world,simulate_channels,review_probability


class ChannelChoiceTests(unittest.TestCase):
    def config(self,**kwargs):
        return ChannelConfig(**{**dict(n=12,firms=2,degree=4,steps=12),**kwargs})

    def forced(self,**kwargs):
        c=self.config(initial_dispute=1,customer_availability=1,employee_reachability=1,
            domestic_cost=0,customer_cost=0,multi_extra_cost=0,wait_discount=0,
            domestic_prior=.8,customer_prior=.8,belief_spread=0,domestic_information=0,
            perceived_dependence=0,**kwargs)
        w=world(c,0)
        w['merit'][:]=1
        for draws in w['draws'].values():
            draws[:]=0
        return c,w

    def test_no_disputes(self):
        r=simulate_channels(self.config(initial_dispute=0))
        self.assertTrue(r['cases'].empty)
        self.assertEqual(r['trajectory'].new_complaints.sum(),0)

    def test_no_firm_response_no_resolution(self):
        c,w=self.forced(firm_response_enabled=False,domestic_implementation_enabled=False)
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        self.assertTrue(r['cases'].resolved.isna().all())
        self.assertEqual(r['cases'].compensation.sum(),0)

    def test_both_routes_single_case_single_capacity_single_payment(self):
        c,w=self.forced(domestic_delay=0,customer_delay=0,firm_delay=0,firm_capacity=2,
                        domestic_implementation_enabled=False)
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        cases=r['cases']; events=r['events']; tr=r['trajectory']
        self.assertEqual(len(cases),c.n)
        self.assertEqual(tr.new_complaints.sum(),c.n)
        self.assertTrue((cases.initial_choice=='d+e').all())
        self.assertEqual(cases.compensation.sum(),c.n)
        self.assertTrue(events[events.event=='case_resolved'].case_id.is_unique)
        self.assertTrue((tr.new_resolved<=c.firms*c.firm_capacity).all())
        self.assertEqual(len(events[events.event=='firm_case_processed']),c.n)

    def test_no_customer_access_never_external(self):
        for kwargs in ({'customer_availability':0},{'employee_reachability':0},{'customer_enabled':False}):
            r=simulate_channels(self.config(**kwargs))
            self.assertTrue(r['cases'].e_submitted.isna().all())

    def test_material_not_resolution(self):
        c,w=self.forced(domestic_success=0,customer_enabled=False,material_rate=1)
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        self.assertTrue(r['cases'].material_at.notna().all())
        self.assertTrue(r['cases'].resolved.isna().all())
        self.assertTrue((r['cases'].d_status=='closed_unresolved').all())

    def test_same_round_material_not_used_until_next_round(self):
        c,w=self.forced(domestic_delay=0,customer_delay=0,firm_delay=0,
            domestic_success=0,material_rate=1,customer_review_base=0,customer_review_merit=0,evidence_help=1)
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        self.assertTrue(r['cases'].customer_intervened_at.isna().all())
        c.customer_delay=1
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        self.assertTrue(r['cases'].customer_intervened_at.notna().all())

    def test_wait_is_not_customer_failure(self):
        c,w=self.forced(customer_delay=20,domestic_enabled=False)
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        self.assertTrue((r['cases'].e_status=='submitted').all())
        self.assertFalse((r['events'].event=='customer_declined').any())

    def test_domestic_then_add_customer_same_id(self):
        c,w=self.forced(domestic_delay=0,firm_delay=0,domestic_success=0,escalation_wait=2)
        w['beliefs'][:]=[.9,.8]; w['costs'][:]=[.1,.3]
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        self.assertTrue((r['cases'].initial_choice=='d').all())
        self.assertTrue((r['cases'].e_submitted==2).all())
        self.assertEqual(len(r['cases']),c.n)
        self.assertEqual(r['trajectory'].new_complaints.sum(),c.n)
        self.assertEqual(r['trajectory'].new_additional_channel.sum(),c.n)

    def test_accounting_recurrence_and_route_tracking(self):
        r=simulate_channels(self.config(initial_dispute=1,new_dispute_rate=.5))
        tr=r['trajectory']; ca=r['cases']; ev=r['events']
        self.assertTrue((tr.total_cases==tr.unresolved+tr.cumulative_resolved).all())
        self.assertTrue((tr.unresolved==tr.unsubmitted+tr.pending+tr.closed_unresolved).all())
        self.assertTrue(ca.case_id.is_unique)
        self.assertTrue(ev[ev.event=='case_resolved'].case_id.is_unique)
        self.assertEqual(tr.new_complaints.sum(),ca.first_submitted.notna().sum())
        self.assertTrue(ca[ca.first_submitted.notna()].status.isin(['pending','closed_unresolved','resolved']).all())
        for key,event in [('d','domestic_received'),('e','customer_received')]:
            self.assertEqual(ca[key+'_submitted'].notna().sum(),(ev.event==event).sum())
            self.assertTrue(ev[ev.event==event].case_id.is_unique)

    def test_reproducibility(self):
        a=simulate_channels(self.config(),17); b=simulate_channels(self.config(),17)
        for key in a:
            if key!='graph': pd.testing.assert_frame_equal(a[key],b[key])

    def test_horizon_prefix_stable(self):
        a=simulate_channels(self.config(steps=8),3)
        b=simulate_channels(self.config(steps=18),3)
        pd.testing.assert_frame_equal(a['trajectory'],b['trajectory'].iloc[:8].reset_index(drop=True))
        pd.testing.assert_frame_equal(a['events'],b['events'][b['events'].time<8].reset_index(drop=True))

    def test_ablation_keeps_population_and_draws(self):
        a=world(self.config(evidence_help=0),4); b=world(self.config(evidence_help=.3),4)
        for key in ('firm','initial','has_customer','dependence','reachable','merit','beliefs','costs'):
            np.testing.assert_array_equal(a[key],b[key])
        for key in a['draws']:
            np.testing.assert_array_equal(a['draws'][key],b['draws'][key])

    def test_evidence_help_off_has_no_objective_effect(self):
        c=self.config(evidence_help=0)
        self.assertEqual(review_probability(c,.5,False),review_probability(c,.5,True))

    def test_invalid_config(self):
        for kwargs in ({'firm_response_enabled':'false'},{'domestic_success':2},
                       {'escalation_wait':-1},{'domestic_cost':-1},{'visibility_window':0}):
            with self.assertRaises(ValueError): self.config(**kwargs)

    def test_domestic_resolution_not_misattributed_to_customer(self):
        from experiments.channel_experiments import endpoint
        c,w=self.forced(domestic_delay=0,customer_delay=10,firm_delay=0)
        with patch('src.channel_choice.world',return_value=w): r=simulate_channels(c)
        values=endpoint(r)
        self.assertEqual(values['external_path_trigger_rate'],0.)
        self.assertEqual(values['external_user_case_resolution_rate'],1.)
        self.assertTrue((r['cases'].e_status=='cancelled_by_resolution').all())

    def test_named_design_overrides_response_switches(self):
        from experiments.channel_experiments import design
        scenarios=design(self.config(domestic_enabled=False,customer_enabled=False,firm_response_enabled=False))
        central=scenarios['domestic_0.55_help_on']
        self.assertTrue(central.domestic_enabled and central.customer_enabled and central.firm_response_enabled)
        self.assertFalse(scenarios['no_firm_response'].firm_response_enabled)


if __name__=='__main__': unittest.main()
