import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.channel_choice import (ChannelConfig, CASE_COLUMNS, world, observe,
    choose_additions, joint_success, simulate_channels)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.c=ChannelConfig(n=12,firms=3,degree=4,steps=12)
        self.nb=[[1]]+[[] for _ in range(11)]
        self.sharing=np.ones((12,12),dtype=bool)

    def case(self,status='submitted',finished=None):
        case={key:None for key in CASE_COLUMNS}
        case.update(employee=1,d_submitted=0,d_status=status,d_finished=finished,
                    e_status='unused',reachable=True)
        return case

    def test_old_pending_remains_visible_without_false_outcome(self):
        signals,_=observe(self.c,[self.case()],8,self.nb,self.sharing)
        self.assertEqual(signals['activity'][0,0],1)
        self.assertEqual(signals['recent'][0,0],0)
        self.assertEqual(signals['success'].sum()+signals['failure'].sum(),0)
        old,_=observe(replace(self.c,observation_mode='recent'),[self.case()],8,self.nb,self.sharing)
        self.assertEqual(old['activity'][0,0],0)

    def test_end_outcomes_next_round_expire_and_cancel_is_not_failure(self):
        for status,label in [('successful','success'),('closed_unresolved','failure')]:
            case=self.case(status,5)
            for t,expected in [(5,0),(6,1),(8,1),(9,0)]:
                signals,_=observe(self.c,[case],t,self.nb,self.sharing)
                self.assertEqual(signals[label][0,0],expected)
        signals,_=observe(self.c,[self.case('cancelled_by_resolution',5)],6,self.nb,self.sharing)
        self.assertEqual(signals['failure'].sum()+signals['success'].sum(),0)

    def test_sharing_and_isolates(self):
        signals,_=observe(self.c,[self.case()],8,self.nb,np.zeros((12,12),bool))
        for value in signals.values(): self.assertEqual(value.sum(),0)

    def test_same_round_submission_invisible(self):
        signals,_=observe(self.c,[self.case()],0,self.nb,self.sharing)
        for value in signals.values(): self.assertEqual(value.sum(),0)

    def test_unique_people_in_window(self):
        signals,_=observe(self.c,[self.case('successful',5),self.case('successful',4)],6,self.nb,self.sharing)
        self.assertEqual(signals['success'][0,0],1)

    def test_objective_does_not_change_initial_belief(self):
        a=world(replace(self.c,domestic_success=.15),7)
        b=world(replace(self.c,domestic_success=.95),7)
        np.testing.assert_array_equal(a['beliefs'],b['beliefs'])
        z=world(replace(self.c,domestic_signal=.95),7)
        self.assertTrue((z['beliefs'][:,0]>=a['beliefs'][:,0]).all())
        for k in a['draws']: np.testing.assert_array_equal(a['draws'][k],z['draws'][k])

    def test_dependence_mean_and_dispersion_separate(self):
        low=world(replace(self.c,dependence_mean=.15),8)['dependence']
        high=world(replace(self.c,dependence_mean=.85),8)['dependence']
        self.assertAlmostEqual(low.mean(),.15)
        self.assertAlmostEqual(high.mean(),.85)
        np.testing.assert_allclose(high-low,.7)
        self.assertAlmostEqual(low.std(),high.std())
        with self.assertRaises(ValueError): replace(self.c,dependence_mean=.05,dependence_spread=.1)

    def test_learning_directions_and_no_repeated_accumulation(self):
        w=world(self.c,3); case=self.case('unused'); case['d_submitted']=None
        _,base=choose_additions(self.c,w,case,6,[0,0],np.zeros(2),np.zeros(2))
        _,good=choose_additions(self.c,w,case,6,[0,0],np.ones(2),np.zeros(2))
        _,bad=choose_additions(self.c,w,case,6,[0,0],np.zeros(2),np.ones(2))
        self.assertGreater(good['p_d'],base['p_d'])
        self.assertLess(bad['p_d'],base['p_d'])
        _,again=choose_additions(self.c,w,case,7,[0,0],np.ones(2),np.zeros(2))
        self.assertEqual(good['p_d'],again['p_d'])

    def test_actual_delay_and_expected_wait_separate(self):
        a=simulate_channels(self.c,2)['decisions']
        b=simulate_channels(replace(self.c,domestic_delay=8),2)['decisions']
        pd.testing.assert_frame_equal(a[a.time==0],b[b.time==0])
        w=world(self.c,2); case=self.case('unused'); case['d_submitted']=None
        _,fast=choose_additions(self.c,w,case,0,[0,0])
        _,slow=choose_additions(replace(self.c,expected_domestic_wait=9),w,case,0,[0,0])
        self.assertNotEqual(fast['utilities'],slow['utilities'])

    def test_disable_escalation(self):
        r=simulate_channels(replace(self.c,allow_escalation=False),8)
        self.assertEqual(r['trajectory'].new_additional_channel.sum(),0)

    def test_joint_probability_bounds(self):
        self.assertAlmostEqual(joint_success([.4,.6],0),.76)
        self.assertAlmostEqual(joint_success([.4,.6],1),.6)
        for rho in [0,.3,1]:
            self.assertTrue(.6<=joint_success([.4,.6],rho)<=.76)

    def test_actual_pending_simulation_survives_window(self):
        c=replace(self.c,initial_dispute=1,domestic_cost=0,domestic_prior=1,
                  belief_spread=0,customer_enabled=False,firm_response_enabled=False,
                  domestic_delay=100,domestic_implementation_enabled=False)
        r=simulate_channels(c,3)
        self.assertEqual(r['trajectory'].iloc[-1].visible_domestic,c.n)
        self.assertEqual(r['decisions'].failure_d.sum(),0)


if __name__=='__main__': unittest.main()
