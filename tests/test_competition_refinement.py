import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from src.channel_choice import ChannelConfig,world,choose_additions,simulate_channels
from src.wait_learning import expected_waits


def case():
    return dict(employee=0,material_at=None,reachable=True,d_status='unused',e_status='unused',
        d_submitted=None,e_submitted=None,d_finished=None,e_finished=None)


class RefinementTests(unittest.TestCase):
    def test_exact_ties_not_fixed_to_domestic(self):
        c=ChannelConfig(tie_break='random',domestic_cost=.1,customer_cost=.1,subjective_route_overlap=1.)
        choices=[]
        for seed in range(100):
            w=world(c,seed); w['beliefs'][:]=.5; w['costs'][:]=.1
            choices.append(choose_additions(c,w,case(),0,[0,0])[0])
        self.assertEqual(set(choices),{(0,),(1,)})
        self.assertTrue(25<choices.count((0,))<75)

    def test_legacy_tie(self):
        c=ChannelConfig(subjective_route_overlap=1.); w=world(c,0)
        w['beliefs'][:]=.5; w['costs'][:]=.1
        self.assertEqual(choose_additions(c,w,case(),0,[0,0])[0],(0,))

    def test_pending_and_visibility(self):
        c=ChannelConfig(n=4,firms=1,degree=2,wait_learning_weight=1.,wait_prior_strength=1.)
        ca=case(); ca.update(d_submitted=0,d_status='submitted')
        sharing=np.ones((4,4),dtype=bool); nb=[[1],[0],[],[]]
        x=expected_waits(c,[ca],9,nb,sharing,3)
        np.testing.assert_allclose(x[:,0],[6,6,3,3])
        # No information from a submission in the decision round.
        np.testing.assert_allclose(expected_waits(c,[ca],0,nb,sharing,3),3.)

    def test_cancellation_and_old_completion_excluded(self):
        c=ChannelConfig(n=4,firms=1,degree=2,wait_learning_weight=1.)
        ca=case(); ca.update(d_submitted=0,d_finished=2,d_status='cancelled_by_resolution')
        nb=[[1],[0],[],[]]; sh=np.ones((4,4),bool)
        np.testing.assert_allclose(expected_waits(c,[ca],3,nb,sh,3),3.)
        ca['d_status']='closed_unresolved'
        np.testing.assert_allclose(expected_waits(c,[ca],8,nb,sh,3),3.)

    def test_completed_duration_not_repeated_accumulation(self):
        c=ChannelConfig(n=4,firms=1,degree=2,wait_learning_weight=1.,wait_prior_strength=1.)
        ca=case(); ca.update(d_submitted=0,d_finished=7,d_status='closed_unresolved')
        nb=[[],[],[],[]]; sh=np.ones((4,4),bool)
        a=expected_waits(c,[ca],8,nb,sh,3); b=expected_waits(c,[ca],9,nb,sh,3)
        self.assertEqual(a[0,0],5.); np.testing.assert_array_equal(a,b)

    def test_reproducible_learning_and_ties(self):
        c=ChannelConfig(n=12,firms=2,steps=12,degree=4,tie_break='random',wait_learning_weight=.5)
        a=simulate_channels(c,4); b=simulate_channels(c,4)
        pd.testing.assert_frame_equal(a['cases'],b['cases'])
        pd.testing.assert_frame_equal(a['decisions'],b['decisions'])

    def test_invalid_config(self):
        for args in [dict(tie_break='x'),dict(wait_learning_weight=2.),dict(wait_prior_strength=0.)]:
            with self.assertRaises(ValueError): ChannelConfig(**args)

    def test_zero_learning_ignores_wait_prior(self):
        c=ChannelConfig(n=12,firms=2,steps=10,degree=4)
        pd.testing.assert_frame_equal(simulate_channels(c,3)['cases'],simulate_channels(replace(c,wait_prior_strength=20.),3)['cases'])


if __name__=='__main__': unittest.main()
