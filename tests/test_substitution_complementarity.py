import unittest
from dataclasses import replace
import numpy as np
from src.channel_choice import ChannelConfig,simulate_channels
from src.cohort_trends import CohortSchedule
from experiments.substitution_complementarity import measures


class MaterialTests(unittest.TestCase):
    def config(self):
        return ChannelConfig(n=24,firms=2,degree=4,steps=15,initial_dispute=1.,
            domestic_cost=0.,customer_enabled=False,material_rate=1.,domestic_success=0.,
            material_requires_support=True)

    def test_no_support_no_supporting_material(self):
        r=simulate_channels(self.config(),0)
        self.assertTrue(r['cases'].material_at.isna().all())

    def test_old_material_rule_remains_available(self):
        r=simulate_channels(replace(self.config(),material_requires_support=False),0)
        self.assertTrue(r['cases'].material_at.notna().any())

    def test_material_does_not_resolve(self):
        r=simulate_channels(replace(self.config(),domestic_success=1.,domestic_implementation_enabled=False),0)
        self.assertTrue(r['cases'].material_at.notna().any())
        self.assertTrue(r['cases'].resolved.isna().all())

    def test_matched_world_unchanged_when_material_inactive(self):
        c=replace(self.config(),material_rate=0.)
        a=simulate_channels(c,0); b=simulate_channels(replace(c,material_requires_support=False),0)
        import pandas as pd
        pd.testing.assert_frame_equal(a['cases'],b['cases'])

    def test_material_can_enable_customer_resolution_when_timely(self):
        c=replace(self.config(),customer_enabled=True,domestic_success=1.,
            domestic_implementation_enabled=False,customer_cost=0.,multi_extra_cost=0.,
            customer_availability=1.,employee_reachability=1.,customer_delay=4,
            customer_review_base=0.,customer_review_merit=0.,evidence_help=1.,
            firm_customer_base=1.,dependence_effect=0.)
        helped=simulate_channels(c,0); control=simulate_channels(replace(c,evidence_help=0.),0)
        self.assertTrue(helped['cases'].resolved.notna().any())
        self.assertTrue(control['cases'].resolved.isna().all())
        used=helped['cases'][helped['cases'].resolved.notna()]
        self.assertTrue(used.material_at.lt(used.e_responded).all())

    def test_measure_denominators_and_no_material_score(self):
        c=replace(self.config(),steps=22,customer_enabled=True,customer_cost=.1,
            customer_availability=1.,employee_reachability=1.,domestic_success=.5)
        s=CohortSchedule(cohorts=4,spacing=3,followup=13,preparation_max=1)
        r=simulate_channels(c,1,cohort_schedule=s); m=measures(r['cases'],s,c)
        np.testing.assert_allclose(m.resolved+m.unresolved,1.)
        np.testing.assert_allclose(m.customer_route_success,m.customer_only_resolution+m.joint_resolution)
        np.testing.assert_allclose(m.customer_user_any_resolution+m.customer_unresolved,1.)
        self.assertTrue((m.mechanical_material_increment>=0).all())


if __name__=='__main__': unittest.main()
