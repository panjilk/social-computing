import unittest
from dataclasses import asdict
import pandas as pd
from src.channel_choice import ChannelConfig, simulate_channels
from experiments.competition import design, measures


class CompetitionTests(unittest.TestCase):
    def test_only_availability_differs(self):
        specs=design(ChannelConfig(material_rate=1,new_dispute_rate=.5,prevention=.5))
        configs=[]
        for c in specs.values():
            d=asdict(c); d.pop('domestic_enabled'); d.pop('customer_enabled'); configs.append(d)
            for key in ['material_rate','new_dispute_rate','prevention','evidence_help','perceived_evidence_bonus','domestic_information']:
                self.assertEqual(d[key],0)
        self.assertEqual(configs[0],configs[1]); self.assertEqual(configs[1],configs[2])

    def test_first_choice_partition_and_direction_accounting(self):
        specs=design(ChannelConfig(n=24,firms=3,degree=4,steps=15))
        for name,c in specs.items():
            r=simulate_channels(c,2); m,tr=measures(r)
            self.assertAlmostEqual(sum(m[k] for k in ['first_d_all','first_e_all','first_both_all','never_submitted']),1)
            self.assertAlmostEqual(m['both_use'],m['first_both_all']+m['d_to_e_all']+m['e_to_d_all'])
            self.assertEqual(tr.new_d_to_e.sum(),m['d_to_e_count'])
            self.assertEqual(tr.new_e_to_d.sum(),m['e_to_d_count'])
            self.assertEqual(tr.total_cases.nunique(),1)
            if name!='both': self.assertEqual(m['both_use'],0)

    def test_reverse_direction_recorded(self):
        # A hand-constructed two-case record: one D->E and one E->D.
        r=simulate_channels(ChannelConfig(n=4,firms=1,degree=2,steps=8,initial_dispute=1),0)
        ca=r['cases'].iloc[:2].copy()
        ca['initial_choice']=['d','e']; ca['first_submitted']=[0,0]
        ca['d_submitted']=[0,3]; ca['e_submitted']=[2,0]
        r['cases']=ca
        m,tr=measures(r)
        self.assertEqual(m['d_to_e_all'],.5); self.assertEqual(m['e_to_d_all'],.5)
        self.assertEqual(m['d_to_e_conditional'],1); self.assertEqual(m['e_to_d_conditional'],1)
        self.assertEqual(tr.loc[tr.time==2,'new_d_to_e'].iloc[0],1)
        self.assertEqual(tr.loc[tr.time==3,'new_e_to_d'].iloc[0],1)

    def test_reproducibility(self):
        c=design(ChannelConfig(n=12,firms=2,degree=4,steps=8))['both']
        a,ta=measures(simulate_channels(c,6)); b,tb=measures(simulate_channels(c,6))
        pd.testing.assert_series_equal(pd.Series(a),pd.Series(b))
        pd.testing.assert_frame_equal(ta,tb)
