import unittest
from dataclasses import replace
import pandas as pd
from src.channel_choice import ChannelConfig,simulate_channels
from src.customer_network import CustomerNetwork
from experiments.channel_experiments import audit


class CustomerNetworkTests(unittest.TestCase):
    def config(self):
        return ChannelConfig(n=24,firms=6,steps=15,customer_availability=1.,employee_reachability=1.)

    def test_pooled_matches_original_and_topology_irrelevant(self):
        c=self.config(); base=simulate_channels(c,2)
        for k in (1,3,6):
            r=simulate_channels(c,2,customer_network=CustomerNetwork(6,k,True))
            pd.testing.assert_frame_equal(base['cases'],r['cases'].drop(columns='customer'))
            pd.testing.assert_frame_equal(base['trajectory'],r['trajectory'])

    def test_capacity_accounting_reproducibility(self):
        c=self.config()
        r=simulate_channels(c,3,customer_network=CustomerNetwork(6,1))
        s=simulate_channels(c,3,customer_network=CustomerNetwork(6,1))
        audit(r)
        pd.testing.assert_frame_equal(r['cases'],s['cases'])
        q=r['customer_queues']
        self.assertTrue((q.reviewed<=2).all())
        self.assertTrue((q.eligible==q.reviewed+q.eligible_left).all())
        self.assertTrue((q.groupby('time').nominal_capacity.sum()==12).all())
        self.assertEqual(r['customer_edges'].customer.nunique(),1)

    def test_domestic_only_unchanged(self):
        c=replace(self.config(),customer_enabled=False)
        a=simulate_channels(c,1,customer_network=CustomerNetwork(6,6))
        b=simulate_channels(c,1,customer_network=CustomerNetwork(6,1))
        pd.testing.assert_frame_equal(a['trajectory'],b['trajectory'])

    def test_no_cases(self):
        c=replace(self.config(),initial_dispute=0.)
        r=simulate_channels(c,0,customer_network=CustomerNetwork())
        audit(r)
        self.assertEqual(r['customer_queues'].reviewed.sum(),0)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            simulate_channels(replace(self.config(),customer_capacity=11),customer_network=CustomerNetwork())

    def test_fifo_local(self):
        from src.channel_choice import world
        c=self.config(); net=CustomerNetwork(6,1); net.start(c,world(c,0),0)
        queue=[dict(case_id=str(i),firm=i,e_submitted=i) for i in range(4)]
        self.assertEqual([x['case_id'] for x in net.select(queue,5)],['0','1'])
