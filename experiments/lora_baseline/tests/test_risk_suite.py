import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_risk_suite import plan, selected_epoch


class RiskSuiteTests(unittest.TestCase):
    def test_order_preserves_all_controls_and_seed_one_boundary(self):
        jobs = plan()
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({j['name'] for j in jobs}), 18)
        self.assertTrue(all(j['group']=='C' for j in jobs[:9]))
        self.assertEqual({j['seed'] for j in jobs[9:12]}, {1})
        self.assertEqual({j['disease'] for j in jobs[9:12]}, {'ckd','fatty_liver','diabetes'})
        self.assertEqual([j['seed'] for j in jobs[12:]], [2,2,2,3,3,3])

    def test_selection_ties_and_nonfinite_are_not_accuracy_thresholds(self):
        rows = [dict(epoch=i,train_weighted_bce=1.,validation_weighted_bce=v,
                     validation_weighted_auroc=.4,validation_weighted_brier=.6)
                for i,v in enumerate([.5,.3,.3],1)]
        self.assertEqual(selected_epoch(rows),2)
        rows[2]['validation_weighted_auroc']=float('nan')
        with self.assertRaises(ValueError):
            selected_epoch(rows)
        with self.assertRaises(ValueError):
            selected_epoch([])


if __name__=='__main__':
    unittest.main()
