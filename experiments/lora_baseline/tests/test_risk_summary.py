import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_risk_suite import Scores, bootstrap
from metrics import paired_bootstrap, weighted_auroc, weighted_brier


class RiskSummaryTests(unittest.TestCase):
    def test_weighted_ties_and_duplicate_patient_draws(self):
        y=np.array([0,1,0,1,0,1]); p=np.array([.2,.2,.4,.8,.8,.9])
        w=np.array([1.,4.,2.,7.,3.,5.]); counts=np.array([3,2,0,1,2,4])
        mass=w*counts; pos=y==1; neg=~pos
        pair=(p[pos,None]>p[None,neg])+.5*(p[pos,None]==p[None,neg])
        expected=np.sum(pair*mass[pos,None]*mass[None,neg])/(mass[pos].sum()*mass[neg].sum())
        actual=Scores(y,p,w).metrics(counts)
        self.assertAlmostEqual(actual[0],expected,places=14)
        self.assertAlmostEqual(actual[1],np.average((p-y)**2,weights=mass),places=14)

    def test_same_patient_draws_reproduce_reference_bootstrap(self):
        rng=np.random.default_rng(7); y=np.array([0,1]*20)
        p=np.round(rng.random(40),1); b=np.round(rng.random(40),1); w=rng.uniform(.5,4,40)
        _,result=bootstrap(y,[p,b],w,3,100)
        for key,fn in [('auroc',weighted_auroc),('brier',weighted_brier)]:
            old=paired_bootstrap(lambda y,p,b,w:fn(y,p,w)-fn(y,b,w),[y,p,b,w],100,3)
            for field in ('point','ci_low','ci_high'):
                self.assertAlmostEqual(result[0][key][field],old[field],places=12)


if __name__=='__main__':
    unittest.main()
