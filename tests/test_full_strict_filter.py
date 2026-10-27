import sys,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scriptsv2/eval_search'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scriptsv2/data_collect'))
from full_filter_lookahead import evaluate_filter
from filter_first_try_grasp import grasp_metrics
from filter_strict_distance_success import distance_metrics
class Tests(unittest.TestCase):
 def test_exact_retrospective_metrics_match_dataset_code(self):
  d=np.array([.2,.2,.18,.1,.05]);g=np.array([.1,.02,.021,.023,.024]);c=np.array([0,1,1,0,1]);o=np.array([1,0,0,0,0])
  h=[dict(deb=float(a),dge=float(b),contact=bool(e),open=bool(f),success=i==4) for i,(a,b,e,f) in enumerate(zip(d,g,c,o))]
  x=evaluate_filter(h);json.dumps(x);m=grasp_metrics(o,c,g,d);n=distance_metrics(d)
  self.assertTrue(x['passed']);self.assertAlmostEqual(x['backtrack_m'],n['backtrack'])
  self.assertEqual(x['max_contact_gap'],m['max_contact_gap']);self.assertAlmostEqual(x['separation_rise_m'],m['dge_rise'])
  h[-1]['deb']=.25;self.assertFalse(evaluate_filter(h)['checks']['bounded_backtrack'])
 def test_missing_grasp_and_nonfinite_fail(self):
  h=[dict(deb=.2,dge=.1,contact=False,open=True,success=False)]
  self.assertFalse(evaluate_filter(h)['passed']);h[0]['deb']=float('nan')
  self.assertEqual(evaluate_filter(h)['score'],-1e6)
if __name__=='__main__':unittest.main()
