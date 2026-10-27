import unittest
from types import SimpleNamespace
import torch
from diffusion_policy.policy.strict_eggplant_metric import components
class Tests(unittest.TestCase):
 def setUp(self):
  self.v=SimpleNamespace(horizon=8,ee_key='ee',_consts=lambda a:(torch.eye(3),torch.ones(3),None,None))
  self.obs={'ee':torch.zeros(2,1,3),'strict_task_state':torch.tensor([[0,0,0,.2,0,0,1,.2,.01,0,1.]]).repeat(2,1)}
 def test_held_closed_preferred_to_premature_open(self):
  a=torch.zeros(2,16,7);a[:,:8,0]=.01;a[1,:,6]=1
  c=components(self.v,self.obs,a)
  self.assertGreater(c['task_score'][0],c['task_score'][1]);self.assertEqual(c['task_reopen_penalty'][1],1)
  self.assertTrue(torch.all(c['task_score'].abs()<=1))
  self.obs['strict_task_state'][:,9]=1
  self.assertTrue(torch.all(components(self.v,self.obs,a)['task_score']==0))
 def test_approach_before_grasp_and_ignore_tail(self):
  self.obs['strict_task_state'][:,0]=.2;self.obs['strict_task_state'][:,6]=0;self.obs['strict_task_state'][:,10]=0
  a=torch.zeros(2,16,7);a[0,:8,0]=.01;a[1,:8,0]=-.01
  c=components(self.v,self.obs,a)['task_score'];self.assertGreater(c[0],c[1])
  a[:,8:]=100;torch.testing.assert_close(c,components(self.v,self.obs,a)['task_score'])
if __name__=='__main__':unittest.main()
