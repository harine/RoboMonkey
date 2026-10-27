import unittest
from diffusion_policy.env.simpler.veritas.subtask_eggplant_controller import BasketPositionChecks,EggplantPlanProvider
class Tests(unittest.TestCase):
 def test_basket_release_requires_inside_height_and_stability(self):
  c=BasketPositionChecks([.1,.04,.04],[.12,.16,.01])
  for _ in range(3):d=c.update([0,0,.2],[0,0,.025],[0,0,0],True)
  self.assertTrue(d['valid_by_stage'][3])
  self.assertFalse(c.update([0,0,.2],[.1,0,.025],[0,0,0],True)['valid_by_stage'][3])
  self.assertFalse(c.update([0,0,.2],[0,0,.025],[0,0,0],False)['valid_by_stage'][3])
 def test_prompt_has_correct_objects(self):
  p=object.__new__(EggplantPlanProvider);p.stage=1;p.feedback={}
  s=p.make_prompt('put eggplant in basket',640,480)
  self.assertIn('lower into its interior',s);self.assertNotIn('cube',s);self.assertNotIn('stack height',s)
if __name__=='__main__':unittest.main()
