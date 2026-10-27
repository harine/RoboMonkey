import tempfile,unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from diffusion_policy.env.simpler.veritas.subtask_two_controller import TwoSubtaskController

class Tests(unittest.TestCase):
    def test_physical_grasp_advances_even_with_unfinished_pixel_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=TwoSubtaskController(tmp,tmp,1,'stack',None)
            def plan(frame,t,tcp,reason,checks):
                c.current=SimpleNamespace(episode=SimpleNamespace(done=False,progress=0),update=lambda tcp:np.zeros(13))
                c.last_request=t;c.last_progress_t=t;c.plan_calls+=1
            c._plan=Mock(side_effect=plan)
            c.observe(None,0,[0,0,.05],[0,0,.03],[.1,0,.03],True,True)
            c.observe(None,8,[0,0,.08],[0,0,.06],[.1,0,.03],False,True)
            self.assertEqual(c.stage,1)
            self.assertEqual(c.plan_calls,2)
            self.assertFalse(c.events[0]['pixel_plan_done'])
            for t in [16,17,18]:
                c.observe(None,t,[.1,0,.13],[.1,0,.06],[.1,0,.03],True,True)
            self.assertTrue(c.complete)
            self.assertEqual(c.plan_calls,2)
    def test_closed_gripper_or_lift_alone_not_completion(self):
        checks=dict(held=False,source_lift_m=.1,valid_by_stage=[True,False,False,False],tcp_source_distance_m=.1)
        self.assertFalse(TwoSubtaskController.physical_complete(0,checks))
        self.assertFalse(TwoSubtaskController.physical_complete(1,checks))
if __name__=='__main__':unittest.main()
