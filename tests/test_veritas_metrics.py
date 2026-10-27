import unittest
import json
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
import torch
from diffusion_policy.policy.veritas_metric_verifier import MetricTracker,VeritasMetricVerifier

CAM=Path(__file__).resolve().parents[1]/'diffusion_policy/diffusion_policy/config/veritas/camera_widowx_stack_cube.json'
def plan():
    return NS(grasp_radius_m=.04,lift_clearance_m=.06,goal_xy_tolerance_m=.03,
        goal_height_offset_m=.03,grasp_offset_z_m=0,
        stages=[NS(uv=[320,240],gripper='open' if i in (0,5,6) else 'closed') for i in range(7)])
class Tests(unittest.TestCase):
    def test_close_without_object_motion_not_grasp(self):
        tr=MetricTracker(plan())
        for z in (.93,.95,.97):tr.update([0,0,z],[0,0,.91],[.2,0,.91],False)
        self.assertFalse(tr.held)
    def test_lift_and_coupling_then_drop(self):
        tr=MetricTracker(plan());tr.update([0,0,.91],[0,0,.91],[.2,0,.91],False)
        tr.update([0,0,.94],[0,0,.94],[.2,0,.91],False)
        self.assertTrue(tr.held)
        json.dumps(tr.metrics)
        tr.update([0,0,1.1],[0,0,.91],[.2,0,.91],False)
        self.assertFalse(tr.held)
    def test_stable_needs_release_and_time(self):
        tr=MetricTracker(plan())
        for _ in range(4):s=tr.update([0,0,.94],[0,0,.94],[0,0,.91],False)
        self.assertEqual(s[10],0)
        for _ in range(3):s=tr.update([0,0,1.0],[0,0,.94],[0,0,.91],True)
        self.assertEqual(s[10],1)
    def test_scores_gripper_and_only_executed_window(self):
        v=VeritasMetricVerifier(camera_file=str(CAM),horizon=8,health_check=False)
        state=torch.tensor([[3,0,0,1,.04,.03,0,0,1,1,0,320,240,0]],dtype=torch.float32)
        obs={'end_effector_pose':torch.tensor([[0,0,1]],dtype=torch.float32),'veritas_metrics':state}
        a=torch.zeros(1,16,7);base=v.get_value(obs,a)
        a[:,:,6]=1;self.assertGreater(base.item(),v.get_value(obs,a).item())
        a[:,:,6]=0;a[:,8:]=100
        torch.testing.assert_close(base,v.get_value(obs,a))
        rep={k:x.expand(4,*x.shape[1:]) for k,x in obs.items()}
        torch.testing.assert_close(v.get_value(rep,a.expand(4,-1,-1)),base.expand(4))
    def test_v2_close_then_lift_keeps_closed_gripper_valid(self):
        v=VeritasMetricVerifier(camera_file=str(CAM),horizon=8,health_check=False,revision='v2')
        state=torch.tensor([[1,0,0,.92,.04,.03,0,0,.92,0,0,320,240,0,.2,0,.95,.92]])
        obs={'end_effector_pose':torch.tensor([[0,0,.94]]),'veritas_metrics':state}
        a=torch.zeros(1,16,7);a[0,0,2]=-.02;a[0,1,2]=.08
        c=v.components(obs,a)
        self.assertAlmostEqual(c['gripper_cost'].item(),0.)
        self.assertAlmostEqual(c['geometry'].item(),1.,places=5)
        # Closing before reaching the source remains penalized.
        obs['end_effector_pose']=torch.tensor([[.2,0,.94]])
        self.assertEqual(v.components(obs,a)['gripper_cost'].item(),1.)
    def test_v2_goal_offset_rotates_with_target(self):
        tr=MetricTracker(plan(),revision='v2',calibration={'goal_offset_target_frame':[.1,0,.03]})
        q=2**-.5
        s=tr.update([0,0,.94],[0,0,.94],[.2,.3,.91,q,0,0,q],True)
        np.testing.assert_allclose(s[14:17],[.2,.4,.94],atol=1e-6)
    def test_v2_carry_target_allows_lateral_progress(self):
        tr=MetricTracker(plan(),revision='v2',calibration={'goal_offset_target_frame':[0,0,.03]})
        tgt=[.2,0,.91,1,0,0,0]
        tr.update([0,0,.91],[0,0,.91],tgt,False)
        s=tr.update([.01,0,.94],[.01,0,.94],tgt,False)
        self.assertTrue(tr.held)
        self.assertGreater(s[1],.15)
if __name__=='__main__':unittest.main()
