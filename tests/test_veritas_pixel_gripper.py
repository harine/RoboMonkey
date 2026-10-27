import unittest
from pathlib import Path
import torch
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier
from diffusion_policy.policy.veritas_gripper_verifier import VeritasGripperVerifier

CAM=Path(__file__).resolve().parents[1]/'diffusion_policy/diffusion_policy/config/veritas/camera_widowx_stack_cube.json'

class Tests(unittest.TestCase):
    def test_zero_weight_exactly_preserves_original_cost(self):
        base=VeritasPixelVerifier(camera_file=str(CAM),horizon=8,health_check=False)
        new=VeritasGripperVerifier(camera_file=str(CAM),horizon=8,health_check=False,gripper_weight=0)
        obs={'end_effector_pose':torch.tensor([[-.1,0,1.]]).repeat(16,1),
             'veritas_targets':torch.tensor([[3,60,200,220,250,240,280,300.]]).repeat(16,1),
             'veritas_gripper_targets':torch.tensor([[1,0,0,1,1.]]).repeat(16,1)}
        a=torch.randn(16,16,7,generator=torch.Generator().manual_seed(42))*.02
        a[:,:,6]=torch.arange(16)[:,None]%2
        torch.testing.assert_close(new.get_value(obs,a),base.get_value(obs,a)/new.pixel_cost_scale,rtol=0,atol=0)
        a[:,8:]=100
        torch.testing.assert_close(new.get_value(obs,a),base.get_value(obs,a)/new.pixel_cost_scale,rtol=0,atol=0)

    def test_closed_after_reaching_grasp_remains_valid(self):
        v=VeritasGripperVerifier(camera_file=str(CAM),horizon=8,health_check=False)
        tcp=torch.tensor([[-.1,0,1.]])
        uv=v.project(tcp).flatten()
        obs={'end_effector_pose':tcp,'veritas_targets':torch.tensor([[1,60,*uv,-1,-1,-1,-1]]),
             'veritas_gripper_targets':torch.tensor([[1,0,-1,-1,1.]])}
        a=torch.zeros(1,16,7);a[0,1:8,2]=.05
        self.assertEqual(v.components(obs,a)['gripper_cost'].item(),0.)
        a[:,:,6]=1
        self.assertEqual(v.components(obs,a)['gripper_cost'].item(),1.)

    def test_final_open_and_no_change_to_pixel_cost(self):
        v=VeritasGripperVerifier(camera_file=str(CAM),horizon=8,health_check=False)
        obs={'end_effector_pose':torch.tensor([[-.1,0,1.]]),
             'veritas_targets':torch.tensor([[0,-1,-1,-1,-1,-1,-1,-1.]]),
             'veritas_gripper_targets':torch.tensor([[0,-1,-1,-1,1.]])}
        a=torch.zeros(1,16,7);closed=v.components(obs,a)
        a[:,:,6]=1;opened=v.components(obs,a)
        self.assertEqual(opened['pixel_cost'].item(),closed['pixel_cost'].item())
        self.assertAlmostEqual((opened['score']-closed['score']).item(),.1)
        self.assertLessEqual(closed['gripper_penalty'].item(),.100001)

if __name__=='__main__':unittest.main()
