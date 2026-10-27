import unittest
import torch
from diffusion_policy.policy.gripper_candidate_aug import augment_gripper_candidates
class Tests(unittest.TestCase):
    def test_preserve_original_and_arm_paths_without_mutation(self):
        x=torch.randn(1,16,16,7);copy=x.clone();out=augment_gripper_candidates(x)
        self.assertEqual(tuple(out.shape),(1,48,16,7))
        torch.testing.assert_close(x,copy);torch.testing.assert_close(out[:,:16],x)
        for k in [1,2]:torch.testing.assert_close(out[:,k*16:(k+1)*16,:,:6],x[:,:,:,:6])
        self.assertTrue(torch.all(out[:,16:32,:,6]==1));self.assertTrue(torch.all(out[:,32:,:,6]==0))
if __name__=='__main__':unittest.main()
