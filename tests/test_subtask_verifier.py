import json,tempfile,unittest
from google.genai import _transformers,types
from diffusion_policy.env.simpler.veritas.pixel_gripper_plan import GripPlan
from diffusion_policy.env.simpler.veritas.metric_plan import PlanProvider
from diffusion_policy.env.simpler.veritas.subtask_controller import PositionChecks

class Tests(unittest.TestCase):
    def test_sdk_schema_has_no_unsupported_tuple(self):
        schema=GripPlan.model_json_schema()
        self.assertNotIn('prefixItems',json.dumps(schema))
    def test_trial_budget_shared_across_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PlanProvider(tmp,'stack',tmp,max_requests_per_episode=None,budget_group='trial',group_cap_usd=10)
            for i in range(100):p.reserve(str(i))
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):p.reserve('new-reset')
            self.assertLessEqual(sum(r['charged_or_reserved_usd'] for r in json.loads(p.ledger.read_text())['requests']),10.000001)
    def test_closed_alone_not_grasp_and_release_required(self):
        c=PositionChecks()
        self.assertFalse(c.update([0,0,.05],[0,0,.03],[.1,0,.03],False)['held'])
        self.assertTrue(c.update([0,0,.08],[0,0,.06],[.1,0,.03],False)['held'])
        for _ in range(3):v=c.update([.1,0,.08],[.1,0,.06],[.1,0,.03],False)
        self.assertFalse(v['valid_by_stage'][3])
        v=c.update([.1,0,.10],[.1,0,.06],[.1,0,.03],True)
        self.assertTrue(v['valid_by_stage'][3])
        self.assertFalse(v['held'])
if __name__=='__main__':unittest.main()
