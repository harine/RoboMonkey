import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from diffusion_policy.env.simpler.veritas.metric_plan import PlanProvider


class Tests(unittest.TestCase):
    def test_budget_and_per_rollout_limits_fail_before_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PlanProvider(tmp,'stack',tmp)
            for _ in range(3):p.reserve('one-reset')
            with self.assertRaisesRegex(RuntimeError,'maximum 3'):p.reserve('one-reset')
            ledger=json.loads(p.ledger.read_text())
            ledger['requests'][0]['charged_or_reserved_usd']=99.75
            p.ledger.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):p.reserve('new-reset')
            self.assertEqual(len(json.loads(p.ledger.read_text())['requests']),3)

    @patch('diffusion_policy.env.simpler.veritas.metric_plan.load_api_keys',return_value=[('test','unused')])
    @patch('diffusion_policy.env.simpler.veritas.metric_plan.genai.Client')
    def test_same_seed_gets_fresh_plan_after_each_reset(self,client,keys):
        plan=dict(source='cube',target='cube',relationship='on',
            stages=[dict(phase=s,uv=[500,500],gripper='closed') for s in
                    ['approach','grasp','lift','transport','lower','release','retreat']],
            grasp_radius_m=.04,lift_clearance_m=.06,goal_xy_tolerance_m=.03,
            goal_height_offset_m=.03,grasp_offset_z_m=0,notes='test')
        client.return_value.models.generate_content.return_value=SimpleNamespace(
            text=json.dumps(plan),usage_metadata=None)
        with tempfile.TemporaryDirectory() as tmp:
            p=PlanProvider(tmp,'stack',tmp)
            frame=np.zeros((480,640,3),dtype=np.uint8)
            p.get(123,frame,'stack cubes')
            first=json.loads((Path(tmp)/'plans/seed123.json').read_text())
            frame[0,0]=255
            p.get(123,frame,'stack cubes')
            second=json.loads((Path(tmp)/'plans/seed123.json').read_text())
            self.assertEqual(client.return_value.models.generate_content.call_count,2)
            self.assertNotEqual(first['reset_id'],second['reset_id'])
            self.assertNotEqual(first['frame_sha256'],second['frame_sha256'])
            self.assertEqual(len(json.loads(p.ledger.read_text())['requests']),2)


if __name__=='__main__':unittest.main()
