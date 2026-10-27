"""Rescore saved BC arm paths with original/open/closed gripper commands; no API calls."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
from diffusion_policy.policy.veritas_gripper_verifier import VeritasGripperVerifier
from diffusion_policy.policy.gripper_candidate_aug import augment_gripper_candidates
p=Path(sys.argv[1]);repo=Path(__file__).resolve().parents[2]
z=np.load(next((p/'search_q').glob('*.npz')))
traces=[json.loads(x) for x in (p/'seed100067.jsonl').read_text().splitlines()]
v=VeritasGripperVerifier(camera_file=str(repo/'diffusion_policy/diffusion_policy/config/veritas/camera_widowx_stack_cube.json'),horizon=8,health_check=False,gripper_weight=.1,plan_done_mode='zero')
records=[];summary=[]
for j,d in enumerate(traces):
 a=torch.tensor(z['candidate_actions_full'][j])[None];n=a.shape[1]
 aug=augment_gripper_candidates(a)[0]
 obs={k.removeprefix('policy_obs__'):torch.tensor(z[k][j]).expand(3*n,*z[k][j].shape[1:]) for k in z.files if k.startswith('policy_obs__')}
 comp={k:x.detach().numpy() for k,x in v.components(obs,aug).items()}
 np.testing.assert_allclose(comp['score'][:n],d['values'],atol=2e-5,rtol=2e-5)
 pick=int(comp['score'].argmax());m=d['metrics']['position_checks']
 for i in range(3*n):
  records.append(dict(t=d['t'],base_candidate=i%n,variant=['original','open','closed'][i//n],selected=i==pick,**{k:float(x[i]) for k,x in comp.items()}))
 summary.append(dict(t=d['t'],phase=d['metrics']['phase'],selected_base=pick%n,selected_variant=['original','open','closed'][pick//n],original_selected=d['selected_index'],gripper_min=float(a[...,6].min()),gripper_max=float(a[...,6].max()),open_count_first8=int((a[:,:,:8,6]>=.5).sum()),open_count_tail8=int((a[:,:,8:,6]>=.5).sum()),mean_open_minus_closed=float(np.mean(comp['score'][n:2*n]-comp['score'][2*n:])),source_target_xy_m=m['source_target_xy_m'],source_target_height_m=m['source_target_height_m'],held=m['held']))
with (p/'gripper_counterfactual_scores.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
(p/'gripper_counterfactual_summary.json').write_text(json.dumps(summary,indent=2))
for x in summary:print(x)
