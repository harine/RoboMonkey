"""Adapt eval_search.py `search_q/*.npz` -> the schema viz_action_branches.py
expects (the older `bon_q` field names).

The eval already records candidate actions, per-candidate verifier values, the
chosen index, the frame, the camera intrinsic/extrinsic, and the EE world pose
-- so augment_bon_q.py (which loads SimplerEnv to *add* camera/pose) is NOT
needed. We only rename keys and reshape the camera extrinsic. The widowx
base->world yaw flip is applied inside viz_action_branches.py, so candidate
actions are passed through unchanged.

Usage: python adapt_search_q.py <src_search_q_dir> <out_dir>
"""
import sys
import glob
import os

import numpy as np


def main(src: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    if not files:
        print(f"no npz in {src}")
        return
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        R = int(np.asarray(d["candidate_actions"]).shape[0])
        Twc = np.asarray(d["cam_T_wc"], dtype=np.float32)        # (4,4) world->cam
        ext = np.tile(Twc[:3, :][None], (R, 1, 1))               # (R,3,4)
        bt = np.asarray(d["branch_t"]).astype(np.int64)
        rev = int(bt[1] - bt[0]) if bt.size > 1 else int(np.asarray(d["candidate_actions"]).shape[2])
        # Carry over all original keys (seed, ep_idx, ee_xyz, ...) and add the
        # bon-schema names viz_action_branches.py expects.
        out_d = dict(d)
        out_d.update(dict(
            candidate_actions=np.asarray(d["candidate_actions"], dtype=np.float32),
            per_candidate_mean_reward=np.asarray(d["values"], dtype=np.float32),
            selected_index=np.asarray(d["selected_index"]).astype(np.int64),
            selected_reward=np.asarray(d["selected_value"], dtype=np.float32),
            tcp_world_p=np.asarray(d["ee_xyz"], dtype=np.float32),
            cam_intrinsic=np.asarray(d["cam_K"], dtype=np.float32),
            cam_extrinsic=ext,
            bon_replan_every_n_steps=np.int64(rev),
            branch_t=bt,
        ))
        np.savez_compressed(os.path.join(out, os.path.basename(f)), **out_d)
        print("adapted", os.path.basename(f))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python adapt_search_q.py <src_search_q_dir> <out_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
