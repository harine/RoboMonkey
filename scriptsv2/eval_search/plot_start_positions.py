"""Diagnostic plot: start positions of the assets + arm from the TRAINING
dataset vs the EVAL episodes, highlighting the visualized (search_q) episodes.

Top-down (XY) scatter, one panel per entity (manipulated asset / target asset /
end-effector): the training-set initial positions are a faint gray cloud, the
eval episode starts are colored points, and the episodes that got a search_q
visualization are highlighted with red stars.

Reads:
  - training zarr (per task): meta/episode_ends + data/obs/{insertive,receptive,
    end_effector}_asset_pose|end_effector_pose -> first-step xyz per episode.
  - <eval_dir>/episodes.jsonl: per-episode "init_state" (added by eval_search.py).
  - <eval_dir>/search_q/ep*.npz (or ep_idx < viz_episodes): which to highlight.

Usage:
  python plot_start_positions.py --eval-dir <dir> --task <task> \
      [--viz-episodes K] [--out <png>] [--train-root <dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# friendly/widowx task id -> training data root (mirrors collect_simpler.sbatch)
_TASK_ROOT = {
    "eggplant": "/gscratch/robotics/harine/data/eggplant_in_basket_complete",
    "widowx_put_eggplant_in_basket": "/gscratch/robotics/harine/data/eggplant_in_basket_complete",
    "carrot": "/gscratch/robotics/harine/data/carrot_on_plate",
    "widowx_carrot_on_plate": "/gscratch/robotics/harine/data/carrot_on_plate",
    "spoon": "/gscratch/robotics/harine/data/spoon_on_towel",
    "widowx_spoon_on_towel": "/gscratch/robotics/harine/data/spoon_on_towel",
    "stack": "/gscratch/robotics/harine/data/stack_cube",
    "widowx_stack_cube": "/gscratch/robotics/harine/data/stack_cube",
}

# entity -> (training zarr obs key, eval init_state key, label)
_ENTITIES = [
    ("data/obs/insertive_asset_pose", "insertive_xyz", "manipulated asset"),
    ("data/obs/receptive_asset_pose", "receptive_xyz", "target asset"),
    ("data/obs/end_effector_pose", "ee_xyz", "end-effector"),
]


def _zarr_dirs(train_root: str) -> List[str]:
    shards = sorted(glob.glob(os.path.join(train_root, "state*.zarr")))
    if shards:
        return shards
    return [train_root]  # single-zarr dataset


def load_train_starts(train_root: str) -> Dict[str, np.ndarray]:
    import zarr
    acc: Dict[str, List[np.ndarray]] = {k: [] for k, _, _ in _ENTITIES}
    for d in _zarr_dirs(train_root):
        try:
            z = zarr.open(d, "r")
            ends = np.asarray(z["meta/episode_ends"][:]).astype(int)
        except Exception:
            continue
        if ends.size == 0:
            continue
        starts = np.concatenate([[0], ends[:-1]]).astype(int)
        for key, _, _ in _ENTITIES:
            if key in z:
                xyz = np.asarray(z[key][:, :3])[starts]
                acc[key].append(xyz)
    return {k: (np.concatenate(v, 0) if v else np.zeros((0, 3))) for k, v in acc.items()}


def load_eval_starts(eval_dir: str):
    rows = []
    p = os.path.join(eval_dir, "episodes.jsonl")
    if not os.path.exists(p):
        return rows
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ist = d.get("init_state")
        if ist is None:
            continue
        rows.append({"ep_idx": int(d.get("ep_idx", -1)),
                     "seed": d.get("seed"),
                     "success": bool(d.get("success", False)),
                     "init": ist})
    return rows


def _viz_idx_set(eval_dir: str, viz_episodes: int) -> set:
    """Episode indices that got a search_q dump (preferred) or ep_idx<K."""
    sq = sorted(glob.glob(os.path.join(eval_dir, "search_q", "ep*.npz")))
    idxs = set()
    for f in sq:
        base = os.path.basename(f)
        try:
            idxs.add(int(base[2:5]))  # ep###_...
        except Exception:
            pass
    if idxs:
        return idxs
    if viz_episodes and viz_episodes > 0:
        return set(range(viz_episodes))
    return set()


def plot_start_positions(eval_dir: str, task: str, viz_episodes: int = 0,
                         out_path: Optional[str] = None,
                         train_root: Optional[str] = None) -> Optional[str]:
    train_root = train_root or _TASK_ROOT.get(task)
    if train_root is None:
        print(f"[plot_starts] no training root for task {task!r}; skipping")
        return None
    train = load_train_starts(train_root)
    ev = load_eval_starts(eval_dir)
    if not ev:
        print(f"[plot_starts] no eval init_state in {eval_dir}/episodes.jsonl; skipping")
        return None
    viz = _viz_idx_set(eval_dir, viz_episodes)

    # Okabe-Ito colorblind-safe palette + distinct markers (orange cloud,
    # blue eval circles, vermillion eval stars). Eval/visualized points are
    # small + semi-transparent so the (discrete) training cloud stays readable.
    C_TRAIN = "#E69F00"   # orange
    C_EVAL = "#0072B2"    # blue
    C_VIZ = "#D55E00"     # vermillion
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (zkey, ekey, label) in zip(axes, _ENTITIES):
        tr = train.get(zkey, np.zeros((0, 3)))
        if tr.shape[0]:
            # The widowx init poses sit on a discrete grid, so 2094 trajectories
            # overlap onto few cells -- annotate the unique-cell count (@1mm) so
            # the sparse-looking cloud isn't mistaken for missing points.
            _cells = np.unique(np.round(tr[:, :2] / 1e-3).astype(np.int64),
                               axis=0).shape[0]
            ax.scatter(tr[:, 0], tr[:, 1], s=12, c=C_TRAIN, alpha=0.45,
                       linewidths=0,
                       label=f"train ({tr.shape[0]} traj; {_cells} cells @1mm)",
                       zorder=1)
        ex, ey, vx, vy = [], [], [], []
        for r in ev:
            xyz = r["init"].get(ekey)
            if xyz is None:
                continue
            if r["ep_idx"] in viz:
                vx.append(xyz[0]); vy.append(xyz[1])
            else:
                ex.append(xyz[0]); ey.append(xyz[1])
        if ex:
            ax.scatter(ex, ey, s=18, c=C_EVAL, edgecolors="k", alpha=0.55,
                       linewidths=0.3, label=f"eval ({len(ex)})", zorder=2)
        if vx:
            ax.scatter(vx, vy, s=70, marker="*", c=C_VIZ, edgecolors="k",
                       alpha=0.7, linewidths=0.4,
                       label=f"visualized ({len(vx)})", zorder=3)
        ax.set_title(label)
        ax.set_xlabel("x (world)"); ax.set_ylabel("y (world)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"start positions — {task}\n"
                 f"train cloud (orange) vs eval starts (blue, visualized = vermillion ★)"
                 f"  |  {eval_dir}",
                 fontsize=10)
    fig.tight_layout()
    out_path = out_path or os.path.join(eval_dir, "start_positions.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_starts] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--viz-episodes", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-root", default=None)
    a = ap.parse_args()
    plot_start_positions(a.eval_dir, a.task, a.viz_episodes, a.out, a.train_root)
