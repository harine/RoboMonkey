"""eval_dqc.py — Sample eggplant single-step pairs + build the final comparison plot.

Run in the `simpler_env` conda env (has zarr + matplotlib).

This is the orchestration glue for the DQC-vs-verifier comparison. The actual
DQC scoring uses the REAL agent (eval_dqc_critics.py, `dqc` env) — a faithful
forward pass, not a reconstruction — and the verifier uses real LLaVA
(eval_robomonkey.py). This script only:

  default mode   sample balanced success/failure steps from the zarr and write
                 pairs.npz: images (uint8), single raw actions (7), raw 8-step
                 action chunks (8,7 with NaN past the episode end), labels.
                 Both eval_dqc_critics.py and eval_robomonkey.py read pairs.npz,
                 so all three scorers see identical inputs.

  --plot-only    merge dqc_scores.csv (q_value, q_chunk) + verifier_scores.npz
                 (verifier_score) into comparison.png + scores.csv.

Chunk note: the DQC chunk critic was trained on chunks assembled from
*normalized* actions with zero-padding past the episode end (infinite-horizon
backup). We therefore store raw next-8 actions with NaN in the pad slots and let
eval_dqc_critics.py normalize then zero-fill — reproducing training exactly.

Usage
-----
conda run -n simpler_env python eval_dqc.py \
    --zarr-path /home/harine/data/eggplant_in_basket/all_data/state0.zarr \
    --out-dir results/dqc_vs_verifier --max-eps-per-class 40 --stride 4
# ... then eval_robomonkey.py (verifier) and eval_dqc_critics.py (DQC) ...
conda run -n simpler_env python eval_dqc.py --out-dir results/dqc_vs_verifier --plot-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BACKUP_HORIZON = 8  # chunk critic action chunk length (= checkpoint backup_horizon)


def sample_pairs(zarr_path, max_per_class, stride, seed):
    """Balanced success/failure episodes, strided over steps. For each sampled
    step also collect the next BACKUP_HORIZON raw actions (NaN-padded past the
    episode end) for the chunk critic."""
    import zarr
    z = zarr.open(zarr_path, "r")
    obs = z["data"]["obs"]
    if "agentview_image" not in list(obs.keys()):
        raise ValueError(f"{zarr_path} has no agentview_image — needs images.")

    dones, rewards = z["data"]["dones"][:], z["data"]["rewards"][:]
    ep_ends = np.where(dones)[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    success = rewards[ep_ends] > 0

    n_s, n_f = int(success.sum()), int((~success).sum())
    print(f"[sample] {len(ep_ends)} eps — {n_s} success, {n_f} failure")
    rng = np.random.default_rng(seed)
    sel_s = rng.choice(np.where( success)[0], min(max_per_class, n_s), replace=False)
    sel_f = rng.choice(np.where(~success)[0], min(max_per_class, n_f), replace=False)
    eps = np.sort(np.concatenate([sel_s, sel_f]))
    print(f"[sample] {len(sel_s)} success + {len(sel_f)} failure eps, stride={stride}")

    act_ds = z["data"]["actions"]
    A = act_ds.shape[-1]
    g_index, ep_idx, t_real, succ, chunks = [], [], [], [], []
    for e in eps:
        s, en = int(ep_starts[e]), int(ep_ends[e]) + 1   # [s, en)
        for g in range(s, en, stride):
            n = min(BACKUP_HORIZON, en - g)
            chunk = np.full((BACKUP_HORIZON, A), np.nan, np.float32)
            chunk[:n] = np.asarray(act_ds[g:g + n], np.float32)
            chunks.append(chunk)
            g_index.append(g); ep_idx.append(int(e))
            t_real.append(g - s); succ.append(bool(success[e]))
    g_index = np.asarray(g_index)

    print(f"[sample] reading {len(g_index)} frames + single actions ...")
    images  = np.stack([np.asarray(obs["agentview_image"][g]) for g in g_index]).astype(np.uint8)
    actions = np.stack([np.asarray(act_ds[g]) for g in g_index]).astype(np.float32)
    return dict(g_index=g_index, ep_idx=np.asarray(ep_idx), t_real=np.asarray(t_real),
                success=np.asarray(succ), images=images, actions=actions,
                action_chunks=np.stack(chunks))   # (G, 8, 7) raw, NaN-padded


def build_comparison(out: Path):
    import pandas as pd
    from plot_utils import make_plot

    dqc_csv = out / "dqc_scores.csv"
    vfile = out / "verifier_scores.npz"
    if not dqc_csv.exists():
        raise SystemExit(f"{dqc_csv} not found — run eval_dqc_critics.py (dqc env) first.")
    df = pd.read_csv(dqc_csv)   # ep_idx, success, branch_t, q_value, q_chunk
    if vfile.exists():
        df["verifier_score"] = np.load(vfile, allow_pickle=True)["verifier_score"]
    else:
        print(f"[plot] {vfile} missing — DQC-only (run eval_robomonkey.py for the verifier).")
        df["verifier_score"] = np.nan
    df["source"] = "zarr"
    df.to_csv(out / "scores.csv", index=False)

    n_s = df.groupby("ep_idx")["success"].first().sum()
    n_f = df["ep_idx"].nunique() - n_s
    make_plot(df, out / "comparison.png",
              f"DQC single-step Q vs DQC chunk Q vs RoboMonkey verifier — eggplant (single-step zarr)\n"
              f"{n_s} success / {n_f} failure eps  |  {len(df)} (s,a) pairs", "zarr")
    print(f"[plot] wrote {out/'comparison.png'} and {out/'scores.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr-path",
                    default="/home/harine/data/eggplant_in_basket/all_data/state0.zarr")
    ap.add_argument("--out-dir", default="results/dqc_vs_verifier")
    ap.add_argument("--instruction", default="put the eggplant in the basket")
    ap.add_argument("--max-eps-per-class", type=int, default=40)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        build_comparison(out)
        return

    pairs = sample_pairs(args.zarr_path, args.max_eps_per_class, args.stride, args.seed)
    pairs_file = out / "pairs.npz"
    np.savez_compressed(pairs_file, instruction=np.asarray(args.instruction), **pairs)
    print(f"[pairs] wrote {pairs_file}  ({pairs_file.stat().st_size/1e6:.0f} MB) — "
          f"feeds eval_dqc_critics.py (DQC) and eval_robomonkey.py (verifier)")


if __name__ == "__main__":
    main()
