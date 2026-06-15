"""eval_robomonkey.py — Score the RoboMonkey verifier on the same single-step pairs.

Run in the `monkey-verifier` conda env (torch + LLaVA-7B). Reads the
`pairs.npz` written by eval_dqc.py (images + raw 7-D actions + success labels),
so the verifier scores byte-identical (image, action) pairs to the DQC Q.

Action encoding mirrors eval_diffusion.py exactly: each 7-D action → OpenVLA
token IDs via TokenActionConverter(n_action_bins=256, unnorm_key="bridge_orig").
The verifier (RobotRewardModel) is loaded in-process by default; needs MODEL_DIR
(LLaVA-7B weights, ~14 GB GPU). Pass --reward-server-port >0 to hit a running
infer_server instead.

If eval_dqc.py already wrote dqc_scores.csv, this also emits the combined
comparison.png + a per-episode AUC table — the head-to-head success/failure view.

Usage
-----
conda run -n monkey-verifier python eval_robomonkey.py \
    --out-dir results/dqc_vs_verifier \
    --instruction "put the eggplant in the basket"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]            # /home/harine/RoboMonkey


def _auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = np.asarray(pos)[:, None] - np.asarray(neg)[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def score_verifier(images, actions, instruction, port, batch):
    sys.path.insert(0, str(_REPO / "openvla-mini"))
    sys.path.insert(0, str(_REPO / "monkey-verifier" / "src"))

    from transformers.configuration_utils import PretrainedConfig
    if not getattr(PretrainedConfig, "_safe_repr_patched", False):
        PretrainedConfig.__repr__ = lambda self: f"<{self.__class__.__name__}>"
        PretrainedConfig._safe_repr_patched = True
    from experiments.robot.token_action_converter import TokenActionConverter
    from verifier_client import VerifierClient

    converter = TokenActionConverter(n_action_bins=256, unnorm_key="bridge_orig")
    token_ids = np.stack([np.asarray(converter.action_to_token(a), dtype=np.int64)
                          for a in actions.astype(np.float32)])

    client = VerifierClient.from_port(port)
    client.health_check()
    G = len(actions)
    print(f"[verifier] scoring {G} pairs (instruction={instruction!r}) ...")
    scores = np.zeros(G, np.float32)
    for s in range(0, G, batch):
        ims = [images[i] for i in range(s, min(s + batch, G))]
        scores[s:s + batch] = client.score_paired(instruction, ims, token_ids[s:s + batch])
        print(f"  {min(s + batch, G)}/{G}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/dqc_vs_verifier")
    ap.add_argument("--instruction", default="put the eggplant in the basket")
    ap.add_argument("--reward-server-port", type=int, default=0,
                    help="<=0 in-process LLaVA; >0 HTTP to a running infer_server.")
    ap.add_argument("--verifier-batch", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    pairs_file = out / "pairs.npz"
    if not pairs_file.exists():
        sys.exit(f"{pairs_file} not found — run eval_dqc.py first to create it.")
    d = np.load(pairs_file, allow_pickle=True)
    images, actions = d["images"], d["actions"]
    ep_idx, t_real, success = d["ep_idx"], d["t_real"], d["success"].astype(bool)
    instruction = args.instruction or str(d["instruction"])

    scores = score_verifier(images, actions, instruction,
                            args.reward_server_port, args.verifier_batch)

    import pandas as pd
    np.savez_compressed(out / "verifier_scores.npz", verifier_score=scores,
                        instruction=np.asarray(instruction))
    vdf = pd.DataFrame({"ep_idx": ep_idx, "success": success, "branch_t": t_real,
                        "q_value": np.nan, "verifier_score": scores, "source": "zarr"})
    vdf.to_csv(out / "robomonkey_scores.csv", index=False)
    print(f"[verifier] {len(vdf)} rows → {out/'robomonkey_scores.csv'}")

    g = vdf.groupby("ep_idx").agg(s=("success", "first"), v=("verifier_score", "mean"))
    print(f"[auc] verifier per-episode AUC="
          f"{_auc(g[g.s]['v'].values, g[~g.s]['v'].values):.3f}")

    # Combined comparison if DQC already ran.
    dqc_csv = out / "dqc_scores.csv"
    if dqc_csv.exists():
        from plot_utils import make_plot  # jax-free
        ddf = pd.read_csv(dqc_csv)
        merged = ddf.drop(columns=["verifier_score"]).copy()
        merged["verifier_score"] = scores
        merged.to_csv(out / "scores.csv", index=False)
        n_s = merged.groupby("ep_idx")["success"].first().sum()
        n_f = merged["ep_idx"].nunique() - n_s
        make_plot(merged, out / "comparison.png",
                  f"DQC Q vs RoboMonkey verifier — single-step zarr\n{instruction}  |  "
                  f"{n_s} success / {n_f} failure eps  |  {len(merged)} (s,a) pairs", "zarr")
        print(f"[compare] wrote {out/'comparison.png'} and {out/'scores.csv'}")
    else:
        print(f"[compare] {dqc_csv} not found — run eval_dqc.py for the combined plot.")


if __name__ == "__main__":
    main()
