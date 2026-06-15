#!/usr/bin/env python
"""Offline action-error (RMSE) eval for the RoboMonkey reproduction.

For each frame in the SUCCESS-filtered val set we:
  1. sample M=max(N) actions from the OpenVLA policy server (one batched /batch
     call) on the center-cropped policy image -- exactly the collection pipeline;
  2. score all M candidates with the in-process RoboMonkey verifier (single-step
     scoring vs the save_reward_img verifier image) -- the original RoboMonkey
     verifier, not chunk-sum;
  3. for each N in {1,2,4,...,M} take Best-of-first-N (argmax verifier reward over
     the first N iid samples) and measure RMSE of the selected continuous action
     vs the reference action data/actions[frame] (the action actually executed in
     the successful trajectory).

Because the candidates are iid policy samples, the first N are a valid random
N-subset, so a single M-sample pass yields the whole nested N-sweep. N=1 is the
no-selection baseline (expected RMSE ~ policy sampling spread vs the reference);
a good verifier should drive RMSE *below* that baseline as N grows.

Reference = executed action (eggplant-in-basket has NO scripted expert;
expert_action_mean is just the mean of the collection-time OpenVLA samples).

Runs in the `monkey-verifier` env. Assumes the OpenVLA sglang server is already
up on --action-server-port (the .sbatch wrapper launches it). The verifier loads
in-process. All verifier caches are disabled (env, below) so per-frame scores can
never collide on a reused temp image path.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import requests

# --- disable every verifier cache BEFORE importing the client ---------------
# VerifierClient's reward LRU keys images by *path string* (_hash_image), so a
# reused temp filename would alias scores across frames. Off => always fresh.
os.environ.setdefault("ROBOMONKEY_REWARD_CACHE_SIZE", "0")
os.environ.setdefault("ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE", "0")
os.environ.setdefault("ROBOMONKEY_PROC_IMAGE_CACHE_SIZE", "0")

import tensorflow as tf  # CPU-only here; used only for the image preprocessing
tf.config.set_visible_devices([], "GPU")

from PIL import Image
import zarr


# ===========================================================================
#  Image preprocessing -- copied verbatim from the eval pipeline so the policy
#  and verifier see byte-for-byte the same images they would in run_simpler_eval.
# ===========================================================================
def crop_and_resize(image, crop_scale, batch_size):
    """openvla_utils.crop_and_resize (center crop to crop_scale area, resize 224)."""
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded = True
    else:
        expanded = False
    new_h = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_w = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    h_off = (1 - new_h) / 2
    w_off = (1 - new_w) / 2
    boxes = tf.stack([h_off, w_off, h_off + new_h, w_off + new_w], axis=1)
    image = tf.image.crop_and_resize(image, boxes, tf.range(batch_size), (224, 224))
    if expanded:
        image = image[0]
    return image


def preprocess_policy_image(image_uint8, path):
    """collect_trajectories._preprocess_and_save_image (center_crop=True)."""
    image = Image.fromarray(image_uint8).convert("RGB")
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, 0.9, 1)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    Image.fromarray(image.numpy()).convert("RGB").save(path)
    return str(Path(path).absolute())


def save_reward_img(image_uint8, path):
    """simpler_utils.save_reward_img (RLDS-style 256x256 double-lanczos jpg)."""
    image = tf.convert_to_tensor(image_uint8)
    image = tf.image.encode_jpeg(image)
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = tf.io.encode_jpeg(image, quality=95)
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    Image.fromarray(image.numpy()).save(path)
    return str(Path(path).absolute())


# ===========================================================================
#  OpenVLA policy server
# ===========================================================================
def get_batch_actions(instruction, image_path, batch_size, temperature, port):
    payload = {"instructions": [instruction] * batch_size,
               "image_path": image_path, "temperature": temperature}
    r = requests.post(f"http://127.0.0.1:{port}/batch", data=json.dumps(payload),
                      headers={"Content-Type": "application/json"}, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"action server error {r.status_code}: {r.text[:300]}")
    d = r.json()
    return np.array(d["output_ids"]), np.array(d["actions"], dtype=np.float32)


# OpenVLA action-token range (openvla_server.TokenToAction: vocab_size=32000,
# 256 bins -> 255 bin centers, discretized clipped to [0,254]). Valid action
# token ids are therefore [VOCAB-(N_BINS-1), VOCAB-1] = [31745, 31999]. The
# server clips strays to this range when producing the CONTINUOUS action, but
# the raw output_ids it returns are unclipped -- and the verifier does
# `token_id - 1000` then embeds, so a sampled id < 1000 indexes negative and
# triggers a CUDA device-side assert. Clipping here mirrors the server's bin
# clipping exactly, so the tokens we score == the continuous actions we RMSE.
_VOCAB, _N_BINS = 32000, 256
TOK_LO, TOK_HI = _VOCAB - (_N_BINS - 1), _VOCAB - 1   # 31745, 31999


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
        default="/gscratch/robotics/harine/data/eggplant_in_basket_val_success/val_success.zarr")
    ap.add_argument("--out-dir",
        default="/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_action_error")
    ap.add_argument("--max-n", type=int, default=64, help="2^6; sweep is 1..max-n")
    ap.add_argument("--n-frames", type=int, default=2000,
                    help="random subsample of frames; <=0 means ALL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--action-server-port", type=int, default=3200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    Ns = [1 << k for k in range(int(np.log2(args.max_n)) + 1)]  # 1,2,4,...,max_n
    assert Ns[-1] == args.max_n, f"max-n must be a power of two, got {args.max_n}"

    g = zarr.open(args.dataset, mode="r")
    images = g["data"]["obs"]["agentview_image"]
    ref_actions = g["data"]["actions"]
    n_total = images.shape[0]
    instruction = str(g.attrs.get("task_description", "")).lower()
    assert instruction, "dataset missing task_description attr"

    rng = np.random.default_rng(args.seed)
    if args.n_frames is not None and args.n_frames > 0 and args.n_frames < n_total:
        frame_idx = np.sort(rng.choice(n_total, size=args.n_frames, replace=False))
    else:
        frame_idx = np.arange(n_total)
    nf = len(frame_idx)

    run = args.tag or f"M{args.max_n}_f{nf}_s{args.seed}"
    out_dir = os.path.join(args.out_dir, run)
    os.makedirs(out_dir, exist_ok=True)
    tdir = os.path.join(out_dir, "transfer")
    os.makedirs(tdir, exist_ok=True)
    policy_path = os.path.join(tdir, "policy_img.jpg")
    reward_path = os.path.join(tdir, "reward_img.jpg")

    print(f"[cfg] dataset={args.dataset}", flush=True)
    print(f"[cfg] instruction='{instruction}'  frames={nf}/{n_total}  "
          f"M={args.max_n}  Ns={Ns}  temp={args.temperature}", flush=True)

    from verifier_client import VerifierClient
    print("[verifier] loading in-process model ...", flush=True)
    client = VerifierClient("in_process")
    print("[verifier] ready", flush=True)

    # per-N accumulators of per-frame RMSE (overall 7D, arm 6D, gripper 1D)
    rmse_all = {N: [] for N in Ns}
    rmse_arm = {N: [] for N in Ns}
    rmse_grip = {N: [] for N in Ns}
    per_frame = []
    # Raw pool cache: the whole point of saving these is that EVERY metric
    # variant (z-scored NRMSE, shuffled-pool subsampling, per-dim breakdown,
    # different selection rules) can be recomputed offline from this with NO GPU
    # re-run. See nrmse_postprocess.py.
    raw_acts, raw_rewards, raw_ref = [], [], []

    t0 = time.time()
    n_clipped = 0          # stray tokens clipped (proves the fix engaged)
    n_frames_clipped = 0
    for i, f in enumerate(frame_idx):
        img = np.asarray(images[f])
        ref = np.asarray(ref_actions[f], dtype=np.float32)
        preprocess_policy_image(img, policy_path)
        save_reward_img(img, reward_path)

        out_ids, acts = get_batch_actions(instruction, policy_path,
                                          args.max_n, args.temperature,
                                          args.action_server_port)
        # Clip stray sampled tokens into the action-bin range so they match the
        # (already-clipped) continuous actions and never index the verifier
        # embedding out of bounds. See TOK_LO/TOK_HI note above.
        raw = np.asarray(out_ids)
        n_stray = int(((raw < TOK_LO) | (raw > TOK_HI)).sum())
        if n_stray:
            n_clipped += n_stray
            n_frames_clipped += 1
        out_ids = np.clip(out_ids, TOK_LO, TOK_HI).astype(np.int64)
        # score token-ids (matches run_simpler_eval's get_rewards(output_ids))
        rewards = np.asarray(client.score_candidates(instruction, reward_path, out_ids),
                             dtype=np.float64)

        # cache the raw 64-candidate pool for offline metric recomputation
        raw_acts.append(np.asarray(acts, dtype=np.float32))      # (M, 7)
        raw_rewards.append(rewards.astype(np.float32))           # (M,)
        raw_ref.append(ref.astype(np.float32))                   # (7,)

        # inline metric below is RAW + first-N only -- a live sanity readout.
        # The authoritative NRMSE / shuffled-pool numbers come from postprocess.
        rec = {"frame": int(f)}
        for N in Ns:
            sel = int(np.argmax(rewards[:N]))
            a = acts[sel]
            rmse_all[N].append(rmse(a, ref))
            rmse_arm[N].append(rmse(a[:6], ref[:6]))
            rmse_grip[N].append(rmse(a[6:7], ref[6:7]))
            rec[f"sel_n{N}"] = sel
            rec[f"rmse_n{N}"] = rmse_all[N][-1]
        per_frame.append(rec)

        if (i + 1) % 25 == 0 or i + 1 == nf:
            dt = time.time() - t0
            eta = dt / (i + 1) * (nf - i - 1)
            cur = {N: float(np.mean(rmse_all[N])) for N in Ns}
            print(f"[{i+1:>5}/{nf}] {dt:6.0f}s  {dt/(i+1):4.2f}s/fr  ETA {eta/60:5.1f}m"
                  f"  rmse(N=1)={cur[1]:.4f} rmse(N={args.max_n})={cur[args.max_n]:.4f}"
                  f"  clipped={n_clipped}tok/{n_frames_clipped}fr",
                  flush=True)

    # --- aggregate ----------------------------------------------------------
    def summarize(d):
        out = {}
        for N in Ns:
            v = np.asarray(d[N])
            out[str(N)] = {"mean": float(v.mean()), "std": float(v.std()),
                           "sem": float(v.std() / np.sqrt(len(v))), "n": int(len(v))}
        return out

    summary = {"dataset": args.dataset, "instruction": instruction,
               "n_frames": nf, "n_total": n_total, "max_n": args.max_n, "Ns": Ns,
               "temperature": args.temperature, "seed": args.seed,
               "reference": "data/actions (executed action in successful traj)",
               "stray_tokens_clipped": n_clipped, "frames_with_stray": n_frames_clipped,
               "rmse_overall_7d": summarize(rmse_all),
               "rmse_arm_6d": summarize(rmse_arm),
               "rmse_gripper_1d": summarize(rmse_grip)}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, "per_frame.npz"),
                        frame=np.array([r["frame"] for r in per_frame]),
                        **{f"rmse_n{N}": np.array([r[f"rmse_n{N}"] for r in per_frame])
                           for N in Ns})
    # RAW POOL -> enables offline NRMSE / shuffled-pool / per-dim recompute.
    raw_path = os.path.join(out_dir, "raw_pool.npz")
    np.savez_compressed(raw_path,
                        frame=np.asarray([r["frame"] for r in per_frame], dtype=np.int64),
                        actions=np.stack(raw_acts, axis=0),    # (F, M, 7) unnormalized
                        rewards=np.stack(raw_rewards, axis=0), # (F, M) verifier scalar
                        ref=np.stack(raw_ref, axis=0))         # (F, 7) executed action
    print(f"[raw] cached pool -> {raw_path}  "
          f"actions={np.stack(raw_acts).shape}", flush=True)

    print("\n==================== ACTION-ERROR (RMSE) vs executed action ====================")
    print(f"  frames={nf}  reference=executed action  (lower = closer to successful action)")
    print(f"  {'N':>4} | {'RMSE(7D)':>10} {'±sem':>7} | {'arm(6D)':>9} | {'gripper':>9}")
    print("  " + "-" * 56)
    base = summary["rmse_overall_7d"][str(Ns[0])]["mean"]
    for N in Ns:
        o = summary["rmse_overall_7d"][str(N)]
        a = summary["rmse_arm_6d"][str(N)]["mean"]
        gp = summary["rmse_gripper_1d"][str(N)]["mean"]
        delta = (o["mean"] - base) / base * 100
        print(f"  {N:>4} | {o['mean']:>10.4f} {o['sem']:>7.4f} | {a:>9.4f} | {gp:>9.4f}"
              f"   ({delta:+5.1f}% vs N=1)")
    print("=" * 80)
    print(f"[done] {time.time()-t0:.0f}s  ->  {out_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
