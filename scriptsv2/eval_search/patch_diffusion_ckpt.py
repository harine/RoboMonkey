#!/usr/bin/env python3
"""Rewrite the baked DQC-verifier paths in a diffusion-search checkpoint so it can
be eval'd on-cluster.

The diffusion search policies were trained OFF-cluster, so their baked
`cfg.policy.verifier.{ckpt_dir,text_emb_file}` point at non-existent `/home/harine/...`
paths. This rewrites them to the valid gscratch / mmfs1 locations (verifier SEMANTICS
unchanged: critic_type=chunk, score_full_horizon=True, q_agg=min). Same fix already
applied to the 10k/20k/33k pathfix ckpts.

    python scriptsv2/eval_search/patch_diffusion_ckpt.py <src.ckpt> <dst.ckpt>
"""
import sys
import torch

CKPT_DIR = "/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017"
TEXT_EMB = "/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_use3.npy"


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", ck.get("config"))
    v = cfg["policy"]["verifier"]
    try:
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass
    old_ck, old_te = v.get("ckpt_dir"), v.get("text_emb_file")
    v["ckpt_dir"] = CKPT_DIR
    v["text_emb_file"] = TEXT_EMB
    # sanity: targets must exist
    import os
    assert os.path.isdir(CKPT_DIR), f"missing {CKPT_DIR}"
    assert os.path.isfile(TEXT_EMB), f"missing {TEXT_EMB}"
    torch.save(ck, dst)
    print(f"patched {src}\n   ckpt_dir : {old_ck} -> {v['ckpt_dir']}")
    print(f"   text_emb : {old_te} -> {v['text_emb_file']}\n   -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
