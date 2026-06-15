"""get_muse_emb.py — Download USE-4 and save the 512-dim embedding for the task.

Usage:
    conda run -n simpler_env python get_muse_emb.py \
        --instruction "put the eggplant in the basket" \
        --out task_emb.npy
"""
import argparse
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF startup noise

import tensorflow as tf
import tensorflow_hub as hub

USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="put the eggplant in the basket")
    parser.add_argument("--out", default="task_emb.npy")
    args = parser.parse_args()

    print(f"Loading USE-4 from {USE_URL} ...")
    print("(downloads ~1 GB on first run, cached after)")
    model = hub.load(USE_URL)

    emb = model([args.instruction]).numpy()[0].astype(np.float32)
    assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"
    print(f"Embedding shape: {emb.shape}  norm: {np.linalg.norm(emb):.4f}")

    np.save(args.out, emb)
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
