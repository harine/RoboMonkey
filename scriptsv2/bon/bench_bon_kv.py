"""Eval/BoN-pattern A/B speedup of the verifier: KV-prefix on vs off.

Best-of-N eval scores N candidate actions against ONE observation frame, then
the frame ADVANCES (new image next step) -> the prefix is NOT reused across
steps, only within a step (encode the frame once, forward N action suffixes vs N
full forwards). So KV's win grows with N and is ~0 at N=1.

Sweeps N and, for each, times S env-steps (fresh image per step, cache cleared
between steps) KV-off vs KV-on. Same-process A/B so contention cancels.

Env: BON_N_LIST (default "1 4 16 64 128"), BON_STEPS (steps/N, default 4).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))

INSTRUCTION = "put the eggplant in the basket"
N_LIST = [int(x) for x in os.environ.get("BON_N_LIST", "1 4 16 64 128").split()]
STEPS = int(os.environ.get("BON_STEPS", 4))


def main() -> int:
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print(f"[bon] N_LIST={N_LIST} steps/N={STEPS}", flush=True)
    rng = np.random.RandomState(0)

    for N in N_LIST:
        # Distinct frame per step; N candidate actions per frame.
        frames = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(STEPS)]
        fkeys = [rrm._raw_image_key(f) for f in frames]
        acts = [rng.uniform(-1, 1, size=(N, 7)).astype(np.float32) for _ in range(STEPS)]

        def off():
            for s in range(STEPS):
                rrm.get_rewards_paired(INSTRUCTION, [frames[s]] * N, acts[s])

        def on():
            for s in range(STEPS):
                rrm.clear_prefix_cache()  # new frame each step -> no cross-step reuse
                rrm.get_rewards_paired_kvprefix(
                    INSTRUCTION, [frames[s]] * N, acts[s], [fkeys[s]] * N)

        off(); on()  # warmup
        torch.cuda.synchronize(); t0 = time.perf_counter(); off()
        torch.cuda.synchronize(); off_t = time.perf_counter() - t0
        torch.cuda.synchronize(); t0 = time.perf_counter(); on()
        torch.cuda.synchronize(); on_t = time.perf_counter() - t0
        nsc = STEPS * N
        print(f"[bon] N={N:4d}: off={1000*off_t/nsc:6.1f} ms/score  "
              f"on={1000*on_t/nsc:6.1f} ms/score  speedup={off_t/on_t:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
