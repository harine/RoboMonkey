# scriptsv2 — L2S data + eval pipeline

Scripts that move data from raw sources to trained diffusion-policy
checkpoints, run SimplerEnv evaluations on those checkpoints, and analyze
their action-variance / verifier-ranked performance.

Organized into five sub-pipelines. Each directory is self-contained: the
shell wrapper inside it knows how to find its sibling Python file.

```
scriptsv2/
├── bridge_to_zarr/      # 1. download + filter Bridge V2  →  SIMPLER-style zarr
├── plot_zarr/           # 2. plot a (multi-)zarr dataset
├── eval_diffusion/      # 3. evaluate a diffusion_policy ckpt in SimplerEnv
├── bon_eval/            # 4. RoboMonkey best-of-N verifier sweeps
└── action_variance/     # 5. measure per-step action variance (motivation for BON)
```

The companion **[README_L2S.md](../README_L2S.md)** walks through these
sub-pipelines end-to-end with concrete examples.

---

## 1. `bridge_to_zarr/` — Bridge V2 → Zarr

| Script | What it does |
|---|---|
| `filter_bridge_v2.py` | Download Bridge V2 `meta/*` from HuggingFace, filter tasks by keyword (e.g. `carrot,plate`), download only the matching `.parquet` + `.mp4` episodes. |
| `bridge_to_zarr.py` | Convert the filtered LeRobot subset into a SIMPLER-compatible zarr shard (proxy `insertive_asset_pose` / `receptive_asset_pose`, finite-diff EE velocity, etc.). |
| `bridge_to_zarr.sh` | One-shot wrapper: runs the filter stage, then the convert stage for both carrot and eggplant. |

```bash
# Full pipeline (filter + download + convert):
bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh

# Dry run (just print task matches, no download):
DRY_RUN=1 bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh

# Filter only (skip zarr conversion):
SKIP_CONVERT=1 bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh
```

## 2. `plot_zarr/` — Zarr dataset plots

| Script | What it does |
|---|---|
| `plot_zarr.py` | Multi-shard dashboard for any RoboMonkey/Bridge zarr (one or many `--shard` paths merged into a single figure). Supports `--align-to-arm-base` to overlay SIMPLER + Bridge shards in the same frame. |
| `plot_bridge_lerobot.py` | Pre-conversion preview: plot from raw LeRobot parquet + mp4 thumbnails before running `bridge_to_zarr.py`. |

```bash
# Single shard
python scriptsv2/plot_zarr/plot_zarr.py --shard data/state0.zarr

# Multiple shards, merged into one plot
python scriptsv2/plot_zarr/plot_zarr.py \
    --shard data/state0.zarr data/state1.zarr

# Overlay SIMPLER + Bridge V2 (auto-align frames)
python scriptsv2/plot_zarr/plot_zarr.py \
    --shard data/state0.zarr data/bridge_v2_carrot.zarr \
    --align-to-arm-base
```

## 3. `eval_diffusion/` — SimplerEnv evaluation

| Script | What it does |
|---|---|
| `eval_diffusion.py` | Roll out a single diffusion_policy checkpoint (U-Net or MLP) in SimplerEnv for N episodes; logs per-episode JSONL + a summary `eval_log.json`. Supports BON via `--bon-k`. With `--seeds 17,50,3,9` runs an explicit seed list (one episode per seed, overriding `--num-episodes`). With `--viz-q` *and* BoN enabled, also dumps per-replan candidate actions + verifier rewards (Q-values) + scored frame to `<output_dir>/bon_q/ep<idx>_seed<seed>.npz`. |
| `eval_diffusion.sh` | Conda-env activation + headless `xvfb-run` wrapper around the Python eval. Forwards `SEEDS` and `VIZ_Q` env vars to the Python flags above. |
| `eval_diffusion_both.sh` | Convenience: evaluates BOTH U-Net and MLP checkpoints back-to-back for the configured task (`TASK=carrot` or `eggplant`). |
| `eval_summary.py` | Pretty-print one or more `eval_log.json` files as a comparison table. |

```bash
# Single checkpoint (eggplant)
bash scriptsv2/eval_diffusion/eval_diffusion.sh \
    ~/diffusion_policy/.../checkpoints/latest.ckpt 100

# Both architectures, eggplant
bash scriptsv2/eval_diffusion/eval_diffusion_both.sh

# Both architectures, carrot
TASK=carrot bash scriptsv2/eval_diffusion/eval_diffusion_both.sh

# Compare existing logs
python scriptsv2/eval_diffusion/eval_summary.py \
    data/eval/<run_a>/eval_log.json \
    data/eval/<run_b>/eval_log.json
```

## 4. `bon_eval/` — Best-of-N verifier sweeps

| Script | What it does |
|---|---|
| `bon_eval.sh` | Ordered sweep over `k ∈ {2,4,8,16,32,64}` (U-Net only by default; set `SKIP_MLP=0` and pass an `<mlp_ckpt>` to also run the MLP rows), using fixed `replan_every_n_steps=4 / score_num_actions=4`. Has resume support (skips runs whose `eval_log.json` already exists; set `FORCE=1` to override). Aggregates everything into a side-by-side summary at the end. Supports `SEEDS="..."` to use an explicit seed list (output dirs are namespaced with `_seeds<tag>` so they don't collide with default sweeps), and `VIZ_Q=1` to dump per-replan Q-values for branching viz. |

```bash
# Default U-Net-only sweep on eggplant (100 episodes per cell)
bash scriptsv2/bon_eval/bon_eval.sh

# Quick smoke test (5 episodes per cell)
bash scriptsv2/bon_eval/bon_eval.sh 5

# Re-run only k=16 cells
ONLY_K="16" bash scriptsv2/bon_eval/bon_eval.sh

# Switch task
TASK=carrot bash scriptsv2/bon_eval/bon_eval.sh

# Sweep ONE custom U-Net checkpoint, explicit seeds, save Q-values
# (per-replan candidate actions + verifier rewards) for later branching viz:
SEEDS="17,50,3,9" VIZ_Q=1 \
    bash scriptsv2/bon_eval/bon_eval.sh \
        /path/to/checkpoints/latest.ckpt
# Each row writes data/eval/bon/<run_name>/replan4_k<k>_score4_seeds17_50_3_9/
#   ├── eval_log.json
#   ├── episodes.jsonl
#   └── bon_q/ep000_seed17.npz, ep001_seed50.npz, ...

# Include MLP rows too (legacy mode):
SKIP_MLP=0 bash scriptsv2/bon_eval/bon_eval.sh \
    /path/to/unet/checkpoints/latest.ckpt \
    /path/to/mlp/checkpoints/latest.ckpt
```

Requires the monkey-verifier server running on `REWARD_SERVER_PORT` (default `3100`).

## 5. `action_variance/` — Per-step action variance

| Script | What it does |
|---|---|
| `analyze_variance.py` | Sample N actions at each rollout step (or read MLP `log_std` directly), report empirical std per action dim, render an annotated video where each frame shows the per-dim std bars. |
| `analyze_variance.sh` | Runs the above for BOTH eggplant checkpoints back-to-back and prints a side-by-side mean-std comparison. |

```bash
bash scriptsv2/action_variance/analyze_variance.sh
N_SAMPLES=64 bash scriptsv2/action_variance/analyze_variance.sh
```

---

## Notes

- `scripts/` (sibling directory) holds environment / infrastructure scripts
  only (`setup.sh`, `env_*.sh`, `vulkan.sh`, `run_openvla_server.sh`) plus the
  two SIMPLER rollout collectors (`collect_carrot_on_plate.sh`,
  `collect_eggplant_in_basket.sh`). Those feed shards into the pipeline above
  but are not part of `scriptsv2/`.
- All `bash *.sh` scripts here assume the `simpler_env` conda env exists and
  the `diffusion_policy` repo is cloned at `$DIFFUSION_POLICY_ROOT`
  (default `~/diffusion_policy`).
