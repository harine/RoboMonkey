# README_L2S — Learning-to-Sample data collection & policy evaluation

End-to-end recipe for the two SIMPLER tasks we use for L2S work:

- **`widowx_carrot_on_plate`** — "put carrot on plate"
- **`widowx_put_eggplant_in_basket`** — "put eggplant in basket"

Two data sources feed the same downstream Zarr layout:

1. **Synthetic** — OpenVLA rollouts in SIMPLER, written directly to a Zarr shard.
2. **Real** — Bridge V2 LeRobot subset, filtered by task description and converted to the same Zarr layout.

Both feed [diffusion_policy](https://github.com/columbia-ai-robotics/diffusion_policy) training (MLP / Diffusion U-Net), which is then evaluated back in SIMPLER, optionally with a Best-of-N action verifier.

---

## 0. Prerequisites

One-time environment setup (creates the `simpler_env`, `sglang-vla`, and `monkey-verifier` conda envs):

```bash
bash scripts/setup.sh
```

This README assumes:

- `~/miniconda3` exists with the three envs above.
- The `diffusion_policy` repo is cloned at `~/diffusion_policy` (override with `DIFFUSION_POLICY_ROOT`).

---

## 1. Synthetic data collection (SIMPLER + OpenVLA)

### 1a. Launch the OpenVLA action server

The collector talks to a long-running sglang OpenVLA server on port `3200`.

```bash
# terminal A — leave running
bash scripts/run_openvla_server.sh
# overrides:
#   CUDA_VISIBLE_DEVICES=1 SEED=42 bash scripts/run_openvla_server.sh
```

First launch JIT-compiles FlashInfer kernels (~60–90 s); cached in `~/.cache/flashinfer/`.

### 1b. Collect rollouts → Zarr shard

```bash
# terminal B — carrot on plate (10k trajectories, single shard `state0.zarr`)
bash scripts/collect_carrot_on_plate.sh

# eggplant in basket (10k trajectories)
bash scripts/collect_eggplant_in_basket.sh

# sharded usage (start_index, num, shard_name):
bash scripts/collect_carrot_on_plate.sh 0    5000 state0.zarr
bash scripts/collect_carrot_on_plate.sh 5000 5000 state1.zarr

# Optional: also persist the per-step agentview RGB into the zarr.
# Used by the state-image L2S verifier path so training does not have to
# re-render images via SIMPLER. Adds a uint8 (T, H, W, 3) array under
# data/obs/agentview_image.
SAVE_IMAGES=True bash scripts/collect_eggplant_in_basket.sh
```

Output locations:

- Carrot: [openvla-mini/data/carrot_on_plate/state0.zarr](openvla-mini/data/carrot_on_plate/state0.zarr)
- Eggplant: [openvla-mini/data/eggplant_in_basket/state0.zarr](openvla-mini/data/eggplant_in_basket/state0.zarr)
- Per-run logs land alongside the shard as `collect-<task>-<timestamp>.log`.

### 1c. Sanity-check a shard

```bash
python scripts/zarr_dataset_dashboard.py \
    openvla-mini/data/carrot_on_plate/state0.zarr
```

Writes `state0_dashboard.png`, `state0_start_positions.png`, and (if applicable) `state0_success_episodes.txt` next to the shard.

---

## 2. Real data — Bridge V2 filtering procedure

Bridge V2 ([`jesbu1/bridge_v2_lerobot`](https://huggingface.co/datasets/jesbu1/bridge_v2_lerobot)) bundles **53,192 episodes across 19,974 freeform language tasks**. We download only the carrot+plate and eggplant+basket subsets and convert them to our Zarr layout.

### 2a. How the filter works ([scriptsv2/filter_bridge_v2.py](scriptsv2/filter_bridge_v2.py))

The script takes "task groups" of the form `name=kw1,kw2,...`. A Bridge task description matches a group iff **all** keywords appear (case-insensitive) and **none** of the `--exclude` phrases do.

Steps the script performs:

1. Downloads only `meta/info.json`, `meta/tasks.jsonl`, and `meta/episodes.jsonl` from the HF repo.
2. Scans `tasks.jsonl` for descriptions matching each group's keywords (minus excludes).
3. Resolves the matching `episode_indices` from `episodes.jsonl`.
4. Selectively downloads only those episodes' `.parquet` data + `image_*.mp4` videos, preserving the upstream `chunk-XXX/episode_YYYYYY.*` layout.
5. Writes a filtered `meta/tasks.jsonl`, `meta/episodes.jsonl`, updated `meta/info.json`, and a per-group `filter_summary.json`.

Defaults baked into [scriptsv2/download_carrot_and_eggplant.sh](scriptsv2/download_carrot_and_eggplant.sh):

| group              | keywords          | excludes              | result                                   |
| ------------------ | ----------------- | --------------------- | ---------------------------------------- |
| `carrot_on_plate`  | `carrot, plate`   | `take carrot off`     | task 184 "put carrot on plate", 332 eps  |
| `eggplant_in_basket` | `eggplant, basket` | (none)              | matches "put eggplant in basket" tasks   |

> **Note on eggplant:** the Bridge task list does not contain the literal token "basket" near every eggplant pick-and-place. If a group matches 0 tasks the script warns loudly. Override the keyword set via the `EGGPLANT_GROUP` env var, e.g.
> ```bash
> EGGPLANT_GROUP="eggplant_in_pot=eggplant,pot" bash scriptsv2/download_carrot_and_eggplant.sh
> ```

### 2b. Run the filter + selective download

```bash
# Inspect matches without downloading parquet/mp4:
DRY_RUN=1 bash scriptsv2/download_carrot_and_eggplant.sh

# Real download (default OUT_DIR=data/bridge_v2_filtered):
bash scriptsv2/download_carrot_and_eggplant.sh

# Cap episodes per group (useful for quick iteration):
MAX_EPISODES_PER_GROUP=20 bash scriptsv2/download_carrot_and_eggplant.sh
```

Output location: [data/bridge_v2_filtered/](data/bridge_v2_filtered/) with subdirs `meta/`, `data/chunk-XXX/...parquet`, `videos/chunk-XXX/.../*.mp4`, plus `filter_summary.json` summarizing per-group counts.

### 2c. Optional: visualize the filtered subset

```bash
python scriptsv2/plot_bridge_v2_dashboard.py \
    --bridge_dir data/bridge_v2_filtered
```

### 2d. Convert filtered Bridge → SIMPLER-style Zarr ([scriptsv2/bridge_to_zarr.py](scriptsv2/bridge_to_zarr.py))

The Zarr written here mirrors the synthetic collector's layout (same keys under `data/obs/*`, `data/actions`, `meta/episode_ends`) so downstream training and dashboards work unchanged. Bridge `observation.state = [x, y, z, roll, pitch, yaw, gripper]` is converted to `(x, y, z, qx, qy, qz, qw)` via extrinsic XYZ, with linear/angular EE velocities computed by finite difference and missing fields zero-filled.

```bash
# carrot
python scriptsv2/bridge_to_zarr.py \
    --bridge_dir data/bridge_v2_filtered \
    --task_filter "put carrot on plate" \
    --out openvla-mini/data/carrot_on_plate/bridge_v2_carrot.zarr

# eggplant
python scriptsv2/bridge_to_zarr.py \
    --bridge_dir data/bridge_v2_filtered \
    --task_filter "put eggplant in basket" \
    --out openvla-mini/data/eggplant_in_basket/bridge_v2_eggplant.zarr
```

Output locations:

- [openvla-mini/data/carrot_on_plate/bridge_v2_carrot.zarr](openvla-mini/data/carrot_on_plate/bridge_v2_carrot.zarr)
- [openvla-mini/data/eggplant_in_basket/bridge_v2_eggplant.zarr](openvla-mini/data/eggplant_in_basket/bridge_v2_eggplant.zarr)

---

## 3. Train a diffusion_policy lowdim policy

Training itself lives in the sibling `~/diffusion_policy` repo. Configs of interest:

- `bridge_v2_carrot_lowdim` (carrot)
- `eggplant_in_basket_lowdim` (eggplant)

Each config is trained twice — once as `train_mlp_*` and once as `train_diffusion_unet_*`. Checkpoints land under:

```
~/diffusion_policy/data/outputs/<YYYY.MM.DD>/<HH.MM.SS_train_<arch>_<config>>/checkpoints/latest.ckpt
```

Currently used checkpoints:

| task     | arch        | checkpoint                                                                                                                          |
| -------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| carrot   | Diffusion U-Net | `~/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt` |
| carrot   | MLP         | `~/diffusion_policy/data/outputs/2026.04.28/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt` |
| eggplant | Diffusion U-Net | `~/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt` |
| eggplant | MLP         | `~/diffusion_policy/data/outputs/2026.04.29/16.40.54_train_mlp_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt` |

---

## 3.1 Train the L2S search policy (eggplant, state-based)

The state-based RoboMonkey search policy lives in
`diffusion_policy/diffusion_policy/policy/search_policy_robomonkey.py` and is
configured by
[diffusion_policy/diffusion_policy/config/robomonkey_eggplant_search_state.yaml](file:///home/harine/diffusion_policy/diffusion_policy/config/robomonkey_eggplant_search_state.yaml).

The verifier needs the monkey-verifier HTTP server running on port 3100:

```bash
# terminal A — leave running
conda activate monkey-verifier
cd /home/harine/RoboMonkey/monkey-verifier/src
python infer_server.py            # 0.0.0.0:3100
```

`policy.corrupt_obs` toggles the DDPM-style obs-feature noising. The flag is
baked into the run name (and therefore the saved hydra config + wandb run),
so both runs land in distinct output dirs:

```bash
cd ~/diffusion_policy

# Noised search policy (obs-feature DDPM noise on)
# → name = robomonkey_eggplant_search_state_corrupt
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=True

# Vanilla search policy (no obs corruption)
# → name = robomonkey_eggplant_search_state_clean
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=False
```

Output dir: `~/diffusion_policy/data/outputs/<YYYY.MM.DD>/<HH.MM.SS_robomonkey_eggplant_search_state_{corrupt,clean}_<task_name>>/`
(contains `checkpoints/`, hydra `config.yaml`, and wandb run metadata).

Note: the current verifier (`RoboMonkeyStateVerifier`) renders an RGB frame
on the fly from the saved state via SIMPLER. Once you re-collect the zarr
with `SAVE_IMAGES=True bash scripts/collect_eggplant_in_basket.sh` (see §1b),
you can swap `policy.verifier._target_` to the plain `RoboMonkeyVerifier`
and feed `agentview_image` straight from the dataset for a much faster
training step.

---

## 4. Evaluate the policy in SIMPLER

The single-checkpoint runner is [scriptsv2/run_eval_diffusion_policy.sh](scriptsv2/run_eval_diffusion_policy.sh). It accepts the task via the `TASK` env var.

### 4a. Carrot — single checkpoint

```bash
TASK=widowx_carrot_on_plate \
bash scriptsv2/run_eval_diffusion_policy.sh \
    ~/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt \
    100
```

### 4b. Eggplant — single checkpoint

```bash
TASK=widowx_put_eggplant_in_basket \
bash scriptsv2/run_eval_diffusion_policy.sh \
    ~/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt \
    100
```

### 4c. Both architectures at once

```bash
# carrot (MLP + U-Net, 100 episodes each)
bash scriptsv2/run_eval_both_architectures.sh

# eggplant (MLP + U-Net, 100 episodes each)
bash scriptsv2/run_eval_both_eggplant.sh
```

### 4d. Best-of-N with the action verifier

Spin up the verifier first (this is the same `infer_server.py` referenced in the main README):

```bash
# terminal C — leave running
conda activate monkey-verifier
cd monkey-verifier/src
python infer_server.py        # default port 3100
```

Then re-run the eval with `BON_K > 1`:

```bash
TASK=widowx_put_eggplant_in_basket \
BON_K=8 \
BON_REPLAN_EVERY_N_STEPS=0 \
BON_SCORE_NUM_ACTIONS=1 \
REWARD_SERVER_PORT=3100 \
REWARD_BATCH_SIZE=16 \
bash scriptsv2/run_eval_diffusion_policy.sh \
    ~/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt \
    100
```

Sweeps over `BON_K`:

```bash
bash scriptsv2/run_bon_mlp_unet_sweep.sh
bash scriptsv2/run_bon_ordered_sweep.sh
```

### 4e. Output locations

Each eval writes to `data/eval/<run_name>/`, where `<run_name>` is the basename of the checkpoint's `outputs/<date>/<run>` directory. Existing runs:

- [data/eval/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/](data/eval/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/) — carrot U-Net
- [data/eval/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/](data/eval/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/) — carrot MLP
- [data/eval/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/](data/eval/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/) — eggplant U-Net
- [data/eval/16.40.54_train_mlp_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/](data/eval/16.40.54_train_mlp_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/) — eggplant MLP
- [data/eval/bon/](data/eval/bon/) — Best-of-N sweep results
- [data/eval/variance_eggplant_unet/](data/eval/variance_eggplant_unet/) — action-variance study
- [data/eval/videos_eggplant_unet/](data/eval/videos_eggplant_unet/) — saved MP4 rollouts

Each run dir contains `eval_log.json` (per-episode results) and `episodes.jsonl`. Summarize:

```bash
python scriptsv2/summarize_evals.py data/eval/<run_name>/eval_log.json
```

---

## 5. Quick-reference data map

| Stage | Path | Produced by |
| --- | --- | --- |
| Filtered Bridge V2 (parquet + mp4 + meta) | [data/bridge_v2_filtered/](data/bridge_v2_filtered/) | `scriptsv2/download_carrot_and_eggplant.sh` |
| Filter summary (matched tasks/episodes) | [data/bridge_v2_filtered/filter_summary.json](data/bridge_v2_filtered/filter_summary.json) | `scriptsv2/filter_bridge_v2.py` |
| Synthetic carrot Zarr | [openvla-mini/data/carrot_on_plate/state0.zarr](openvla-mini/data/carrot_on_plate/state0.zarr) | `scripts/collect_carrot_on_plate.sh` |
| Synthetic eggplant Zarr | [openvla-mini/data/eggplant_in_basket/state0.zarr](openvla-mini/data/eggplant_in_basket/state0.zarr) | `scripts/collect_eggplant_in_basket.sh` |
| Bridge → carrot Zarr | [openvla-mini/data/carrot_on_plate/bridge_v2_carrot.zarr](openvla-mini/data/carrot_on_plate/bridge_v2_carrot.zarr) | `scriptsv2/bridge_to_zarr.py` |
| Bridge → eggplant Zarr | [openvla-mini/data/eggplant_in_basket/bridge_v2_eggplant.zarr](openvla-mini/data/eggplant_in_basket/bridge_v2_eggplant.zarr) | `scriptsv2/bridge_to_zarr.py` |
| Trained checkpoints | `~/diffusion_policy/data/outputs/<date>/<run>/checkpoints/latest.ckpt` | `diffusion_policy` repo |
| SIMPLER eval logs | [data/eval/&lt;run_name&gt;/](data/eval/) | `scriptsv2/run_eval_diffusion_policy.sh` |
