# README_L2S — Learning-to-Sample data collection & policy evaluation

End-to-end recipe for the two SIMPLER tasks we use for L2S work:

- **`widowx_carrot_on_plate`** — "put carrot on plate"
- **`widowx_put_eggplant_in_basket`** — "put eggplant in basket"

Two data sources feed the same downstream Zarr layout:

1. **Synthetic** — OpenVLA rollouts in SIMPLER, written directly to a Zarr shard.
2. **Real** — Bridge V2 LeRobot subset, filtered by task description and converted to the same Zarr layout.

Both feed [diffusion_policy](https://github.com/columbia-ai-robotics/diffusion_policy) training (MLP / Diffusion U-Net), which is then evaluated back in SIMPLER, optionally with a Best-of-N action verifier.

> All pipeline scripts live under [scriptsv2/](scriptsv2/), organized into
> `bridge_to_zarr/`, `plot_zarr/`, `eval_diffusion/`, `bon_eval/`,
> `action_variance/`. See **[scriptsv2/README.md](scriptsv2/README.md)** for a
> per-script index. The `scripts/` directory holds only env / setup / VLA
> rollout collection.

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

- Carrot: `~/data/carrot_on_plate/state_only/state0.zarr`
- Eggplant: `~/data/eggplant_in_basket/state_only/state0.zarr`
- Per-run logs land alongside the shard as `collect-<task>-<timestamp>.log`.

### 1c. Sanity-check a shard

```bash
python scriptsv2/plot_zarr/plot_zarr.py \
    --shard ~/data/carrot_on_plate/state_only/state0.zarr
```

Writes `state0_dashboard.png`, `state0_start_positions.png`, and `state0_scatters.png` next to the shard. Multiple `--shard` arguments are merged into a single combined figure.

---

## 2. Real data — Bridge V2 filtering procedure

Bridge V2 ([`jesbu1/bridge_v2_lerobot`](https://huggingface.co/datasets/jesbu1/bridge_v2_lerobot)) bundles **53,192 episodes across 19,974 freeform language tasks**. We download only the carrot+plate and eggplant+basket subsets and convert them to our Zarr layout.

### 2a. How the filter works ([scriptsv2/bridge_to_zarr/filter_bridge_v2.py](scriptsv2/bridge_to_zarr/filter_bridge_v2.py))

The script takes "task groups" of the form `name=kw1,kw2,...`. A Bridge task description matches a group iff **all** keywords appear (case-insensitive) and **none** of the `--exclude` phrases do.

Steps the script performs:

1. Downloads only `meta/info.json`, `meta/tasks.jsonl`, and `meta/episodes.jsonl` from the HF repo.
2. Scans `tasks.jsonl` for descriptions matching each group's keywords (minus excludes).
3. Resolves the matching `episode_indices` from `episodes.jsonl`.
4. Selectively downloads only those episodes' `.parquet` data + `image_*.mp4` videos, preserving the upstream `chunk-XXX/episode_YYYYYY.*` layout.
5. Writes a filtered `meta/tasks.jsonl`, `meta/episodes.jsonl`, updated `meta/info.json`, and a per-group `filter_summary.json`.

Defaults baked into [scriptsv2/bridge_to_zarr/bridge_to_zarr.sh](scriptsv2/bridge_to_zarr/bridge_to_zarr.sh):

| group              | keywords          | excludes              | result                                   |
| ------------------ | ----------------- | --------------------- | ---------------------------------------- |
| `carrot_on_plate`  | `carrot, plate`   | `take carrot off`     | task 184 "put carrot on plate", 332 eps  |
| `eggplant_in_basket` | `eggplant, basket` | (none)              | matches "put eggplant in basket" tasks   |

> **Note on eggplant:** the Bridge task list does not contain the literal token "basket" near every eggplant pick-and-place. If a group matches 0 tasks the script warns loudly. Override the keyword set via the `EGGPLANT_GROUP` env var, e.g.
> ```bash
> EGGPLANT_GROUP="eggplant_in_pot=eggplant,pot" bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh
> ```

### 2b. Run the filter + download + convert (one-shot)

```bash
# Inspect matches without downloading parquet/mp4:
DRY_RUN=1 bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh

# Full pipeline: download filtered subset AND convert to zarr.
bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh

# Cap episodes per group (useful for quick iteration):
MAX_EPISODES_PER_GROUP=20 bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh

# Filter+download only, skip zarr conversion:
SKIP_CONVERT=1 bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh
```

Output locations after a full run:
- Filtered LeRobot subset: [data/bridge_v2_filtered/](data/bridge_v2_filtered/) (meta + parquet + mp4 + `filter_summary.json`)
- Zarr shards: `~/data/carrot_on_plate/state_only/bridge_v2_carrot.zarr` and `~/data/eggplant_in_basket/state_only/bridge_v2_eggplant.zarr`

### 2c. Optional: visualize the filtered subset *before* converting

If you want to preview the raw LeRobot dataset (with mp4 thumbnails) before running the zarr conversion:

```bash
python scriptsv2/plot_zarr/plot_bridge_lerobot.py \
    --data_dir data/bridge_v2_filtered
```

After conversion, use `scriptsv2/plot_zarr/plot_zarr.py` on the zarr output instead (cleaner / supports multi-shard overlay).

### 2d. Convert filtered Bridge → SIMPLER-style Zarr ([scriptsv2/bridge_to_zarr/bridge_to_zarr.py](scriptsv2/bridge_to_zarr/bridge_to_zarr.py))

This is run automatically by `bridge_to_zarr.sh` above. The Zarr written mirrors the synthetic collector's layout (same keys under `data/obs/*`, `data/actions`, `meta/episode_ends`) so downstream training and dashboards work unchanged. Bridge `observation.state = [x, y, z, roll, pitch, yaw, gripper]` is converted to `(x, y, z, qx, qy, qz, qw)` via extrinsic XYZ, with linear/angular EE velocities computed by finite difference and missing fields zero-filled.

To run the convert stage manually (e.g. for a custom task filter):

```bash
python scriptsv2/bridge_to_zarr/bridge_to_zarr.py \
    --bridge_dir data/bridge_v2_filtered \
    --task_filter "put carrot on plate" \
    --out ~/data/carrot_on_plate/state_only/bridge_v2_carrot.zarr
```

Output locations:

- `~/data/carrot_on_plate/state_only/bridge_v2_carrot.zarr`
- `~/data/eggplant_in_basket/state_only/bridge_v2_eggplant.zarr`

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

The verifier has two backends, selected by `policy.verifier.server_url`:

- **In-process (recommended)** — `policy.verifier.server_url=in_process`.
  The training process loads `RobotRewardModel` directly and scores
  (image, action) pairs in batched GPU forwards. No HTTP, no disk JPEGs,
  no SIMPLER re-rendering — the dataset's `agentview_image` is fed
  straight to the verifier. Requires the verifier deps importable in
  the training env (easiest: train inside the `monkey-verifier` env, or
  set `MONKEY_VERIFIER_SRC=/path/to/RoboMonkey/monkey-verifier/src`).
  Routes through the shared client at
  [monkey-verifier/src/verifier_client.py](monkey-verifier/src/verifier_client.py).
- **HTTP server** — leave `server_url=http://127.0.0.1:3100` (default) and
  start `infer_server.py` in a separate terminal. Slower, kept for the
  case where the training and verifier envs are deliberately disjoint.

`policy.corrupt_obs` toggles the DDPM-style obs-feature noising. The flag is
baked into the run name (and therefore the saved hydra config + wandb run),
so both runs land in distinct output dirs.

### 3.1.a One-time `monkey-verifier` env setup (for in-process)

The base `monkey-verifier` env is missing diffusion_policy's runtime deps.
[monkey-verifier/l2s_inprocess_env.yaml](monkey-verifier/l2s_inprocess_env.yaml)
collects every extra install we've hit (hydra-core, dill, diffusers, wandb,
numba, bitsandbytes pin, …). Run once:

```bash
conda env update -n monkey-verifier -f monkey-verifier/l2s_inprocess_env.yaml

# Plus the one-off CUDA mismatch fix that doesn't fit cleanly in conda yaml
# (monkey-verifier ships torch+cu117 but pip installs torchvision+cu118):
pip install --force-reinstall --no-deps \
    torchvision==0.15.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
```

Append any further `ModuleNotFoundError`s to that yaml as you encounter them.

### 3.1.b Launch (in-process)

```bash
conda activate monkey-verifier
cd ~/diffusion_policy

# Recommended starting point — fits a 32 GB GPU with headroom.
ROBOMONKEY_PAIRED_CHUNK_SIZE=16 \
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=True \
    policy.verifier.server_url=in_process \
    dataloader.batch_size=32 val_dataloader.batch_size=32 \
    policy.max_actions=8

# Vanilla search policy (no obs corruption) — same overrides.
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=False \
    policy.verifier.server_url=in_process \
    dataloader.batch_size=32 val_dataloader.batch_size=32 \
    policy.max_actions=8
```

Output dir: `~/diffusion_policy/data/outputs/<YYYY.MM.DD>/<HH.MM.SS_robomonkey_eggplant_search_state_{corrupt,clean}_<task_name>>/`
(contains `checkpoints/`, hydra `config.yaml`, and wandb run metadata).

### 3.1.c Throughput knobs

The verifier dominates each training step: per step it does
`(max_actions − 1)` calls × `batch_size` (image, action) pairs through
LLaVA-7B. Tune these in order of impact:

| Knob | Default | What it does |
| --- | --- | --- |
| `dataloader.batch_size` | 256 (yaml) → use **32** | Direct linear factor on verifier calls per step. |
| `policy.max_actions` | 16 (yaml) → use **8** | Linear factor on verifier calls per step. |
| `ROBOMONKEY_PAIRED_CHUNK_SIZE` env | 4 → use **16** | Rows per GPU forward inside one verifier call. Higher = better GPU utilization, more peak memory. Drop to 2/1 if OOM, raise to 24/32 if you have headroom. |
| `ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE` env | 128 | LRU of CLIP image features (~4.7 MB/image on GPU). Within one training step the same 32 batch images are reused for all `max_actions − 1` verifier calls, so this cuts CLIP forwards ~7×. |
| `ROBOMONKEY_REWARD_CACHE_SIZE` env | 100000 | LRU of `(instruction, image, action_tokens) → reward`. Mostly helps converged-policy training and BoN sweeps where the same query repeats. Set to `0` to disable. |

Watch the first few steps in `nvidia-smi`; if you're well under 32 GB,
double the chunk size. Going from the yaml defaults
(`batch_size=256, max_actions=16, chunk_size=4`) to
(`batch_size=32, max_actions=8, chunk_size=16`) is roughly a 17×
end-to-end speedup before caches.

### 3.1.d HTTP server path (legacy)

```bash
# terminal A — leave running
conda activate monkey-verifier
cd /home/harine/RoboMonkey/monkey-verifier/src
python infer_server.py            # 0.0.0.0:3100
```

```bash
cd ~/diffusion_policy

python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=True

python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=False
```

---

## 4. Evaluate the policy in SIMPLER

The single-checkpoint runner is [scriptsv2/eval_diffusion/eval_diffusion.sh](scriptsv2/eval_diffusion/eval_diffusion.sh). It accepts the task via the `TASK` env var. See [scriptsv2/README.md](scriptsv2/README.md) for the full per-script index.

### 4a. Carrot — single checkpoint

```bash
TASK=widowx_carrot_on_plate \
bash scriptsv2/eval_diffusion/eval_diffusion.sh \
    ~/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt \
    100
```

### 4b. Eggplant — single checkpoint

```bash
TASK=widowx_put_eggplant_in_basket \
bash scriptsv2/eval_diffusion/eval_diffusion.sh \
    ~/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt \
    100
```

### 4c. Both architectures at once

Single wrapper covers both tasks; `TASK` short names are accepted (`carrot` / `eggplant`, default eggplant):

```bash
# eggplant — MLP + U-Net, 100 episodes each
bash scriptsv2/eval_diffusion/eval_diffusion_both.sh

# carrot
TASK=carrot bash scriptsv2/eval_diffusion/eval_diffusion_both.sh

# custom checkpoints, custom episode count
bash scriptsv2/eval_diffusion/eval_diffusion_both.sh <unet_ckpt> <mlp_ckpt> 50
```

### 4d. Best-of-N with the action verifier

Two ways to run the verifier:

- **In-process (recommended for `eval_diffusion`)** — `REWARD_SERVER_PORT=0`.
  The eval script loads `RobotRewardModel` directly into the eval process and
  scores all K candidates in a single GPU forward. No HTTP, no disk image
  hop, the CLIP vision tower runs once per step (not K times), the prompt
  template is tokenized once per instruction, and the LLaMA prefix is
  prefilled once with a shared KV cache reused across candidates. A
  `(instruction, image, action_tokens) → reward` LRU
  (`ROBOMONKEY_REWARD_CACHE_SIZE`, default 100k) further short-circuits
  exact repeats across rollouts. Requires the verifier deps (llava, peft,
  bitsandbytes, …) importable in the eval env — easiest is to run inside
  the `monkey-verifier` env (see §3.1.a for the one-time setup), or
  pip-install the missing deps into `simpler_env`.

- **HTTP server** — leave `REWARD_SERVER_PORT=3100` (default) and start
  `infer_server.py` in a separate terminal as before. Use this when the eval
  env can't import the verifier directly (e.g. shared GPU, separate Python
  envs, or the existing search-policy training path in §3.1 that talks
  to the server over HTTP).

In-process (single GPU, no extra terminal):

```bash
TASK=widowx_put_eggplant_in_basket \
BON_K=8 \
BON_REPLAN_EVERY_N_STEPS=0 \
BON_SCORE_NUM_ACTIONS=1 \
REWARD_SERVER_PORT=0 \
CONDA_ENV=monkey-verifier \
bash scriptsv2/eval_diffusion/eval_diffusion.sh \
    ~/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt \
    100
```

HTTP server path (legacy):

```bash
# terminal C — leave running
conda activate monkey-verifier
cd monkey-verifier/src
python infer_server.py        # default port 3100
```

```bash
TASK=widowx_put_eggplant_in_basket \
BON_K=8 \
BON_REPLAN_EVERY_N_STEPS=0 \
BON_SCORE_NUM_ACTIONS=1 \
REWARD_SERVER_PORT=3100 \
REWARD_BATCH_SIZE=16 \
bash scriptsv2/eval_diffusion/eval_diffusion.sh \
    ~/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt \
    100
```

Full ordered sweep over `BON_K ∈ {2,4,8,16,32,64}` for both architectures with resume support. `bon_eval.sh` forwards `REWARD_SERVER_PORT` and `CONDA_ENV` to every inner `eval_diffusion.sh` call, so the in-process toggle works identically here:

```bash
# In-process verifier (recommended) — fastest, no extra terminal.
REWARD_SERVER_PORT=0 CONDA_ENV=monkey-verifier \
    bash scriptsv2/bon_eval/bon_eval.sh             # full sweep, 100 eps per cell

# HTTP path (requires `infer_server.py` running in another terminal):
bash scriptsv2/bon_eval/bon_eval.sh             # full sweep, 100 eps per cell
bash scriptsv2/bon_eval/bon_eval.sh 5           # smoke test, 5 eps per cell
ONLY_K="15" bash scriptsv2/bon_eval/bon_eval.sh # re-run just k=15
```

### 4e. Action variance study (optional)

To measure per-step action variance for both architectures (motivates BON):

```bash
bash scriptsv2/action_variance/analyze_variance.sh
N_SAMPLES=64 bash scriptsv2/action_variance/analyze_variance.sh
```

### 4f. Output locations

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
python scriptsv2/eval_diffusion/eval_summary.py data/eval/<run_name>/eval_log.json
```

---

## 5. Quick-reference data map

| Stage | Path | Produced by |
| --- | --- | --- |
| Filtered Bridge V2 (parquet + mp4 + meta) | [data/bridge_v2_filtered/](data/bridge_v2_filtered/) | `scriptsv2/bridge_to_zarr/bridge_to_zarr.sh` |
| Filter summary (matched tasks/episodes) | [data/bridge_v2_filtered/filter_summary.json](data/bridge_v2_filtered/filter_summary.json) | `scriptsv2/bridge_to_zarr/filter_bridge_v2.py` |
| Synthetic carrot Zarr (state-only) | `~/data/carrot_on_plate/state_only/state0.zarr` | `scripts/collect_carrot_on_plate.sh` |
| Synthetic eggplant Zarr (state-only) | `~/data/eggplant_in_basket/state_only/state0.zarr` | `scripts/collect_eggplant_in_basket.sh` |
| Bridge → carrot Zarr | `~/data/carrot_on_plate/state_only/bridge_v2_carrot.zarr` | `scriptsv2/bridge_to_zarr/bridge_to_zarr.py` |
| Bridge → eggplant Zarr | `~/data/eggplant_in_basket/state_only/bridge_v2_eggplant.zarr` | `scriptsv2/bridge_to_zarr/bridge_to_zarr.py` |
| Trained checkpoints | `~/diffusion_policy/data/outputs/<date>/<run>/checkpoints/latest.ckpt` | `diffusion_policy` repo |
| SIMPLER eval logs | [data/eval/&lt;run_name&gt;/](data/eval/) | `scriptsv2/eval_diffusion/eval_diffusion.sh` |
| BON sweep logs | [data/eval/bon/](data/eval/bon/) | `scriptsv2/bon_eval/bon_eval.sh` |
| Action-variance logs | [data/eval/variance_eggplant_*/](data/eval/) | `scriptsv2/action_variance/analyze_variance.sh` |
