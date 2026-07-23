# Eggplant policy checkpoint registry

Identifying info for the eggplant (`widowx_put_eggplant_in_basket`) policy checkpoints.

**All policies (BC, gaussian search, diffusion search) are data-matched** — same loader
(`Sim2RealImageMultiDataset`), same shards `state0.zarr` (726 ep) + `state1.zarr` (316 ep)
= **1042 successful episodes** (`success_only=true`; all already successful), same split
`val_ratio=0.05`, `seed=42` (~990 train / ~52 val). The BC/gaussian reference this as
`/gscratch/robotics/harine/data/eggplant_in_basket_complete` (a symlink dir → those two
`eggplant_in_basket/state{0,1}.zarr` shards, 391 GB); the diffusion search recorded the
off-cluster path `/home/harine/data/eggplant_in_basket` (~51 GB) but the **same two shards**
(per training notes) — same demos, just trained elsewhere and copied.

All are ordinary conditional-diffusion / gaussian **BC** on the successful chunks (epsilon
prediction). The **search/verifier is applied at INFERENCE only** (sample `max_actions`
candidate chunks, the DQC chunk critic picks the best) — NOT in the training loss.

Last updated: 2026-06-19.

> **ARCHIVED 2026-07-01:** all policies in this doc (BC jun05/11/12, gaussian search,
> diffusion search + its pathfix eval ckpts) were trained on the UNFILTERED success set and
> have been moved to `/gscratch/robotics/harine/unfiltered_success_policies/{bc,gaussian_search,diffusion_search}/`
> (see its README). The paths below are their ORIGINAL (pre-move) locations. They are
> superseded by the filtered (rmse_filtered / sub100) policies under
> `/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered/`. `launch_ft_verifier_bon.sh`
> no longer hardcodes any policy — it requires SEARCH_POLICIES/BC_CKPT and errors otherwise.

## BC policies (`DiffusionUnetImagePolicy`, state obs via FlattenObsEncoder)

| Label | Trained | Horizon | n_obs | rollout (n_action) | Latest ckpt | Size | Data | Notes |
|---|---|---|---|---|---|---|---|---|
| **BC-H16-jun05a** | 2026-06-05 11:20 | 16 | 2 | 8 | `step_0007000.ckpt` | — | complete (val .05/seed 42) | early |
| **BC-H16-jun05b** | 2026-06-05 15:29 | 16 | 2 | 8 | `best_ep55_val0.104.ckpt` | — | complete | |
| **BC-H16-jun11** | 2026-06-11 14:39 | 16 | 2 | 8 | `best.ckpt` | ~1 GB | complete | ⭐ **canonical** — used in ALL BoN sweeps (task 3 BC+LLaVA, task 4 BC+DQC) |
| **BC-H8-jun12** | 2026-06-12 15:04 | **8** | 2 | **4** | `best.ckpt` | — | complete | ⭐ **H8 baseline** — matches search policies' horizon-8 / rollout-4 |

Paths (under `diffusion_policy/data/outputs/<date>/<time>_robomonkey_eggplant_bc_robomonkey_eggplant_state/checkpoints/`):
- BC-H16-jun05a: `2026.06.05/10.22.28_...`
- BC-H16-jun05b: `2026.06.05/13.11.48_...`
- BC-H16-jun11:  `2026.06.11/13.52.09_.../checkpoints/best.ckpt`
- BC-H8-jun12:   `2026.06.12/14.35.54_.../checkpoints/best.ckpt`

## Search policies (BC base + DQC chunk verifier at inference; verifier cfg baked in)

Both are conditional/gaussian BC (search is inference-only). n_obs=1, pad_before=0 → they
predict a clean 8-step chunk a_{t:t+8} from the current frame (matches the chunk critic).
Baked verifier: DQC `critic_type=chunk`, `q_agg=min`, `score_full_horizon=true`, ckpt
`sd000s_137913.0.20260605_163017`, USE-3 task emb.

| Label | Trained | Policy | Horizon | n_obs | rollout | Latest ckpt | Data | Notes |
|---|---|---|---|---|---|---|---|---|
| **SEARCH-gaussian-H8** | 2026-06-19 (in progress) | `SearchPolicyRoboMonkey` | 8 | 1 | 4 | `step_0013000.ckpt` (→50k) | eggplant_in_basket_complete ✅ data-matched | task-2 training; gaussian per-token |
| **SEARCH-diffusion-H8** | 2026-06-18 (off-cluster, copied) | `SearchPolicyRoboMonkeyDiffusion` | 8 | 1 | 4 | `step_0033001.ckpt` | `/home/harine/data/eggplant_in_basket` (same state0+state1 shards as BC, off-cluster) | DDIM-8 inference (8 steps); state-conditioned; verifier paths patched (below) |

Paths:
- SEARCH-gaussian-H8: `/gscratch/robotics/harine/RoboMonkey_data/train/dqc_chunk/eggplant_gaussian_dqc_chunk8_hyak_predhorizon8_rollout4/checkpoints/`
- SEARCH-diffusion-H8 (original): `data/models/2026.06.18/17.17.07_eggplant_diffusion_dqc_h8_roll4_robomonkey_eggplant_state/checkpoints/step_0033001.ckpt`
- SEARCH-diffusion-H8 (**path-fixed for eval**): `/gscratch/robotics/harine/RoboMonkey_data/eval_ckpts/diffusion_dqc_h8_roll4_step33001_pathfix.ckpt` — baked verifier `ckpt_dir`/`text_emb_file` rewritten from non-existent `/home/harine/...` to valid `/gscratch/.../dqc_checkpoint/sd000s_137913.0.20260605_163017` + `/mmfs1/home/harine/.../task_emb_use3.npy` (verifier semantics unchanged: critic_type=chunk, score_full_horizon=True, q_agg=min).

## Eval results (eggplant, 50 ep/cell, seeds 1000–1049)
- **SEARCH-diffusion-H8** BoN curve: N1=94 N2=88 N4=90 N8=82 N16=86 N32=82 N64=74 (128/256/512 running) — *declining*; N=1 best (verifier doesn't help this strong policy).
- **SEARCH-gaussian-H8** (step-11k): N4=12%, N16=10% (early checkpoint).
- **BC-H16-jun11** + DQC chunk (task 4): n1=42 n2=42 n4=48 n8=40 n16=46 n32=66 n64=44 n128=52 n256=48 (n512 running).
- **BC-H16-jun11** base sampler (n=1): DDIM=42%, DDPM=52%.
