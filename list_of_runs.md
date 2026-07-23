# List of Runs — training + eval (GPU assignments)

Snapshot: 2026-07-07 15:58 — **4 of 6 new-task trainings ✅ DONE @100k** (all 3 stack + spoon_sub50, COMPLETED 14:59). spoon_sub100 @98k + spoon_sub25 @91k still climbing (weirdlab gpu-l40); eggplant (robotics) at 88–91k. Evals: eggplant sub50 90k in flight (2 cells R), 0 PD — nothing stuck. stack_sub100 peak @50k=56% (90k=52); spoon_sub100 54–60% (90k=56); spoon_sub25 plateau low-40s (80k=44). Search + BC-BoN Q-videos organized (bc/=15 each). Job IDs/nodes transient; **partition routing** is the stable part. Companion to [list_of_configs.md](list_of_configs.md). Auto-refreshed hourly (:48).

---

## ⭐ CURRENT RUNS (dashboard)

### A. Search-policy trainings → 100k  (9 runs)
| Run | Task | Step | Status (detail) | GPU · node |
|---|---|---|---|---|
| stack_sub100 | stack_cube | 100k | ✅ done @100k (COMPLETED) | — |
| stack_sub50 | stack_cube | 100k | ✅ done @100k (COMPLETED) | — |
| stack_sub25 | stack_cube | 100k | ✅ done @100k (COMPLETED) | — |
| spoon_sub100 | spoon_on_towel | 94k | ▶ R · training 94k→100k | weirdlab gpu-l40 · g3115 |
| spoon_sub50 | spoon_on_towel | 97k | ▶ R · training 97k→100k | weirdlab gpu-l40 · g3115 |
| spoon_sub25 | spoon_on_towel | 88k | ▶ R · training 88k→100k (⚠ last write 56m stale) | weirdlab gpu-l40 · g3097 |
| eggplant_sub100 | eggplant | 86k | ▶ R · training 86k→100k | robotics a100 · g3083 |
| eggplant_sub50 | eggplant | 88k | ▶ R · training 88k→100k | robotics a100 · g3085 |
| eggplant_sub25 | eggplant | 87k | ▶ R · training 87k→100k | robotics a40 · g3046 |

### B. BC-policy trainings (n_obs=1, H8) — the search-matched baseline
| Run | Task | Status (detail) | Best n=1 ckpt |
|---|---|---|---|
| eggplant_bc_unet_h8_50k_filtered | eggplant | ✅ done @50k | 50k = 0.88 |
| bc_unet_h8_50k_stack_sub100 | stack_cube | ✅ done @50k | 40k = 0.54 |
| bc_unet_h8_50k_stack_sub50 | stack_cube | ✅ done @50k | 30k = 0.40 |
| bc_unet_h8_50k_stack_sub25 | stack_cube | ✅ done @50k | 40k = 0.36 |
| bc_unet_h8_50k_spoon_sub100 | spoon_on_towel | ✅ done @50k | 20k = 0.64 |
| bc_unet_h8_50k_spoon_sub50 | spoon_on_towel | ✅ done @50k | 10k = 0.56 |
| bc_unet_h8_50k_spoon_sub25 | spoon_on_towel | ✅ done @50k | 10k = 0.48 |

### C. SEARCH-policy BoN evals  (N=1..64 per policy×milestone; **now on free weirdlab-l40 + cse-l40s** after a40/a100 filled up)
Done milestones (in `search_bon_results.md`): all runs 5k/10k/20k; eggplant sub100/50 to 40k too.
| Policy | Milestones in flight | Cell status | done → |
|---|---|---|---|
| stack_sub100 | — (done) | ✅ →90k complete (50k=**56%** peak; 90k=52) | → 90k |
| stack_sub50 | — (done) | ✅ →90k complete (plateau ~30–40) | → 90k |
| stack_sub25 | — (done) | ✅ →90k complete | → 90k |
| spoon_sub100 | — (idle) | ✅ →90k complete | → 90k |
| spoon_sub50 | — (idle) | ✅ →90k complete | → 90k |
| spoon_sub25 | — (idle) | ✅ →80k complete | → 80k |
| eggplant sub100/50/25 | — (idle) | ✅ →80k complete | → 80k |

(Note: stack trainings done — the **100k** milestone eval needs `latest.ckpt` (no `step_0100k` file), so search evals top out at **90k** for stack unless pointed at latest.ckpt.)

### D. BC-policy BoN evals  (`bon_*`, DQC chunk on best n=1 ckpt; correct per-task env)
| Run | Task · ckpt | Cell status | GPU · node |
|---|---|---|---|
| bon_st100 | stack sub100 · 40k | ✅ done | weirdlab gpu-l40 |
| bon_st50 | stack sub50 · 30k | ✅ done | gpu-a40 |
| bon_st25 | stack sub25 · 40k | ✅ done | weirdlab gpu-l40s · g3111 |
| bon_sp100 | spoon sub100 · 20k | ✅ done | weirdlab gpu-l40 |
| bon_sp50 | spoon sub50 · 10k | ✅ done | weirdlab gpu-l40s |
| bon_sp25 | spoon sub25 · 10k | ✅ done | gpu-a40 |

### E. Other (LLaVA-verifier line — separate experiment)
| Run | Status (detail) | GPU · node |
|---|---|---|
| sweep_bc_llava | ▶ R (16 d) · BC+LLaVA BoN sweep | weirdlab gpu-a40 · g3040 |
| tr_diff_llava_na1 | ▶ R (12 d) · diffusion-search training (LLaVA verifier) | weirdlab gpu-l40s · g3104 |

**Legend:** ▶ R = running · ⏳ PD = queued (cap wait) · ✅ done · 🟢 mostly running · 🟡 mostly queued · ⚪ idle/not-yet.
Walls: weirdlab/robotics = 7-day; **cse l40s = 1-day** (spoon chunks daily, resume-safe).
`(AssocGrpGRES / AssocGrpCpuLimit)` PD = per-account GPU/CPU cap, not an error.

---

## 1. Training runs (detail)

### Search policies — DQC chunk verifier, BC-only (`pg_lambda=0`), → 100k steps
`resume:True`, checkpoints every 1k, 1 GPU / 6 CPU / ≤96G. Launcher
`scriptsv2/train_search/launch_multitask_search_trainings.sh` (stack/spoon; eggplant resumed by
targeted sbatch). Output `.../train/{rmse_multitask,rmse_filtered}/<run>/`.

| Run | Task | Data | Job | Status | GPU · node | Resume-from |
|---|---|---|---|---|---|---|
| stack_sub100 | stack_cube | 100 ep | 36884000 | R | weirdlab l40s · g3102 | 49k → 100k |
| stack_sub50 | stack_cube | 50 ep | 36884001 | R | weirdlab l40s · g3112 | 49k → 100k |
| stack_sub25 | stack_cube | 25 ep | 36884002 | R | weirdlab l40s · g3112 | 45k → 100k |
| spoon_sub100 | spoon_on_towel | 100 ep | 36884211 | R | cse l40s · g3103 | 30k → 100k |
| spoon_sub50 | spoon_on_towel | 50 ep | 36884212 | R | cse l40s · g3112 | 41k → 100k |
| spoon_sub25 | spoon_on_towel | 25 ep | 36884213 | R | cse l40s · g3113 | 39k → 100k |
| eggplant_sub100 | eggplant | 100 ep | 36884224 | PD | robotics a100 | 49k → 100k |
| eggplant_sub50 | eggplant | 50 ep | 36884225 | PD | robotics a100 | 49k → 100k |
| eggplant_sub25 | eggplant | 25 ep | 36884226 | PD | robotics a40 | 49k → 100k |

- stack → weirdlab `gpu-l40s` (7-day). spoon → **cse `gpu-l40s`** (1-day wall → daily resume chunks).
  eggplant → **robotics `gpu-a100`×2 / `gpu-a40`×1** (7-day). All `resume:True` from `latest.ckpt`.
- eggplant run dirs: `.../rmse_filtered/diff_search_rmse_{sub100,sub50,sub25}_diverse_data/`.
- paused: `diff_search_rmse_filtered_data` (699 ep) @28k ⏸.
- Resume confirmed (stack_sub100 log: "Checkpoint loaded… preserving existing normalizer statistics").

### BC policies (n_obs=1, H8) — FINISHED (the search-matched baseline)
Baseline BC (`DiffusionUnetImagePolicy`, predict-8 / rollout-4, **`n_obs_steps=1`** to match the search policy's obs window). Launcher `scriptsv2/train_search/train_bc_denoiser.sbatch` (`DENOISER=unet N_OBS_STEPS=1`); all 50k steps ✅ done; lean **4 CPU** on **weirdlab gpu-l40**. Output `diffusion_policy/data/train/bc_denoiser/<run>/`. Same generic 40-dim widowx-state config trains all three tasks (just `DATASET_DIR` swapped).

| Run (`…_nobs1`) | Task | Data | wandb | Best ckpt (n=1 SR) |
|---|---|---|---|---|
| eggplant_bc_unet_h8_50k_filtered | eggplant | 699 ep | BC_eggplant | 50k = 0.88 |
| bc_unet_h8_50k_stack_sub100 | stack_cube | 100 ep | BC_stack | 40k = 0.54 |
| bc_unet_h8_50k_stack_sub50 | stack_cube | 50 ep | BC_stack | 30k = 0.40 |
| bc_unet_h8_50k_stack_sub25 | stack_cube | 25 ep | BC_stack | 40k = 0.36 |
| bc_unet_h8_50k_spoon_sub100 | spoon_on_towel | 100 ep | BC_spoon | 20k = 0.64 |
| bc_unet_h8_50k_spoon_sub50 | spoon_on_towel | 50 ep | BC_spoon | 10k = 0.56 |
| bc_unet_h8_50k_spoon_sub25 | spoon_on_towel | 25 ep | BC_spoon | 10k = 0.48 |

(Earlier BC: eggplant sub100/50/25 nobs1 [stopped early] + n_obs=2 variants — see `data/eval/search_bon_results.md` §4b.)

### Other training (LLaVA-verifier line — separate experiment)
| Run | Job | Status | GPU · node |
|---|---|---|---|
| tr_diff_llava_na1 (diffusion search, LLaVA verifier, n_action=1) | 36360696 | R (12 d) | weirdlab gpu-l40s · g3104 |

---

## 2. Eval runs (Best-of-N sweeps)

Each "run" = one policy × checkpoint-milestone BoN sweep over N=1,2,4,8,16,32,64 (7 cells),
2 CPU / 16G per cell, softmax, 50 ep, seeds 1000–1049. Cells are transient (complete + refire).

### (a) SEARCH-policy BoN — DQC chunk  →  `weirdlab gpu-l40` + `cse gpu-l40s` (rerouted 2026-07-05)
The main deliverable. Launcher `scriptsv2/eval_search/launch_multitask_ckpt_evals.sh`
(sbatch `eval_search_dqc_chunk.sbatch`, 12h wall). **ROUTES repointed to the FREE partitions**
(`weirdlab|gpu-l40`, `cse|gpu-l40s`) after a40/a100 filled to 0-free — policy: when eval cells
stall PD, re-check `hyakalloc` and repoint ROUTES at the free rows.
Cell names `<task>_<sub>_<milestone>_n<N>`. Output `data/eval/{stack_cube_search,spoon_towel_search}/`.
Milestones: **10k 20k 30k … 100k** per run.

| Policy | Milestones done (in doc) | Milestones live now | GPU |
|---|---|---|---|
| stack_sub100 | 5k/10k/20k | 30k, 40k | a40 + a100 |
| stack_sub50 | 5k/10k/20k | 30k, 40k | a40 + a100 |
| stack_sub25 | 5k/10k/20k | 30k, 40k | a40 + a100 |
| spoon_sub100 | 5k/10k/20k | 30k | a40 + a100 |
| spoon_sub50 | 5k/10k/20k | 30k, 40k | a40 + a100 |
| spoon_sub25 | 5k/10k/20k | 30k | a40 + a100 |

50k–100k fire automatically (via the login-node crontab) as the resumed training reaches them.

### (b) BC-policy BoN — DQC chunk, on the BEST n=1 checkpoint  →  spread over free GPUs
Separate family: DQC-chunk BoN (ddpm, softmax, N=1–64) on each BC policy's **best n=1 checkpoint**,
via `eval_bc_dqc_chunk.sbatch` (launcher `launch_bc_dqc_bon.sh`). **Correct per-task env** (verified):
stack → `TASK=widowx_stack_cube` + `DQC_TEXT_EMB=task_emb_stack_use3.npy`; spoon →
`widowx_spoon_on_towel` + `task_emb_spoon_use3.npy` (NOT the eggplant default). Cells
`bon_{st,sp}{100,50,25}_n<N>`, output `data/eval/bc_dqc_chunk/`. 6 hourly ticks `tick_bc_dqc_bon.sh`
(crontab `:02`–`:12`) log each SR-vs-N curve to wandb (`BC_stack`/`BC_spoon`) + self-remove.

| Run | Task · best ckpt | env / task_emb | GPU · node (current) |
|---|---|---|---|
| bon_st100 | stack sub100 · 40k | widowx_stack_cube · stack | weirdlab gpu-l40 · g3115/g3116/g3097/g3098 |
| bon_st50 | stack sub50 · 30k | widowx_stack_cube · stack | gpu-a40 · g3043/g3051/g3062 |
| bon_st25 | stack sub25 · 40k | widowx_stack_cube · stack | weirdlab gpu-l40s · g3111 |
| bon_sp100 | spoon sub100 · 20k | widowx_spoon_on_towel · spoon | weirdlab gpu-l40 (PD on cap) |
| bon_sp50 | spoon sub50 · 10k | widowx_spoon_on_towel · spoon | weirdlab gpu-l40s · g3101/g3103/g3104 |
| bon_sp25 | spoon sub25 · 10k | widowx_spoon_on_towel · spoon | gpu-a40 (PD on cap) |

### (c) BC LLaVA sweep (LLaVA-verifier line — separate experiment)
| Run | Job | Status | GPU · node |
|---|---|---|---|
| sweep_bc_llava | 36180188 | R (16 d) | weirdlab gpu-a40 · g3040 |

---

## 3. GPU-partition summary

| Partition | Wall | Used by |
|---|---|---|
| **weirdlab gpu-l40s** | 7-day | stack search trainings; some BC-BoN cells; tr_diff_llava_na1 |
| **cse gpu-l40s** | **1-day** | spoon search trainings (daily resume chunks); **search-policy BoN eval cells (rerouted)** |
| **robotics gpu-a100** | 7-day | eggplant search trainings (sub100 / sub50) |
| **robotics gpu-a40** | 7-day | eggplant search training (sub25) |
| **weirdlab gpu-l40** | 7-day | **search-policy BoN eval cells (rerouted)**; BC-policy BoN cells; BC n_obs=1 trainings |
| **gpu-a40** (robotics + weirdlab) | 7-day | some BC-BoN cells; sweep_bc_llava (a40/a100 now FULL — evals moved off) |
| **gpu-a100** (cse + robotics) | 7-day (cse 1-day) | (was search-BoN overflow; now full — evals rerouted to l40/l40s) |

**Eval routing policy:** when eval cells stall PD, re-check `hyakalloc` and repoint the launcher
ROUTES (`launch_multitask_ckpt_evals.sh` / `launch_rmse_ckpt_evals.sh`) at whatever partitions are FREE.
`(AssocGrpGRES / AssocGrpCpuLimit)` in a PD reason = waiting on the per-account GPU/CPU cap, not an error.

### Naming cheat-sheet
- `diff_search_rmse_<task>_<sub>_diverse[_data]` → a **training** run (search policy).
- `<task>_<sub>_<milestone>_n<N>` (e.g. `stack_sub50_40k_n16`) → a **search-policy BoN eval** cell (a40/a100).
- `bon_<st|sp><size>_n<N>` (e.g. `bon_sp50_n16`) → a **BC-policy BoN eval** cell (l40/l40s/a40).
