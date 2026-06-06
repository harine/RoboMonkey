# Results notes — deterministic-sampling BoN eval re-runs

Started 2026-06-05. Diffusion candidate sampling is now seeded per-replan
(`f(episode_seed, t)`) so the N candidate sets are reproducible run-to-run;
softmax selection stays stochastic. Sweep N = 2^0..2^10 (1,2,4,...,1024).

Provenance: each cell's `eval_log.json` records `deterministic_sampling: true`
and `sample_seed_base`. Set `STOCHASTIC_SAMPLING=1` to reproduce the old
un-seeded behavior.

## Noise / tmrl sweep (noise0.4, noise0.8, tmrl)
Root: `data/eval/search_noise_sweep/`

Per-cell output dirs (SOFTMAX_TEMP=1.0, START_SEED=1000, NUM_EPISODES=50):
- noise0.4 : `noise0.4/softmaxT1.0_n<N>_seed1000_ep50/`
- noise0.8 : `noise0.8/softmaxT1.0_n<N>_seed1000_ep50/`
- tmrl     : `tmrl/softmaxT1.0_tcont<TC>_n<N>_seed1000_ep50/`   (TC in 0.4, 0.8, 0.0)

Each cell dir contains:
- `eval_log.json`   — summary (success_rate, num_successes, selected-index histogram, flags)
- `episodes.jsonl`  — per-episode rows (success, q_values per replan, selected indices)

Sweep-level:
- summary table + run log : `data/eval/search_noise_sweep/_summaries/softmax_n_sweep_<TS>.txt`
- per-cell stdout/stderr  : `data/eval/search_noise_sweep/_summaries/.sweep_<TS>/`
- SLURM job logs          : `data/eval/search_noise_sweep/_summaries/slurm-<jobid>.out`

Split across two SLURM jobs (run on currently-available GPUs), launched
2026-06-05, FORCE=0 (resumable), NUM_WORKERS=4 each:
- Job A = 35916347 : noise0.4 + noise0.8 on 4x l40s (socialrl), node g3108
- Job B = 35916348 : tmrl (tcont 0.4/0.8/0.0) on 4x l40 (socialrl), node g3098
Verified at launch: N=1..1024, MODE=softmax, "sampling: deterministic
candidates (seed base=0)". Smoke gate (job 35916133) = PASS (two argmax runs
bit-identical).

## BC sweep (deferred until BC training finishes)
Root: `data/eval/search_bc_sweep/`
- Per-cell: `bc/softmaxT1.0_n<N>_seed1000_ep<E>/` (10 fixed seeds shared across N)
- Summaries: `data/eval/search_bc_sweep/_summaries/`

## Archived prior (stochastic) results
`data/eval/search_noise_sweep/_stochastic_bak_20260605/{noise0.4,noise0.8,tmrl}/`
— the pre-change un-seeded runs, moved aside so deterministic runs write fresh.

## Smoke test (seeding sanity check)
`data/eval/_smoke/` — two MODE=argmax runs that must be bit-identical.
Log: `data/eval/_smoke/slurm-<jobid>.out` (look for `SMOKE RESULT: PASS`).
