# Eval seeding and provenance — what was tried, what is live

## Live (2026-10-01)
One seeding path in `scriptsv2/eval_search/eval_search.py`:
- Episode seeds come from a split manifest
  (`diffusion_policy/diffusion_policy/config/splits/simpler/<task>/<rule>_train<N>.json`):
  `--split test|val|train` (`SPLIT=`; `--split-name`/`SPLIT_NAME` are aliases) picks the list,
  each episode's `seed` is rolled out. `--split-file` (`SPLIT_FILE=`) defaults to the
  checkpoint's own `task.dataset.split_file`, so `test` is exactly its held-out set.
  `--seeds a,b,c` gives an explicit list (the only way to reach the archived sub50 sets in
  `demoseeds/`); `--num-episodes K` caps at the first K seeds (default all). A test set is
  held out only w.r.t. its own manifest: e.g. inner_train100's test shares 2 seeds with
  random_train50's train, so that cross-eval is refused.
- Per-replan seed = `SeedSequence([episode_seed, t])`. The policy draws obs-corruption noise
  and every candidate's noise from its per-row stream (`SearchProcedureMixin.set_sample_seeds`,
  shared with maze/PushT); the softmax pick has its own stream keyed on the same seed.
  Identical in the single-env and `TILED_ENVS>1` paths (up to batch-shape float drift).
- Resume reuses an episode record only if `tiled_envs` and `obs_corruption` match.
- Provenance: `run_config.json` records the split manifest (path, md5, dataset dir, seeds)
  and the checkpoint's `cfg.dataset_provenance` (training dir manifest chain, and
  `split_manifest` = path/md5/per-split seeds, stamped by the training workspaces).
  `--split test` refuses seeds that are in the checkpoint's training demos (stamped seeds,
  else its split_file read at eval time, else the demoseeds train set for pre-manifest
  checkpoints). Overlap is by seed, not index: the manifests were remapped onto the
  split-only datasets, which renumbered episodes but kept every split's seeds.
  Every episode record carries `src_shard`/`src_episode`.
- Dataset ↔ seed table: `scriptsv2/analysis/dataset_registry.md` (hand-maintained).
  Seed reconstruction: `reconstruct_demo_seeds.py`, gate: `validate_demo_seeds.py`.
- Smoke: `scriptsv2/eval_search/_smoke_determinism.sbatch` (A1==A2, D1==D2, C differs).

## History (removed)
| when | what | why removed |
|---|---|---|
| ≤2026-06 | un-seeded candidates (`STOCHASTIC_SAMPLING=1`) | not reproducible run-to-run |
| 2026-06-05 | per-replan global `torch.manual_seed(f(ep_seed, t))` | candidates depended on batch position / TILED_ENVS |
| 2026-06 | `START_SEED=1000 + i` episode seeds, `REPEAT_SEED` | unrelated to any dataset; test-vs-train disjointness only by accident |
| 2026-07 | `TILED_SEED_PER_ENV=0` shared per-step seed | correlated envs, shifted with done-timing |
| 2026-09-29 | scheme `v2`: `SeedSequence([ep, t, SAMPLE_SEED_BASE])`, per-row `[seed,0,j]` / `[seed,1]` keys, `SEED_SCHEME` resume tag | SIMPLER-only; replaced by maze's stream |
| 2026-09-30 | `--tcont` / noise-conditioned policy | class dropped (see git history) |

Old result cells named `*_seed1000_ep50` under `data/eval/` were produced with the
`START_SEED` scheme and are not comparable to split-based cells (new cells are named
`*_<split>_ep<N>`). `_viz*` video dumps and the May-2026 `robomonkey_outputs`/`_sweeps`
were deleted on 2026-09-30.


## Simpler verifier experiments — 2026-10-04; return to simple ST + Veritas

Decision: return to the corrected original VeritasPixelVerifier and standard
conditional-diffusion ST, k=16, uniform slot loss, clean observations. Retain
predict16 / score8 / execute8 and DDIM8. Preserve correctness fixes and existing
checkpoints; do not roll back the repository or delete pilot artifacts. This
switch does not submit or resume training. Evaluate argmax and final candidate
separately. Use immutable checkpoint paths, with BC selected by N=1 performance.

Training configuration: diffusion_policy/diffusion_policy/config/robomonkey_veritas_h16.yaml
(the campaign's `kind=uniform`; task-specific camera, calibration and targets must
still be supplied). Evaluation preset:
scriptsv2/eval_search/simple_st_veritas.env.sh. Source it before the normal eval
launcher; it disables pilot scoring and augmentation, but does not choose a
checkpoint, task, split, camera or plan directory. Fresh plans after actual reset
remain the intended protocol; the original campaign's cached-plan evaluation is
a distinct protocol and must be labelled as such, not silently mixed with the
fresh-plan pilots. The preset deliberately does not change plan provisioning.

All counts below are environment benchmark successes over 30 episodes, not
strict/full-filter pass rates. Eggplant BC N=1: 16/30; stack BC N=1: 18/30.

| Variant | Eggplant N=4 | Eggplant N=16 | Stack N=16 |
|---|---:|---:|---:|
| Gemini metric plan v1, kinematic scoring | 12/30 | 12/30 | 17/30 |
| Calibrated metric plan v2 | 11/30 | 12/30 | 20/30 |
| Corrected original pixel trajectory, gripper weight 0 | 19/30 | 11/30 | 19/30 |
| Pixel trajectory + bounded gripper penalty 0.1 | 19/30 | 18/30 | 18/30 |
| Two-subtask analytical trajectory + simulated full filter | not run | 17/30 | not run |
| Simulated pixel trajectory + basket distance | cancelled | 16/30 | not run |
| Simulated pixel trajectory + full filter | not run | 12/30 | not run |

The final three N=16 rows use 16 BC draws expanded to 48 choices with original,
forced-open and forced-closed gripper variants. Earlier rows use 16 unmodified
candidates. These are different candidate-generation protocols. Small differences
on 30 episodes do not establish scaling. No new-verifier ST training was launched.

Single-episode diagnostics also tried four subtasks (approach/pick/transport/stack),
two subtasks (grasp+lift/place+release), explicit gripper variants, and a carry-only
metric. They exposed blocked/stale subtask progress, excessive open-gripper
selection, and benchmark success while an object remained held and was later
carried away. Carry-only scoring selected the same actions in its eggplant trial.
These single episodes are diagnostic evidence, not success-rate estimates.

Full-filter scoring combines progress with the strict-demo requirements: benchmark
success, terminal distance reduction, bounded backtracking, bounded gripper-object
separation, first-try grasp, no sustained reopening/contact loss, and bounded
separation increase during carry. It is a soft candidate score. Historical and
retrospective requirements need not be useful next-action values. Full-filter pass
is not reported as a task success measure.

Latest simulated pixel/full-filter audit: 420 selected eight-action chunks with a
following observed state. Analytical TCP prediction median/p95 error = 1.32/11.12
cm; simulator median approximately zero and p95 2.28 mm. Object p95 error = 9.00 mm.
Nine TCP errors exceeded 1 cm, with a 22.14 cm maximum: residual branching/contact
mismatch remains. On moving chunks alone simulator p95 = 2.57 mm.

At 327/450 decisions all 48 task scores tied. At 79/450 the combined score selected
a strictly worse task score than another available candidate. In 17/18 failed
episodes no candidate predicted success within the eight-action horizon at any
visited decision; this does not rule out successful longer-term behavior. In the
remaining failure a predicted-success candidate was selected. Better dynamics
prediction therefore did not establish better ranking: waypoint quality, score
weighting, saturated historical checks, candidate coverage and short lookahead
remain plausible limitations. These observations do not prove a single cause.

Artifacts (preserved):
/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004/
- sim_lookahead_30ep_episodes.csv
- sim_lookahead_30ep_report.json
- sim_prediction_and_ranking_audit.json
- eval_pixel_sim_full_trial_30ep/eggplant/n16/ (traces and first-five overlays)

Possible later ablation, not launched: simulated trajectory-only vs task-only vs
combined with matched checkpoint, seeds and candidate construction.


## Full Simpler verifier report (2026-10-04)

See [eggplant and stack verifier trials report](docs/reports/simpler_verifier_trials_2026-10-04.md),
[frozen-BC verifier results CSV](docs/reports/simpler_verifier_trials_2026-10-04.csv),
and [trained-ST campaign results CSV](docs/reports/simpler_trained_st_campaign_results_2026-10-04.csv).
W&B report: https://wandb.ai/l2sml/simpler_veritas_h16/reports/Simpler-verifier-trials-eggplant-and-stack-2026-10-04--VmlldzoxODA1NTg1Mg
