# Veritas pixel-waypoint verifier for SimplerEnv

Port of the Veritas verifier (github.com/harine/veritas; PushT port on `diffusion_policy`
branch `harine/pushT`) to the WidowX SimplerEnv tasks. Gemini emits one pixel-waypoint plan
per episode from the 640x480 first frame; a candidate action chunk is scored by upstream's
lookahead endpoint cost (integrate the chunk's xyz deltas from the TCP, project, decayed
distance to the next K=3 waypoints), negated so higher is better. The tracker (active
waypoint, hold/skip state) advances on EXECUTED env steps only.

Code: `diffusion_policy/diffusion_policy/veritas/` (shared: Gemini client, schemas, tracker,
plan store), `diffusion_policy/diffusion_policy/env/simpler/veritas/` (camera model, object
hints, Simpler Gemini client, episode tracker / env wrapper / snapshot store),
`diffusion_policy/diffusion_policy/policy/veritas_verifier.py` (`VeritasPixelVerifier`).
Unit tests: `diffusion_policy/unit_tests/test_veritas_simpler.py`.

## Pipeline (eggplant first, then stack; carrot on hold)

All simulator steps need a GPU node (SAPIEN/Vulkan): `sbatch --export=ALL,CMD="..." scriptsv2/veritas/prep.sbatch`.

1. **Camera model** (once per task): `capture_simpler_camera.py --task <env_task>` →
   `diffusion_policy/config/veritas/camera_<task>.json`.
2. **Seed gate**: `seed_gate.py --task eggplant|stack` → `scriptsv2/eval_search/demoseeds/<task>_strict_seed_gate.json`.
   Seeds follow `100000 + START_INDEX[shard] + src_episode`; episodes whose reset does not
   reproduce the recorded start state (tol 1 mm) are kept out of TEST pools.
3. **Split manifests**: `scriptsv2/splits/make_simpler_splits.py --task eggplant|stack` →
   `diffusion_policy/config/splits/simpler/<task>/<rule>_train<N>.json` (PushT manifest format
   + per-episode seeds). Eggplant: random 400/100/50, quadrant 100/50, inner 100/50; stack:
   random 200/100/50, quadrant 50. Test 50 / val 20, train sizes nested.
4. **Plans** (needs `GEMINI_API_KEY`; model `GEMINI_MODEL`, default gemini-3.8-flash):
   `plan_simpler_episodes.py --task <env_task> --split-file <manifest> --splits train val test --outdir <plan_dir> --skip-planned`.
   Run once over the union of a task's manifests (the random_train400 manifest covers every
   eggplant episode). Records: `<plan_dir>/ep{idx}.json` + `seed{seed}.json` symlink + overlays + cost summary.
5. **Targets for training**: `build_veritas_targets.py --task <env_task> --split-file <manifest> --plan-dir <plan_dir> --horizon 8`
   → `<plan_dir>/veritas_targets.npy/.json` (per-frame tracker state replayed along each demo;
   expert-chunk value mean/std for ST conditioning).
6. **BC per split**: `SUBMIT=1 bash scriptsv2/train_search/submit_bc_splits.sh eggplant`.
7. **BoN sweep**: `CKPT=... SPLIT_FILE=... PLAN_DIR=... SCALE_FIT="[0.72,0.73,0.87]" SUBMIT=1 bash scriptsv2/eval_search/submit_veritas_bon.sh eggplant`
   (arms `scale_null` = upstream 1:1, `scale_fit` = per-axis H=8 base-frame fit from
   `scriptsv2/train_search/calibrate_distance_verifier.py --fit horizon_cumulative --frame base --camera-file ...`).
8. **ST training**: `sbatch --export=ALL,SPLIT_FILE=...,PLAN_DIR=... scriptsv2/train_search/train_search_veritas.sbatch`
   (config `robomonkey_eggplant_search_veritas_outer_inner.yaml`).

Smoke without Gemini: `smoke_synthetic_plans.py` writes a tiny manifest + straight-line plans
under `data/veritas/_smoke/`; steps 5-8 then run on them.

Archive of the non-strict datasets: `DO_IT=1 bash scriptsv2/data/archive_datasets.sh`.
