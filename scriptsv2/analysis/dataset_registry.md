# SIMPLER dataset registry

Datasets under `/gscratch/robotics/harine/data`. Hand-maintained (the generator was retired
2026-10-01); update it when adding, filtering or archiving a dataset.

## Env seeds

Collection used `env.reset(seed = 100000 + global_idx)`; nothing stores the seed, but every
filter/subsample stage keeps `src_shard`/`src_episode` in `filter_manifest.json`, so
`seed = 100000 + start_index[src_shard] + src_episode`. `start_index` per task:

- eggplant (raw `eggplant_in_basket`): {'state0.zarr': 0, 'state1.zarr': 1472} (the rule holds for state0 too; ~13% of eggplant episodes fail a 1 mm start-state gate because the eggplant's settle depends on sim history — `demoseeds/eggplant_strict_seed_gate.json`. The `_eggplant_shard_map.json` greedy layout match is NOT valid.)
- spoon (raw `spoon_on_towel`): {'state0.zarr': 0, 'state1.zarr': 1000}
- stack_cube (raw `stack_cube`): {'state0.zarr': 0, 'state1.zarr': 1000}

Manifests: `scriptsv2/eval_search/demoseeds/<task>_<split>.json` (from
`scriptsv2/analysis/reconstruct_demo_seeds.py`, gated by `validate_demo_seeds.py`).
Eval (`eval_search.py --split`) rolls out exactly these seeds; a checkpoint carries its
training dir's chain + train seeds under `cfg.dataset_provenance`.

## Archived 2026-09-30

Only the strict datasets (`strict_distance_success`, `stack_cube_strict`, `spoon_on_towel_strict`)
are used from now on. The raw collections they were filtered from and the diverse-start
sub-samples / complement test sets were moved to `/gscratch/robotics/harine/data/_archive/`
(`manifest.jsonl` there; `scriptsv2/data/archive_datasets.sh`). Rows below marked (archived)
now live under `_archive/`. Not touched: `carrot_on_plate`, `*_lerobot`, `eggplant_in_basket_val*`,
`eggplant_reset_core_*`.

## Split-only datasets (2026-09-30, used for training and eval)

| dir | episodes | from |
|---|---|---|
| `strict_distance_success_split` | 536 | the in-split episodes of `strict_distance_success` (699) |
| `stack_cube_strict_split` | 272 | the in-split episodes of `stack_cube_strict` (280) |

Built by `scriptsv2/data/make_split_dataset.py`; `filter_manifest.json` maps each episode to its
strict index, shard coordinates and seed. All episodes are recreatable from their seed (end
effector exact; object settle offsets up to ~8 mm are environment stochasticity). The strict
datasets, and with them the 163 + 8 excluded episodes, move to `_archive/` once the four BC
runs still reading them finish (job `archive_strict`).

## Split manifests

`diffusion_policy/diffusion_policy/config/splits/simpler/<task>/<rule>_train<N>.json`, from
`scriptsv2/splits/make_simpler_splits.py` then remapped onto the split-only datasets (test 50 / val 20, train sizes nested, per-episode seeds;
test pools exclude seed-gate failures — `scriptsv2/veritas/seed_gate.py`):

- eggplant (699 eps, 28 start cells): random 400/100/50; quadrant (x<-0.1165, y<0.2125) 100/50; inner (r<=6 mm) 100/50
- stack (280 eps, 6 start cells): random 200/100/50; quadrant (green-cube start y>0) 50

Eval: `eval_search.py --split-file <manifest> --split-name test`. Training: `task.dataset.split_file`.

## Datasets

| dir | zarr shards | episodes | derived from | subsample seed | manifest md5 | demo seeds (split: n, range, manifest md5) |
|---|---|---|---|---|---|---|
| `carrot_on_plate` | state0.zarr, state1.zarr, state2.zarr, state3.zarr | 4000 | - | - | - | - |
| `eggplant_in_basket` (archived) | state0.zarr, state1.zarr, state2.zarr | 2094 | - | - | - | - |
| `eggplant_in_basket_complete` (archived) | state0.zarr, state1.zarr | 2094 | - | - | - | - |
| `eggplant_in_basket_val` | val0.zarr, val1.zarr, val2.zarr, val3.zarr | 1000 | - | - | - | - |
| `eggplant_in_basket_val_success` | val_success.zarr | 343 | - | - | 0ef5c3bd | - |
| `eggplant_reset_core_x-0.22_-0.06_y0.02_0.26` | long0.zarr | 8377 | - | - | - | - |
| `spoon_on_towel` (archived) | state0.zarr, state1.zarr | 2000 | - | - | - | - |
| `spoon_on_towel_strict` | spoon_strict.zarr | 510 | - | - | 1d840227 | - |
| `spoon_on_towel_strict_sub100_diverse_start` (archived) | spoon_on_towel_strict_sub100_diverse_start.zarr | 100 | spoon_on_towel_strict | 0 | f00adbbf | - |
| `spoon_on_towel_strict_sub25_diverse_start` (archived) | spoon_on_towel_strict_sub25_diverse_start.zarr | 25 | spoon_on_towel_strict | 0 | 41fe0433 | - |
| `spoon_on_towel_strict_sub50_diverse_start` (archived) | spoon_on_towel_strict_sub50_diverse_start.zarr | 50 | spoon_on_towel_strict | 0 | 988e6e70 | train: 50 [100054..101987] 8e8346e7 |
| `spoon_on_towel_strict_testset_not_in_sub100` (archived) | spoon_on_towel_strict_testset_not_in_sub100.zarr | 410 | spoon_on_towel_strict minus spoon_on_towel_strict_sub100_diverse_start | - | 9d710443 | test: 50 [100012..101997] bbdbde13 |
| `stack_cube` (archived) | state0.zarr, state1.zarr | 2000 | - | - | - | - |
| `stack_cube_strict` | stack_strict.zarr | 280 | - | - | 9db82c42 | - |
| `stack_cube_strict_sub100_diverse_start` (archived) | stack_cube_strict_sub100_diverse_start.zarr | 100 | stack_cube_strict | 0 | fa0094ae | - |
| `stack_cube_strict_sub25_diverse_start` (archived) | stack_cube_strict_sub25_diverse_start.zarr | 25 | stack_cube_strict | 0 | 37bd6948 | - |
| `stack_cube_strict_sub50_diverse_start` (archived) | stack_cube_strict_sub50_diverse_start.zarr | 50 | stack_cube_strict | 0 | 57dec1e9 | train: 50 [100069..101973] a739a976 |
| `stack_cube_strict_testset_not_in_sub100` (archived) | stack_cube_strict_testset_not_in_sub100.zarr | 180 | stack_cube_strict minus stack_cube_strict_sub100_diverse_start | - | 036c3255 | test: 50 [100000..101997] ff3be828 |
| `strict_distance_success` | strict_distance_success.zarr | 699 | - | - | a74f2b17 | - |
| `strict_distance_success_sub100_diverse_start` (archived) | strict_distance_success_sub100_diverse_start.zarr | 100 | strict_distance_success | 0 | 152129bc | - |
| `strict_distance_success_sub25_diverse_start` (archived) | strict_distance_success_sub25_diverse_start.zarr | 25 | strict_distance_success | 0 | df2cd994 | - |
| `strict_distance_success_sub50_diverse_start` (archived) | strict_distance_success_sub50_diverse_start.zarr | 50 | strict_distance_success | 0 | b68031d1 | train: 50 [100006..102022] 5469462d |
| `strict_distance_success_testset_not_in_sub100` (archived) | strict_distance_success_testset_not_in_sub100.zarr | 599 | strict_distance_success minus strict_distance_success_sub100_diverse_start | - | fd16e9e3 | test: 50 [100004..102093] 4eddfa02 |
