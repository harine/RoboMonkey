# mse_q — calibrate the RoboMonkey verifier to the MSE-error scale

The state-based search policy is trained with **`MSEVerifier`**, whose score is
the negative MSE between a candidate action chunk and the dataset expert
action (`-mean((cand - expert)^2)`, higher = closer to expert).

At **eval time there is no ground-truth action**, so `MSEVerifier` cannot run —
only the **RoboMonkey reward model** is available, and its scores live on a
different, unknown scale.

`mse_q` learns a small MLP

```
g(robomonkey_reward)  ≈  -mse(candidate, expert)
```

so the RoboMonkey verifier can stand in for `MSEVerifier` when evaluating the
MSE-trained search policy.

## How the calibration data is built

For every action window in the offline dataset (the **same shards the search
policy was trained on**), noisy candidate actions are formed:

```
candidate = expert + σ · per_dim_action_std · N(0, 1)
```

over a grid of `σ` values (`σ = 0` keeps the expert action — the `mse = 0` /
max-reward anchor). Each candidate is scored twice:

| quantity     | source                              | scale          |
|--------------|-------------------------------------|----------------|
| `mse_target` | `-mean((cand - expert)^2)`          | MSEVerifier    |
| `reward`     | `RoboMonkeyVerifier(image, cand)`   | reward model   |

The `(reward, mse_target)` pairs train the MLP.

## Files

- [`calibrator.py`](calibrator.py) — `MLPCalibrator`, a self-contained 1-D MLP
  (`reward → -mse`) with input/output standardization stats stored as buffers.
- [`train_mse_q.py`](train_mse_q.py) — builds the calibration set and trains
  the calibrator.
- [`train_mse_q.sh`](train_mse_q.sh) — wrapper (conda env + paths).

## Run

```bash
bash scriptsv2/mse_q/train_mse_q.sh
# quick smoke run:
MAX_SAMPLES=2000 EPOCHS=50 bash scriptsv2/mse_q/train_mse_q.sh
```

**Env split** — no single conda env has both dependency sets, so the wrapper
uses HTTP mode:

| process          | conda env         | needs                                  |
|------------------|-------------------|-----------------------------------------|
| `infer_server.py`| `monkey-verifier` | reward model / `llava` stack            |
| `train_mse_q.py` | `simpler_env`     | `diffusion_policy` / `zarr` / `hydra`   |

`train_mse_q.sh` auto-starts `infer_server.py` in the `monkey-verifier` env if
it is not already running (log: `OUTPUT_DIR/infer_server.log`) and stops it on
exit. If you already have the server up, it is reused. Override envs with
`TRAIN_ENV` / `VERIFIER_ENV`.

## Outputs (`OUTPUT_DIR`, default `data/mse_q/eggplant`)

| file                     | contents                                          |
|--------------------------|---------------------------------------------------|
| `mse_q_calibrator.pt`    | trained `MLPCalibrator` (weights + arch + stats)  |
| `calibration.png`        | reward vs −mse scatter (colored by σ) + fitted g  |
| `calibration_pairs.npz`  | raw `(reward, mse_target, sigma)` arrays          |
| `summary.json`           | fit metrics, per-σ table, run args                |

## Using the calibrator to evaluate the search policy

```python
from scriptsv2.mse_q.calibrator import MLPCalibrator

calibrator, meta = MLPCalibrator.load("data/mse_q/eggplant/mse_q_calibrator.pt")
mse_scale_score = calibrator(robomonkey_reward)   # tensor in, tensor out
```

Wrap a `RoboMonkeyVerifier` so its `get_value` returns
`calibrator(robomonkey_reward)` — the result is on the `MSEVerifier` scale the
search policy was trained against.
