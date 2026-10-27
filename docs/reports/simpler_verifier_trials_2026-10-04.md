# Simpler: BC, ST, and verifier trials
**Stack:** BC20k leads ST. **Eggplant:** uniform ST30k reaches 68% at N16; scaling is nonmonotonic.

## Candidate scaling
50 episodes/cell. Predict16 / score8 / execute8. BC fixed20k; ST checkpoints labeled. Gaps = pending. BC argmax is repeated beside ST final.

Eggplant: 100 demos, cached plans. Stack: random50, fresh plans after reset. ST uniform shown at30k (egg) /20k (stack); noise models at10k. These are checkpoint-specific results.


## Train/test positions
Actual manifest positions; overlap is unjittered. Stack quadrant holds out source y>0. Quadrant BC20k is 0%; ST is 0–2% in completed cells.


## Verifier trials
BC100: eggplant **10k**, stack **40k**. 30 paired episodes. Benchmark success.

| Verifier | Egg N4 | Egg N16 | Stack N4 | Stack N16 |
|---|---:|---:|---:|---:|
| Metric v1 | 40% | 40% | 53% | 57% |
| Metric v2 | 37% | 40% | 47% | 67% |
| Pixel Veritas | 63% | 37% | 50% | 63% |
| Pixel + gripper 0.1 | 63% | 60% | 50% | 60% |
| Two subtasks + sim full filter* | — | 57% | — | — |
| Sim Veritas + distance* | — | 53% | — | — |
| Sim Veritas + full filter* | — | 40% | — | — |

N1: egg **53%**, stack **60%**. *16 samples → 48 gripper choices. Full-filter pass is diagnostic.


## Earlier BC checkpoints
50 demos; 30 paired episodes. Eggplant **20k**, stack **30k**.

| Verifier | Egg N4 / N16 / N64 | Stack N4 / N16 / N64 |
|---|---|---|
| Pixel Veritas | 57 / 60 / 50% | 60 / 67 / 60% |
| + distance | 40 / 63 / 57% | 53 / 63 / 53% |
| Sim Veritas | 50 / 47 / 53% | 53 / 63 / 63% |
| Sim + distance | 40 / 60 / 57% | 60 / 53 / 63% |

N1: egg **53%**, stack **37%**. Stack scales here; the stronger 40k pilot had smaller gains.


## Verifier trajectories
Returned image-space waypoints. Fresh plan after reset; candidate feedback is computed locally.


## Feedback examples
Higher score wins; scales differ. BC checkpoints: egg10k / stack40k.


**Eggplant**

| Verifier | Seed / step | Choice | Score |
|---|---|---:|---:|
| Metric v1 | 100028 / 16 | 12 | -0.486 |
| Metric v2 | 100028 / 16 | 1 | 0.265 |
| Pixel Veritas | 100028 / 8 | 8 | -1.983 |
| Pixel + gripper | 100028 / 8 | 8 | -2.083 |
| Two subtasks + gripper | 100028 / 16 | 38 | -3.510 |
| Carry-only | 100028 / 16 | 38 | -4.268 |
| Two subtasks + full filter | 100028 / 16 | 38 | -3.303 |
| Simulated Veritas + distance | 100028 / 16 | 47 | -1.192 |
| Simulated Veritas + full filter | 100080 / 8 | 41 | -2.257 |

**Stack**

| Verifier | Seed / step | Choice | Score |
|---|---|---:|---:|
| Metric v1 | 100069 / 16 | 15 | -0.652 |
| Metric v2 | 100069 / 16 | 15 | 0.380 |
| Pixel Veritas | 100067 / 8 | 14 | -4.089 |
| Pixel + gripper | 100067 / 8 | 14 | -4.179 |
| Two subtasks + gripper | 100067 / 16 | 28 | -3.224 |
| Four subtasks | 100067 / 16 | 14 | -1.980 |
| Two subtasks | 100067 / 16 | 12 | -3.910 |

## Gripper and drift
| Issue | Tried | Remaining |
|---|---|---|
| Pixel score ignores grasp | Scaled gripper penalty | Wrong open/close choices remain |
| Kinematic drift | Calibrated motion; physics rollout | Simulation alone did not improve scaling |
| Stage switching | Continuous carry; position gates | Stalls / phase errors remain |
| Carry / placement | Full-filter metrics | Wrong trajectory can outweigh task progress |
| Transient success | Placement / release checks | Object can be carried away afterward |

## Rollout examples
Cropped videos; selected candidate and verifier score overlays.


## Eggplant uniform: all checkpoints
Clean observations; 50 episodes/cell.

[Download report data](https://wandb.ai/l2sml/simpler_veritas_h16/artifacts/report-assets/simpler-concise-report-20261008/latest).
