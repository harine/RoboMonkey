#!/usr/bin/env python
"""Log a DQC Best-of-N success-rate curve (SR vs N) into a wandb run.

Reads the per-N eval_log.json files from a DQC BoN sweep
(data/eval/bc_dqc_chunk/<label>_n<N>_seed<seed>_ep<ep>/) and logs SR-vs-N into
the training wandb run (resume by id) on a custom ``eval/<prefix>_N`` x-axis, so
it coexists with the n=1-vs-step curve already on the run.

Idempotent: writes a marker once logged; --require-all (default) waits until all
N are complete. Run in an env with wandb (e.g. monkey-verifier).

Example:
  python scriptsv2/eval_search/log_dqc_bon_to_wandb.py \
      --run-id unet_denoiser_pred8_roll4_50k_filtered \
      --label bcH8filt50k_chunk8_predh8_roll4_ddpm --n-list "1 2 4 8 16 32 64"
"""
import argparse
import json
import os
import sys

BASE = "data/eval/bc_dqc_chunk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", required=True,
                    help="cell label prefix, e.g. bcH8filt50k_chunk8_predh8_roll4_ddpm")
    ap.add_argument("--n-list", default="1 2 4 8 16 32 64")
    ap.add_argument("--entity", default="l2sml")
    ap.add_argument("--project", default="BC_eggplant")
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--metric-prefix", default="dqc_bon")
    ap.add_argument("--require-all", action="store_true", default=True)
    ap.add_argument("--allow-partial", dest="require_all", action="store_false")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ns = [int(x) for x in args.n_list.split()]
    rows, missing = {}, []
    for N in sorted(ns):
        p = os.path.join(
            BASE, f"{args.label}_n{N}_seed{args.start_seed}_ep{args.num_episodes}",
            "eval_log.json")
        try:
            d = json.load(open(p))
        except Exception:
            missing.append(N)
            continue
        if int(d.get("num_episodes", 0)) < args.num_episodes:
            missing.append(N)
            continue
        rows[N] = {"sr": float(d.get("success_rate", 0.0)),
                   "succ": int(d.get("num_successes", 0)),
                   "ep": int(d.get("num_episodes", 0))}

    print(f"[log_bon] ready={sorted(rows)} missing={missing}")
    for N in sorted(rows):
        r = rows[N]
        print(f"  n={N}: {r['succ']}/{r['ep']} = {r['sr']:.3f}")

    if not rows:
        print("[log_bon] nothing ready; exiting.")
        return 0
    if args.require_all and missing:
        print(f"[log_bon] {len(missing)} N not ready; not logging yet.")
        return 0

    marker = os.path.join(BASE, f".bon_logged_{args.run_id}_{args.label}")
    if os.path.exists(marker) and not args.force:
        print(f"[log_bon] already logged (marker {marker}); --force to re-log.")
        return 0

    import wandb
    run = wandb.init(entity=args.entity, project=args.project,
                     id=args.run_id, resume="allow")
    xk = f"eval/{args.metric_prefix}_N"
    yk = f"eval/{args.metric_prefix}_sr"
    wandb.define_metric(xk)
    wandb.define_metric(yk, step_metric=xk)
    wandb.define_metric(f"eval/{args.metric_prefix}_num_successes", step_metric=xk)
    table = wandb.Table(columns=["N", "successes", "episodes", "success_rate"])
    for N in sorted(rows):
        r = rows[N]
        wandb.log({xk: N, yk: r["sr"],
                   f"eval/{args.metric_prefix}_num_successes": r["succ"]})
        table.add_data(N, r["succ"], r["ep"], r["sr"])
    wandb.log({f"eval/{args.metric_prefix}_table": table})
    best = max(rows, key=lambda n: rows[n]["sr"])
    run.summary[f"eval/{args.metric_prefix}_sr_best"] = rows[best]["sr"]
    run.summary[f"eval/{args.metric_prefix}_best_N"] = best
    wandb.finish()

    with open(marker, "w") as f:
        f.write(json.dumps({str(n): rows[n] for n in sorted(rows)}, indent=2))
    print(f"[log_bon] logged {len(rows)} points to "
          f"{args.entity}/{args.project}/{args.run_id} "
          f"(best sr={rows[best]['sr']:.3f} @ N={best}); marker {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
