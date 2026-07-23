#!/usr/bin/env python
"""Log per-checkpoint n=1 success rates into the BC training wandb run.

Reads the eval_log.json files produced by launch_unet50k_n1_evals.sh (one per
milestone checkpoint) and logs the success-rate-vs-step curve back into the
*same* wandb run that trained the policy (resume by id). The eval points are
logged against a custom ``eval/ckpt_step`` x-axis (via define_metric) so they
do not collide with wandb's monotonic global step (which already reached 50k
during training).

Idempotent: writes a marker file once it logs; re-running is a no-op unless
--force. With --require-all (default), nothing is logged until every requested
milestone has a complete eval_log -> safe to call repeatedly from a poller.

Example (filtered run):
  conda run -n monkey-verifier python scriptsv2/eval_search/log_eval_sr_to_wandb.py \
      --run-id unet_denoiser_pred16_roll8_50k_filtered \
      --label-prefix unet50k_filt \
      --milestones 10000 20000 30000 40000 50000
"""
import argparse
import json
import os
import sys

EVAL_BASE = "data/eval/search_bc_sweep"


def cell_path(label_prefix, step, start_seed, num_episodes):
    tag = f"{step // 1000}k"
    return os.path.join(
        EVAL_BASE, f"{label_prefix}_{tag}",
        f"argmax_ddim_n1_seed{start_seed}_ep{num_episodes}", "eval_log.json")


def read_sr(path, num_episodes):
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError:
        return None
    if int(d.get("num_episodes", 0)) < num_episodes:
        return None
    return {
        "successes": int(d.get("num_successes", 0)),
        "episodes": int(d.get("num_episodes", 0)),
        "success_rate": float(d.get("success_rate", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True,
                    help="wandb run id (= training logging.id / WANDB_NAME)")
    ap.add_argument("--label-prefix", required=True,
                    help="BC_LABEL prefix used by the eval launcher (e.g. unet50k_filt)")
    ap.add_argument("--milestones", type=int, nargs="+",
                    default=[10000, 20000, 30000, 40000, 50000])
    ap.add_argument("--entity", default="l2sml")
    ap.add_argument("--project", default="BC_eggplant")
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--require-all", action="store_true", default=True,
                    help="only log once every milestone has a complete eval_log")
    ap.add_argument("--allow-partial", dest="require_all", action="store_false",
                    help="log whatever milestones are ready now")
    ap.add_argument("--force", action="store_true",
                    help="re-log even if the marker file exists")
    args = ap.parse_args()

    rows = {}
    missing = []
    for step in sorted(args.milestones):
        p = cell_path(args.label_prefix, step, args.start_seed, args.num_episodes)
        r = read_sr(p, args.num_episodes)
        if r is None:
            missing.append(step)
        else:
            rows[step] = r

    print(f"[log_sr] ready={sorted(rows)} missing={missing}")
    for step in sorted(rows):
        r = rows[step]
        print(f"  {step//1000}k: {r['successes']}/{r['episodes']} = {r['success_rate']:.3f}")

    if not rows:
        print("[log_sr] nothing ready; exiting.")
        return 0
    if args.require_all and missing:
        print(f"[log_sr] --require-all set and {len(missing)} milestone(s) not ready; "
              f"not logging yet.")
        return 0

    marker = os.path.join(EVAL_BASE, f".sr_logged_{args.run_id}")
    if os.path.exists(marker) and not args.force:
        print(f"[log_sr] already logged (marker {marker}); use --force to re-log.")
        return 0

    import wandb
    run = wandb.init(entity=args.entity, project=args.project,
                     id=args.run_id, resume="allow")
    wandb.define_metric("eval/ckpt_step")
    for m in ("eval/n1_success_rate", "eval/n1_num_successes"):
        wandb.define_metric(m, step_metric="eval/ckpt_step")

    table = wandb.Table(columns=["ckpt_step", "successes", "episodes", "success_rate"])
    for step in sorted(rows):
        r = rows[step]
        wandb.log({
            "eval/ckpt_step": step,
            "eval/n1_success_rate": r["success_rate"],
            "eval/n1_num_successes": r["successes"],
        })
        table.add_data(step, r["successes"], r["episodes"], r["success_rate"])
    wandb.log({"eval/n1_sr_table": table})

    best_step = max(rows, key=lambda s: rows[s]["success_rate"])
    run.summary["eval/n1_sr_best"] = rows[best_step]["success_rate"]
    run.summary["eval/n1_sr_best_step"] = best_step
    wandb.finish()

    with open(marker, "w") as f:
        f.write(json.dumps({str(s): rows[s] for s in sorted(rows)}, indent=2))
    print(f"[log_sr] logged {len(rows)} points to {args.entity}/{args.project}/{args.run_id} "
          f"(best={rows[best_step]['success_rate']:.3f} @ {best_step}); marker {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
