"""Pretty-print one or more `eval_log.json` files (output of
`eval_diffusion_policy_carrot.py`) as a side-by-side table.

Usage
-----
    python scriptsv2/summarize_evals.py \
        data/eval/<unet_run>/eval_log.json \
        data/eval/<mlp_run>/eval_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _label_from_path(path: str) -> str:
    """Try to derive a short label like `diffusion_unet` / `mlp` from the
    enclosing run directory name.
    """
    p = Path(path).resolve()
    # eval_log.json is typically at <out_dir>/eval_log.json
    candidates: List[str] = [p.parent.name, p.parent.parent.name]
    for c in candidates:
        c_lower = c.lower()
        if "diffusion_unet" in c_lower:
            return "diffusion_unet"
        if "_mlp_" in f"_{c_lower}_" or c_lower.startswith("mlp_") or c_lower.endswith("_mlp"):
            return "mlp"
    # Fallback: parent dir name
    return p.parent.name or p.name


def _load(path: str) -> Tuple[str, Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    return _label_from_path(path), data


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs",
        nargs="+",
        help="One or more eval_log.json paths.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated explicit labels (one per log, in order).",
    )
    args = parser.parse_args()

    labels: List[str] | None = None
    if args.labels is not None:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(args.logs):
            print(
                f"--labels has {len(labels)} entries but {len(args.logs)} "
                f"log files were given.",
                file=sys.stderr,
            )
            return 2

    rows: List[Tuple[str, Dict[str, Any], str]] = []
    for i, path in enumerate(args.logs):
        if not os.path.isfile(path):
            print(f"[summarize] skipping (missing): {path}", file=sys.stderr)
            continue
        auto_label, data = _load(path)
        label = labels[i] if labels else auto_label
        rows.append((label, data, path))

    if not rows:
        print("[summarize] no logs found.", file=sys.stderr)
        return 1

    header = (
        f"{'arch':<22}  "
        f"{'success_rate':>12}  "
        f"{'successes':>10}  "
        f"{'truncated':>10}  "
        f"{'episodes':>9}  "
        f"{'mean_ep_s':>9}  "
        f"{'total_s':>9}  "
        f"{'ema':>4}  "
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(" diffusion_policy SimplerEnv eval — combined summary")
    print("=" * len(header))
    print(header)
    print(sep)

    for label, d, _path in rows:
        print(
            f"{label:<22}  "
            f"{_fmt_pct(d.get('success_rate', 0.0)):>12}  "
            f"{d.get('num_successes', 0):>10}  "
            f"{d.get('num_truncated', 0):>10}  "
            f"{d.get('num_episodes', 0):>9}  "
            f"{d.get('mean_episode_time_s', 0.0):>9.2f}  "
            f"{d.get('total_time_s', 0.0):>9.1f}  "
            f"{str(bool(d.get('use_ema', False))):>4}  "
        )

    print(sep)
    print("checkpoints:")
    for label, d, path in rows:
        ckpt = d.get("checkpoint", "?")
        print(f"  - {label:<22} {ckpt}")
        print(f"    log: {path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
