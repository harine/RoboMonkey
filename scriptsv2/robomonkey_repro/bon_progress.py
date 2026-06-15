#!/usr/bin/env python3
"""Cumulative progress tracker for the resumable BoN sweep.

progress.json maps str(N) -> {"success": int, "episodes": int}. This lets a
relaunch (a) skip N values already at the target trial count and (b) append
*additional* non-overlapping episodes by running only the remaining trials and
summing the new successes in.

Subcommands:
  get_done   <progress.json> <N>                  -> prints episodes already done (0 if none)
  update     <progress.json> <N> <block_log>      -> add this block's (success, episodes) to N; prints "succ episodes rate%"
  summary    <progress.json> <out.txt> <header...> -> rewrite the human-readable summary table
"""
import json
import os
import re
import sys


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _parse_block(log):
    """Return (successes, episodes) from a run_simpler_eval block log.

    Uses the last cumulative markers the eval prints per episode:
      '# successes: A (P%)'  and  '# episodes completed so far: B'
    """
    succ = eps = 0
    with open(log, errors="ignore") as f:
        txt = f.read()
    m = re.findall(r"# successes:\s*(\d+)\s*\(", txt)
    if m:
        succ = int(m[-1])
    m = re.findall(r"# episodes completed so far:\s*(\d+)", txt)
    if m:
        eps = int(m[-1])
    return succ, eps


def main():
    cmd = sys.argv[1]
    prog = sys.argv[2]
    if cmd == "get_done":
        N = sys.argv[3]
        d = _load(prog)
        print(d.get(str(N), {}).get("episodes", 0))
    elif cmd == "update":
        N, block_log = sys.argv[3], sys.argv[4]
        d = _load(prog)
        s, e = _parse_block(block_log)
        cur = d.get(str(N), {"success": 0, "episodes": 0})
        cur["success"] += s
        cur["episodes"] += e
        d[str(N)] = cur
        _save(prog, d)
        rate = (100.0 * cur["success"] / cur["episodes"]) if cur["episodes"] else 0.0
        print(f"{cur['success']} {cur['episodes']} {rate:.1f}")
    elif cmd == "summary":
        out = sys.argv[3]
        header = " ".join(sys.argv[4:])
        d = _load(prog)
        lines = [header, "N        success_rate    n_success/n_episodes"]
        for N in sorted(d, key=lambda x: int(x)):
            c = d[N]
            rate = (100.0 * c["success"] / c["episodes"]) if c["episodes"] else 0.0
            lines.append(f"{N:<8} {rate:>6.1f}%         {c['success']}/{c['episodes']}")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    else:
        sys.exit(f"unknown subcommand: {cmd}")


if __name__ == "__main__":
    main()
