"""Filter the bridge_v2_lerobot dataset by task description and download the matching subset.

The Bridge V2 LeRobot dataset (https://huggingface.co/datasets/jesbu1/bridge_v2_lerobot)
contains 53,192 episodes spanning 19,974 freeform language tasks. This script:

1. Downloads the small `meta/` files (info / tasks / episodes).
2. Filters tasks by user-supplied keyword groups (e.g. carrot+plate, eggplant+basket).
3. Resolves the matching episode indices.
4. Selectively downloads only those episodes' `.parquet` data and `.mp4` camera videos,
   preserving the upstream chunked layout so the result loads as a normal LeRobot dataset.
5. Writes filtered `meta/tasks.jsonl`, `meta/episodes.jsonl`, and an updated `meta/info.json`
   alongside a per-group `filter_summary.json`.

Usage:
    python scriptsv2/filter_bridge_v2.py \
        --output_dir data/bridge_v2_filtered \
        --task_groups "carrot_on_plate=carrot,plate" "eggplant_in_basket=eggplant,basket"

A task matches a group when *all* keywords appear in its description (case-insensitive).
Use --exclude to drop unwanted matches (e.g. "take carrot off plate"). Use --dry_run to
inspect matches before downloading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ID = "jesbu1/bridge_v2_lerobot"
REPO_TYPE = "dataset"
CAMERA_KEYS = ("image_0", "image_1", "image_2", "image_3")


def _require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # noqa: F401
    except ImportError:
        sys.exit(
            "huggingface_hub is required. Install with:\n"
            "    pip install 'huggingface_hub[hf_transfer]'"
        )


def download_meta(local_dir: Path) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    files = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]
    paths = []
    for f in files:
        p = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=f,
            local_dir=str(local_dir),
        )
        paths.append(Path(p))
    return paths[0], paths[1], paths[2]


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def parse_task_groups(specs: Sequence[str]) -> Dict[str, List[str]]:
    """Parse ``name=kw1,kw2`` specs into ``{name: [kw1, kw2]}``."""
    groups: Dict[str, List[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid task group spec '{spec}'. Expected 'name=kw1[,kw2,...]'."
            )
        name, kw_str = spec.split("=", 1)
        kws = [k.strip().lower() for k in kw_str.split(",") if k.strip()]
        if not kws:
            raise ValueError(f"No keywords in spec '{spec}'.")
        if not name.strip():
            raise ValueError(f"Empty group name in spec '{spec}'.")
        groups[name.strip()] = kws
    return groups


def match_task_indices(
    tasks: Sequence[dict],
    keywords: Sequence[str],
    excludes: Sequence[str],
) -> List[int]:
    """Indices of tasks whose description contains all keywords and none of the excludes."""
    out: List[int] = []
    excludes = [e.lower() for e in excludes]
    for entry in tasks:
        text = (entry.get("task") or "").lower()
        if not text:
            continue
        if all(k in text for k in keywords) and not any(x in text for x in excludes):
            out.append(int(entry["task_index"]))
    return out


def episodes_for_tasks(
    episodes: Sequence[dict],
    matched_task_strings: set,
) -> List[dict]:
    """Episodes whose `tasks` list contains any of the matched task strings."""
    out: List[dict] = []
    for ep in episodes:
        if any(t in matched_task_strings for t in (ep.get("tasks") or [])):
            out.append(ep)
    return out


def episode_file_patterns(
    episode_indices: Sequence[int],
    chunk_size: int,
) -> List[str]:
    patterns: List[str] = []
    for ei in episode_indices:
        chunk = ei // chunk_size
        patterns.append(f"data/chunk-{chunk:03d}/episode_{ei:06d}.parquet")
        for cam in CAMERA_KEYS:
            patterns.append(
                f"videos/chunk-{chunk:03d}/observation.images.{cam}/episode_{ei:06d}.mp4"
            )
    return patterns


def download_episode_files(
    episode_indices: Sequence[int],
    chunk_size: int,
    local_dir: Path,
    max_workers: int,
) -> None:
    from huggingface_hub import snapshot_download

    if not episode_indices:
        print("  (nothing to download)")
        return

    patterns = episode_file_patterns(episode_indices, chunk_size)
    print(
        f"  Downloading {len(episode_indices)} episode(s) "
        f"= up to {len(patterns)} files (parquet + 4 cameras each)..."
    )
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        allow_patterns=patterns,
        max_workers=max_workers,
    )


def write_filtered_meta(
    output_dir: Path,
    info: dict,
    matched_episodes: List[dict],
    matched_task_indices: set,
    tasks_by_index: Dict[int, str],
) -> None:
    """Rewrite meta files to describe only the filtered subset.

    The original `info.json` is overwritten in-place with updated totals; the original
    chunked layout is preserved (episode/chunk indices stay the same as upstream).
    """
    meta_dir = output_dir / "meta"

    filtered_tasks = [
        {"task_index": ti, "task": tasks_by_index[ti]}
        for ti in sorted(matched_task_indices)
        if ti in tasks_by_index
    ]
    write_jsonl(meta_dir / "tasks.jsonl", filtered_tasks)

    filtered_episodes_sorted = sorted(matched_episodes, key=lambda e: int(e["episode_index"]))
    write_jsonl(meta_dir / "episodes.jsonl", filtered_episodes_sorted)

    info_out = dict(info)
    info_out["total_episodes"] = len(filtered_episodes_sorted)
    info_out["total_frames"] = int(sum(int(e["length"]) for e in filtered_episodes_sorted))
    info_out["total_tasks"] = len(filtered_tasks)
    n_cams = len(CAMERA_KEYS)
    info_out["total_videos"] = len(filtered_episodes_sorted) * n_cams
    if filtered_episodes_sorted:
        last_ep = int(filtered_episodes_sorted[-1]["episode_index"])
        chunk_size = int(info.get("chunks_size", 1000))
        info_out["total_chunks"] = (last_ep // chunk_size) + 1
    else:
        info_out["total_chunks"] = 0
    info_out["splits"] = {"train": f"0:{info_out['total_episodes']}"}

    (meta_dir / "info.json").write_text(json.dumps(info_out, indent=2))


def main() -> int:
    _require_hf_hub()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/bridge_v2_filtered"),
        help="Local directory for the filtered LeRobot dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--task_groups",
        nargs="+",
        default=[
            "carrot_on_plate=carrot,plate",
            "eggplant_in_basket=eggplant,basket",
        ],
        help=(
            "One or more 'name=kw1,kw2,...' specs. A task matches a group when ALL "
            "keywords appear (case-insensitive) in its description."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["take carrot off"],
        help="Substrings whose presence in a task description disqualifies it.",
    )
    parser.add_argument(
        "--max_episodes_per_group",
        type=int,
        default=None,
        help="If set, cap the number of episodes downloaded per group.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print matches and write the filter summary, but do not download episode files.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Parallel download workers for huggingface_hub (default: %(default)s).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading meta files from {REPO_ID} -> {args.output_dir}/meta/")
    info_path, tasks_path, episodes_path = download_meta(args.output_dir)
    info = json.loads(info_path.read_text())
    tasks = load_jsonl(tasks_path)
    episodes = load_jsonl(episodes_path)
    tasks_by_index: Dict[int, str] = {
        int(t["task_index"]): t.get("task", "") for t in tasks
    }
    chunk_size = int(info.get("chunks_size", 1000))
    print(
        f"  upstream: total_tasks={len(tasks)}, total_episodes={len(episodes)}, "
        f"total_frames={info.get('total_frames')}, chunks_size={chunk_size}"
    )

    print("[2/4] Matching tasks per group")
    groups = parse_task_groups(args.task_groups)
    summary: Dict[str, dict] = {}
    all_episode_indices: set = set()
    all_task_indices: set = set()
    all_matched_episodes: List[dict] = []
    seen_episode_indices: set = set()

    for name, kws in groups.items():
        ti = match_task_indices(tasks, kws, args.exclude)
        matched_strings = {tasks_by_index[i] for i in ti if i in tasks_by_index}
        eps = episodes_for_tasks(episodes, matched_strings)
        if args.max_episodes_per_group is not None:
            eps = eps[: args.max_episodes_per_group]
        ep_indices = [int(e["episode_index"]) for e in eps]
        total_frames = int(sum(int(e["length"]) for e in eps))

        summary[name] = {
            "keywords": list(kws),
            "excludes": list(args.exclude),
            "n_tasks": len(ti),
            "n_episodes": len(ep_indices),
            "total_frames": total_frames,
            "task_indices": ti,
            "task_strings": [tasks_by_index[i] for i in ti],
            "episode_indices": ep_indices,
        }

        print(
            f"  [{name}] keywords={kws} -> "
            f"{len(ti)} task(s), {len(ep_indices)} episode(s), {total_frames} frame(s)"
        )
        for tindex, tstr in list(zip(ti, summary[name]["task_strings"]))[:10]:
            print(f"      task_index={tindex}: {tstr!r}")
        if len(ti) > 10:
            print(f"      ... and {len(ti) - 10} more task(s)")
        if not ti:
            print(
                f"  WARNING: 0 tasks matched keywords={kws}. "
                "Consider broadening with different --task_groups."
            )

        all_task_indices.update(ti)
        for ep in eps:
            ei = int(ep["episode_index"])
            if ei not in seen_episode_indices:
                seen_episode_indices.add(ei)
                all_matched_episodes.append(ep)
        all_episode_indices.update(ep_indices)

    summary_path = args.output_dir / "filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {summary_path}")

    print("[3/4] Writing filtered meta/ files")
    write_filtered_meta(
        args.output_dir,
        info=info,
        matched_episodes=all_matched_episodes,
        matched_task_indices=all_task_indices,
        tasks_by_index=tasks_by_index,
    )
    print(
        f"  meta/tasks.jsonl -> {len(all_task_indices)} task(s); "
        f"meta/episodes.jsonl -> {len(all_matched_episodes)} episode(s)"
    )

    if args.dry_run:
        print("[4/4] --dry_run set; skipping data/video download.")
        return 0

    print("[4/4] Downloading episode parquet + video files")
    download_episode_files(
        sorted(all_episode_indices),
        chunk_size=chunk_size,
        local_dir=args.output_dir,
        max_workers=args.max_workers,
    )
    print(f"Done. Filtered dataset is at: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
