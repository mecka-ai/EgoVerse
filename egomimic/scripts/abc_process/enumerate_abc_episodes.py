"""Enumerate XDOF/ABC-130k episode directories from the Hugging Face Hub.

Walks the dataset's file tree (no downloads) and prints one in-repo episode
directory per line, suitable as the ``--episodes-file`` input to
``convert_abc_batch.py``. With ``--annotated-only`` it keeps only episodes that
ship an ``annotation.mcap`` (subtask labels).

Examples
--------
    # all annotated episodes across every task in the train split
    python -m egomimic.scripts.abc_process.enumerate_abc_episodes \
        --split train --annotated-only --out episodes.txt

    # a couple of tasks, capped, annotated only
    python -m egomimic.scripts.abc_process.enumerate_abc_episodes \
        --split train --tasks fold_and_stack_the_towels --annotated-only \
        --limit 3 --out episodes.txt
"""

import argparse
import os
import sys

from huggingface_hub import HfApi


def _list_dirs(api: HfApi, repo_id: str, path: str, token: str | None) -> list[str]:
    """Immediate subdirectories of ``path`` in the dataset repo."""
    items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path,
        recursive=False,
        token=token,
    )
    out = []
    for it in items:
        # GitFolderInfo has no "size"; files (GitFileInfo) do.
        if not hasattr(it, "size"):
            out.append(it.path)
    return out


def _episodes_in_task(
    api: HfApi, repo_id: str, task_path: str, annotated_only: bool, token: str | None
) -> list[str]:
    """Episode dirs under one task, via a single recursive tree walk.

    One API call per task (not per episode), so enumerating the whole dataset is
    fast. ``annotated_only`` keeps episodes whose dir contains annotation.mcap.
    """
    files = [
        it.path
        for it in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=task_path,
            recursive=True,
            token=token,
        )
        if hasattr(it, "size")  # files have "size"; folders don't
    ]
    episodes = {
        f[: -len("/episode.mcap")] for f in files if f.endswith("/episode.mcap")
    }
    if annotated_only:
        annotated = {
            f[: -len("/annotation.mcap")]
            for f in files
            if f.endswith("/annotation.mcap")
        }
        episodes &= annotated
    return sorted(episodes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="XDOF/ABC-130k")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to include (default: all tasks in the split).",
    )
    ap.add_argument(
        "--annotated-only",
        action="store_true",
        help="Keep only episodes that contain an annotation.mcap.",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Stop after N episodes (0 = no limit)."
    )
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--out", default="", help="Write to file (default: stdout).")
    args = ap.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    api = HfApi()

    tasks = args.tasks
    if not tasks:
        task_dirs = _list_dirs(api, args.repo_id, f"data/{args.split}", token)
        tasks = [d.split("/")[-1] for d in task_dirs]
    tasks = sorted(tasks)  # deterministic order so every shard agrees

    # Collect deterministically; sharding relies on identical ordering on all nodes.
    all_eps: list[str] = []
    for task in tasks:
        eps = _episodes_in_task(
            api, args.repo_id, f"data/{args.split}/{task}", args.annotated_only, token
        )
        all_eps.extend(eps)
        print(f"# {task}: {len(eps)} episodes", file=sys.stderr)
        if args.limit and len(all_eps) >= args.limit:
            all_eps = all_eps[: args.limit]
            break

    sink = open(args.out, "w") if args.out else sys.stdout
    try:
        sink.write("\n".join(all_eps) + "\n")
    finally:
        if args.out:
            sink.close()
    print(f"# enumerated {len(all_eps)} episodes", file=sys.stderr)


if __name__ == "__main__":
    main()
