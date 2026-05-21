"""Shared Modal setup: image, volumes, app, config, and git/repo helpers.

Imported by trainModal.py and curateModal.py. Contains no egomimic imports
at module level so it is safe to evaluate before the repo is cloned.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    remote_repo_dir: str = "/root/EgoVerse"
    python_bin: str = "python3"

    @property
    def train_script(self) -> str:
        return f"{self.remote_repo_dir}/egomimic/trainHydra.py"

    volume_mount_path: str = "/mnt/zarr-data"
    output_mount_path: str = "/root/EgoVerse/logs"

    gpu: str = field(default_factory=lambda: os.environ.get("MODAL_GPU", "A100"))
    cpu: float = field(
        default_factory=lambda: float(os.environ.get("MODAL_CPU", "12.0"))
    )
    memory_mb: int = field(
        default_factory=lambda: (
            int(float(os.environ.get("MODAL_MEMORY_GB")) * 1024)
            if os.environ.get("MODAL_MEMORY_GB")
            else int(os.environ.get("MODAL_MEMORY_MB", "65536"))
        )
    )
    timeout_seconds: int = 86400  # 24 h (Modal max)

    secret_names: list[str] = field(
        default_factory=lambda: [
            "egoverse-r2",
            "egoverse-mongodb",
            "egoverse-db",
            "egoverse-sql",
        ]
    )


CFG = _Config()


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install("git")
    # Bake modal_setup.py into the image so curateModal.py / trainModal.py can
    # import it at module-load time (before the repo is cloned via _prepare_repo).
    # Path(__file__).parent resolves to /root/ in the container, so:
    #   from modal_setup import (...)  works in both local and remote contexts.
    .add_local_file(
        Path(__file__).resolve(), remote_path="/root/modal_setup.py", copy=True
    )
    .pip_install(
        "lightning",
        "hydra-core",
        "omegaconf",
        "wandb",
        "boto3",
        "cloudpathlib",
        "zarr==3.1.5",
        "pyarrow",
        "simplejpeg",
        "h5py",
        "av==12.0.0",
        "mediapy",
        "datasets==4.0.0",
        "transformers==4.57.3",
        "timm",
        "einops",
        "positional-encodings[pytorch]",
        "pytorch-kinematics",
        "arm-pytorch-utilities",
        "geomloss",
        "tslearn",
        "scipy",
        "hydra-submitit-launcher==1.2.0",
        "submitit",
        "opencv-python-headless",
        "projectaria-tools",
        "pyquaternion",
        "sqlalchemy",
        "psycopg[binary]",
        "pandas",
        "rich",
        "tabulate",
        "prettytable",
        "packaging",
        "overrides",
        "typing_extensions",
        "pyyaml",
        "matplotlib",
        "termcolor",
        "tqdm",
        "filelock",
        "imageio",
        "imageio-ffmpeg",
        "safetensors",
        "huggingface-hub",
        "scaleapi",
        "openai",
        "pyzmq",
        "torchvision==0.21.0",
        "s5cmd",
    )
)

zarr_volume = modal.Volume.from_name("mecka_data_v2")
training_outputs_volume = modal.Volume.from_name(
    "egoverse-training-outputs", create_if_missing=True
)
app = modal.App("egomimic-training", image=image)


# ---------------------------------------------------------------------------
# Pause-precompute fan-out worker
# ---------------------------------------------------------------------------
# Lives on the training app so trainHydra.py (running inside run_hydra_train)
# can .map() it without depending on a separately-deployed companion app.
# The image is intentionally minimal — zarr + numpy only, no egomimic imports
# — so cold starts are fast (each shard is a few seconds of I/O + numpy).

pause_precompute_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "zarr==3.1.5", "numpy"
)


@app.function(
    image=pause_precompute_image,
    cpu=2.0,
    memory=4096,
    timeout=900,
    volumes={"/mnt/zarr-data": zarr_volume},
    max_containers=100,
)
def pause_precompute_shard(
    episodes: list[tuple[str, str]],
    epsilon: float,
) -> list[tuple[str, int, list[int]]]:
    """Compute per-episode pause keep_indices for a shard.

    Args:
        episodes: list of (episode_hash, episode_path_str).
        epsilon: L2 threshold for the "frame is paused" test (m, per hand).

    Returns:
        list of (episode_hash, raw_total_frames, keep_indices). For episodes
        missing the obs_ee_pose keys or that fail to open, raw is 0 and
        indices is empty — the caller treats this as "skip".
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import zarr

    LEFT_KEY = "left.obs_ee_pose"
    RIGHT_KEY = "right.obs_ee_pose"

    zarr_volume.reload()

    def _mask(left_pose, right_pose):
        T = len(left_pose)
        if T < 2:
            return np.ones(T, dtype=bool)
        left_d = np.linalg.norm(np.diff(left_pose, axis=0), axis=-1)
        right_d = np.linalg.norm(np.diff(right_pose, axis=0), axis=-1)
        is_paused = (left_d < epsilon) & (right_d < epsilon)
        keep = np.ones(T, dtype=bool)
        in_pause = False
        for t in range(1, T):
            if is_paused[t - 1]:
                if in_pause:
                    keep[t] = False
                else:
                    in_pause = True
            else:
                in_pause = False
        return keep

    def _one(item):
        episode_hash, path_str = item
        try:
            store = zarr.open_group(path_str, mode="r")
            left = np.asarray(store[LEFT_KEY][:])
            right = np.asarray(store[RIGHT_KEY][:])
        except Exception:
            return (episode_hash, 0, [])
        try:
            keep = _mask(left, right)
            indices = np.flatnonzero(keep).astype(np.int64).tolist()
            return (episode_hash, int(left.shape[0]), indices)
        except Exception:
            return (episode_hash, 0, [])

    out: list[tuple[str, int, list[int]]] = []
    with ThreadPoolExecutor(max_workers=64) as ex:
        for r in ex.map(_one, episodes):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _git_commit_and_push(repo_root: Path) -> None:
    """Auto-commit any local changes and push to remote before Modal submission."""

    def _run(cmd):
        return subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)

    if _run(["git", "status", "--porcelain"]).stdout.strip():
        print("Auto-committing local changes before Modal submission...")
        _run(["git", "add", "-A"])
        result = _run(
            ["git", "commit", "--no-verify", "-m", "auto: pre-modal training commit"]
        )
        if result.returncode != 0:
            print(f"[git commit] {result.stderr.strip()}")

    print("Pushing to remote...")
    push = _run(["git", "push", "origin", "HEAD"])
    if push.returncode != 0:
        raise RuntimeError(
            f"git push failed — cannot submit to Modal with unpushed changes:\n{push.stderr.strip()}"
        )
    print("Push complete.")


def _local_wandb_key() -> str:
    """Read WANDB_API_KEY from the local environment or ~/.egoverse_env."""
    env_file = Path("~/.egoverse_env").expanduser()
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("WANDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["WANDB_API_KEY"] = key
    key = os.environ.get("WANDB_API_KEY", "")
    if not key:
        print(
            "Warning: WANDB_API_KEY not set locally — W&B logging will be disabled. "
            "Add it to ~/.egoverse_env to enable."
        )
    return key


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def _resolve_git_state() -> tuple[str, str, bool]:
    """Return (remote_url, commit_sha, is_dirty).

    Raises SystemExit if HEAD has not been pushed to the remote.
    """
    git_remote = _git_output(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    is_dirty = bool(_git_output(["git", "status", "--porcelain"]))

    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "branch", "-r", "--contains", git_commit],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            raise SystemExit(
                f"ERROR: commit {git_commit[:12]} has not been pushed to {git_remote}.\n"
                "Push your branch first, then re-run.\n"
                "  git push --set-upstream origin <branch>"
            )
    except subprocess.CalledProcessError:
        pass

    return git_remote, git_commit, is_dirty


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def _ssh_to_https(url: str) -> str:
    """Convert git@github.com:org/repo.git → https://github.com/org/repo.git"""
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:") :]
        return f"https://github.com/{path}"
    return url


def _prepare_repo(git_remote: str, git_commit: str) -> None:
    """Clone (or update) the repo and check out the exact commit."""
    clone_url = _ssh_to_https(git_remote)
    repo_dir = Path(CFG.remote_repo_dir)

    if (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "--all", "--tags"],
            check=True,
        )
    elif repo_dir.exists():
        subprocess.run(["git", "init", CFG.remote_repo_dir], check=True)
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "remote", "add", "origin", clone_url],
            check=True,
        )
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "origin", "--tags"],
            check=True,
        )
    else:
        subprocess.run(["git", "clone", clone_url, CFG.remote_repo_dir], check=True)

    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "checkout", git_commit], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            CFG.remote_repo_dir,
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        check=True,
    )
    subprocess.run(
        [CFG.python_bin, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd=CFG.remote_repo_dir,
        check=True,
    )


def _prepare_repo_light(git_remote: str, git_commit: str) -> None:
    """Shallow clone without submodules — faster setup for curation shard workers."""
    clone_url = _ssh_to_https(git_remote)
    repo_dir = Path(CFG.remote_repo_dir)

    if not (repo_dir / ".git").exists():
        if repo_dir.exists():
            subprocess.run(["git", "init", str(repo_dir)], check=True)
            subprocess.run(
                ["git", "-C", str(repo_dir), "remote", "add", "origin", clone_url],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--no-recurse-submodules",
                    clone_url,
                    str(repo_dir),
                ],
                check=True,
            )

    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", git_commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--detach", git_commit], check=True
    )
    subprocess.run(
        [CFG.python_bin, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd=str(repo_dir),
        check=True,
    )
