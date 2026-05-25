"""Shared Modal setup: image, volumes, app, config, and git/repo helpers.

Imported by trainModal.py and curateModal.py. Contains no egomimic imports
at module level so it is safe to evaluate before the repo is cloned.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# App naming
# ---------------------------------------------------------------------------

_MODAL_APP_DEFAULT = "egomimic-training"


def app_name_from_hydra_args(hydra_args: list[str]) -> str:
    """Derive a Modal App name as ``<name>-<description>`` from Hydra CLI args.

    Sanitizes to Modal-valid characters (alphanumeric, ``-``, ``_``, ``.``),
    max 64 chars. Falls back to ``egomimic-training`` if neither key is present.
    """
    def _san(s: object) -> str:
        t = re.sub(r"[^a-zA-Z0-9_.-]", "-", str(s or "").strip())
        return t.strip("-_.") or ""

    name = description = ""
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        key = key.lstrip("+")
        if sep and key == "name":
            name = _san(val)
        elif sep and key == "description":
            description = _san(val)

    if name and description:
        label = f"{name}-{description}"
    elif name:
        label = name
    elif description:
        label = description
    else:
        return _MODAL_APP_DEFAULT

    if len(label) > 64:
        label = label[:64].rstrip("-_.")
    return label or _MODAL_APP_DEFAULT


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
# Modal compute (+modal_* launch overrides)
# ---------------------------------------------------------------------------

# Map Hydra-style CLI flags to env vars read at curateModal import time.
MODAL_COMPUTE_ARG_MAP: dict[str, str] = {
    "modal_gpu": "MODAL_GPU",
    "modal_cpu": "MODAL_CPU",
    "modal_memory_gb": "MODAL_MEMORY_GB",
    "modal_memory_mb": "MODAL_MEMORY_MB",
    "modal_volume": "MODAL_VOLUME",           # e.g. mecka_data_v2 or mecka_data_wds
    "modal_ephemeral_disk_gb": "MODAL_EPHEMERAL_DISK_GB",  # local NVMe in GB
}


@dataclass(frozen=True)
class ModalCompute:
    """GPU/CPU/RAM for a Modal ``@app.function`` container."""

    gpu: str | None
    cpu: float
    memory_mb: int

    @classmethod
    def from_environ(
        cls,
        *,
        default_gpu: str | None = "L40S",
        default_cpu: float = 16.0,
        default_memory_mb: int = 131072,
    ) -> ModalCompute:
        """Read ``MODAL_*`` env vars (set locally before ``modal run``)."""
        return cls.from_mapping(
            os.environ,
            default_gpu=default_gpu,
            default_cpu=default_cpu,
            default_memory_mb=default_memory_mb,
        )

    @classmethod
    def from_mapping(
        cls,
        env: dict[str, str],
        *,
        default_gpu: str | None = "L40S",
        default_cpu: float = 16.0,
        default_memory_mb: int = 131072,
    ) -> ModalCompute:
        """Parse ``MODAL_GPU``, ``MODAL_CPU``, ``MODAL_MEMORY_GB`` / ``MODAL_MEMORY_MB``."""
        raw_gpu = env.get("MODAL_GPU")
        if raw_gpu is None:
            gpu = default_gpu
        elif str(raw_gpu).lower() in ("none", "null", "false", ""):
            gpu = None
        else:
            gpu = str(raw_gpu)

        cpu = float(env.get("MODAL_CPU", default_cpu))
        if env.get("MODAL_MEMORY_GB"):
            memory_mb = int(float(env["MODAL_MEMORY_GB"]) * 1024)
        else:
            memory_mb = int(env.get("MODAL_MEMORY_MB", default_memory_mb))
        return cls(gpu=gpu, cpu=cpu, memory_mb=memory_mb)

    def summary(self) -> str:
        gpu_s = self.gpu if self.gpu else "none"
        return f"gpu={gpu_s}  cpu={self.cpu}  memory={self.memory_mb / 1024:.0f}GB"


# ``run_curate`` orchestrator: SQL + spawn only (fixed; ignore +modal_*).
CURATE_ORCHESTRATOR = ModalCompute(gpu=None, cpu=4.0, memory_mb=8192)


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
    .add_local_file(Path(__file__).resolve(), remote_path="/root/modal_setup.py", copy=True)
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
wds_volume = modal.Volume.from_name("mecka_data_wds", create_if_missing=True)
zip_volume = modal.Volume.from_name("mecka_data_zip", create_if_missing=True)
WDS_MOUNT_PATH = "/mnt/zarr-wds"

# Map volume name → (Modal Volume object, container mount path)
VOLUME_MAP: dict[str, tuple] = {
    "mecka_data_v2":  (zarr_volume,  "/mnt/zarr-data"),
    "mecka_data_wds": (wds_volume,   "/mnt/zarr-wds"),
    "mecka_data_zip": (zip_volume,   "/mnt/zarr-zip"),
}
training_outputs_volume = modal.Volume.from_name(
    "egoverse-training-outputs", create_if_missing=True
)
_modal_app_name = (
    os.environ.get("MODAL_APP_NAME", _MODAL_APP_DEFAULT).strip() or _MODAL_APP_DEFAULT
)
app = modal.App(_modal_app_name, image=image)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def launch_detached(
    script_path: Path,
    entrypoint: str,
    hydra_args: list[str],
    modal_env: dict | None = None,
    env: str = "robotics",
) -> None:
    """Push HEAD then fire a fully-detached Modal run.

    The local process can exit or lose connectivity after this returns —
    Modal keeps the spawned containers running to completion.
    Always use this instead of modal run without --detach or .remote() calls.
    """
    _git_commit_and_push(Path(script_path).resolve().parent.parent.parent)
    cmd = [
        sys.executable, "-m", "modal", "run",
        "--detach", "--env", env,
        f"{script_path}::{entrypoint}",
        "--", *hydra_args,
    ]
    print(f"Launching detached: {entrypoint} -- {' '.join(hydra_args)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=modal_env or os.environ.copy())
    sys.exit(result.returncode)


def _git_commit_and_push(repo_root: Path) -> None:
    """Auto-commit any local changes and push to remote before Modal submission."""
    def _run(cmd):
        return subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)

    if _run(["git", "status", "--porcelain"]).stdout.strip():
        print("Auto-committing local changes before Modal submission...")
        _run(["git", "add", "-A"])
        result = _run(["git", "commit", "--no-verify", "-m", "auto: pre-modal training commit"])
        if result.returncode != 0:
            print(f"[git commit] {result.stderr.strip()}")

    print("Pushing to remote...")
    push = _run(["git", "push", "origin", "HEAD"])
    if push.returncode != 0:
        raise RuntimeError(
            f"git push failed — cannot submit to Modal with unpushed changes:\n{push.stderr.strip()}"
        )
    print("Push complete.")


def _local_hf_token() -> str:
    """Read HuggingFace token from local auth cache (hf auth login) or ~/.egoverse_env."""
    for token_path in [
        Path("~/.cache/huggingface/token").expanduser(),
        Path("~/.huggingface/token").expanduser(),
    ]:
        if token_path.exists():
            token = token_path.read_text().strip()
            if token:
                return token
    env_file = Path("~/.egoverse_env").expanduser()
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


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


def pop_init_submodules(
    args: tuple[str, ...] | list[str],
) -> tuple[tuple[str, ...], bool]:
    """Strip ``init_submodules=…`` from launch args; return (remaining, init_submodules)."""
    init_submodules = True
    kept: list[str] = []
    for arg in args:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key == "init_submodules":
            init_submodules = val.strip().lower() not in ("false", "0", "no", "off")
        else:
            kept.append(arg)
    return tuple(kept), init_submodules


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
        path = url[len("git@github.com:"):]
        return f"https://github.com/{path}"
    return url


def _prepare_repo(
    git_remote: str,
    git_commit: str,
    *,
    init_submodules: bool = True,
) -> None:
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
        clone_cmd = ["git", "clone", clone_url, CFG.remote_repo_dir]
        if not init_submodules:
            clone_cmd = [
                "git",
                "clone",
                "--no-recurse-submodules",
                clone_url,
                CFG.remote_repo_dir,
            ]
        subprocess.run(clone_cmd, check=True)

    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "checkout", git_commit], check=True
    )
    if init_submodules:
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


def _prepare_repo_light(
    git_remote: str,
    git_commit: str,
    *,
    init_submodules: bool = False,
) -> None:
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
                ["git", "clone", "--depth=1", "--no-recurse-submodules", clone_url, str(repo_dir)],
                check=True,
            )

    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", git_commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--detach", git_commit], check=True
    )
    if init_submodules:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "submodule",
                "update",
                "--init",
                "--recursive",
            ],
            check=True,
        )
    subprocess.run(
        [CFG.python_bin, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd=str(repo_dir),
        check=True,
    )
