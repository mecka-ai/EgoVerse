"""Modal training entrypoints for EgoVerse.

Usage
-----
Via trainHydra.py (recommended):
    python egomimic/trainHydra.py data=mecka_all_zarr trainer=ddp_modal \\
        logger=wandb model=hpt_bc_flow_mecka name=<run> description=<desc> \\
        [+modal_gpu=H100] [+modal_cpu=32] [+modal_memory_gb=128]

Direct (fire-and-forget):
    modal run --env robotics egomimic/modal/test_run.py::submit -- \\
        data=mecka_all_zarr trainer=ddp_modal logger=wandb model=hpt_bc_flow_mecka

Direct (blocking — streams logs, downloads artifacts when done):
    modal run --env robotics egomimic/modal/test_run.py::run -- \\
        data=mecka_all_zarr trainer=ddp_modal logger=wandb model=hpt_bc_flow_mecka

Volume note
-----------
The zarr dataset volume (egoverse-zarr-data) is mounted at /mnt/zarr-data.
Use LocalEpisodeResolver with folder_path=/mnt/zarr-data (see mecka_all_zarr.yaml).

Training outputs (checkpoints, logs, norm stats, videos) are written to
/root/EgoVerse/logs inside the container, which is backed by the Modal volume
egoverse-training-outputs. After a blocking `run`, artifacts are automatically
downloaded to ./modal-outputs/<name>/<desc>_<timestamp>/ on your local machine.
For fire-and-forget `submit` jobs, pull manually:
    modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/

Notes
-----
- This file is intentionally self-contained (no egomimic imports at module level)
  because Modal mounts it as /root/test_run.py before the repo is cloned.
- Modal runs the *committed* git state. Commit before submitting a real run.
- Secrets must exist in the Modal dashboard (robotics env):
    egoverse-r2      → R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET
    egoverse-mongodb → MONGODB_URI
- WANDB_API_KEY is per-user: add it to ~/.egoverse_env on your local machine.
- GPU/CPU/memory are set via MODAL_GPU / MODAL_CPU / MODAL_MEMORY_GB env vars
  (trainHydra.py sets these from +modal_gpu= / +modal_cpu= / +modal_memory_gb= overrides).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Route CLI to robotics env by default
# ---------------------------------------------------------------------------
os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Inline config (duplicated from modal_config.py so this file is standalone)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class _Config:
    remote_repo_dir: str = "/root/EgoVerse"
    python_bin: str = "python3"

    @property
    def train_script(self) -> str:
        return f"{self.remote_repo_dir}/egomimic/trainHydra.py"

    volume_mount_path: str = "/mnt/zarr-data"
    # Training outputs (checkpoints, logs, norm stats, videos) are persisted here
    output_mount_path: str = "/root/EgoVerse/logs"

    # Overridable via env vars (set by trainHydra.py from +modal_gpu= etc.)
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
        default_factory=lambda: ["egoverse-r2", "egoverse-mongodb", "egoverse-db", "egoverse-sql"]
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
    .pip_install(
        # Core training stack
        "lightning",
        "hydra-core",
        "omegaconf",
        "wandb",
        # Data / storage
        "boto3",
        "cloudpathlib",
        "zarr==3.1.5",
        "pyarrow",
        "simplejpeg",
        "h5py",
        "av==12.0.0",
        "mediapy",
        "datasets==4.0.0",
        # Model / ML
        "transformers==4.57.3",
        "timm",
        "einops",
        "positional-encodings[pytorch]",
        "pytorch-kinematics",
        "arm-pytorch-utilities",
        "geomloss",
        "tslearn",
        "scipy",
        # Hydra plugins
        "hydra-submitit-launcher==1.2.0",
        "submitit",
        # Vision
        "opencv-python-headless",
        "projectaria-tools",
        "pyquaternion",
        # Database / infra
        "sqlalchemy",
        "psycopg[binary]",
        "pandas",
        # Utilities
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
        "datasets==4.0.0",
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
# Local helpers (execute on the submitting machine)
# ---------------------------------------------------------------------------


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

    Raises SystemExit if HEAD has not been pushed to the remote — the container
    clones from GitHub so unpushed commits will cause a checkout failure there.
    """
    git_remote = _git_output(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    is_dirty = bool(_git_output(["git", "status", "--porcelain"]))

    # Verify the commit exists on the remote before submitting
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
        pass  # fetch failed (no network / auth) — let the container surface the error

    return git_remote, git_commit, is_dirty


def _build_train_cmd(hydra_args: tuple[str, ...]) -> list[str]:
    return [CFG.python_bin, CFG.train_script, *hydra_args]


def _resolve_volume_paths(hydra_args: tuple[str, ...]) -> tuple[str, ...]:
    """Prepend the output volume mount path to relative values for path-bearing keys.

    Allows callers to pass  ckpt_path=mecka_modal/run/checkpoints/last.ckpt
    instead of the full container path  ckpt_path=/root/EgoVerse/logs/...
    """
    _PATH_KEYS = {"ckpt_path", "norm_stats.precomputed_norm_path"}
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if (
            sep
            and key in _PATH_KEYS
            and val
            and val != "null"
            and not val.startswith("/")
        ):
            val = f"{CFG.output_mount_path}/{val}"
            arg = f"{key}={val}"
        fixed.append(arg)
    return tuple(fixed)


def _download_run_artifacts(output_rel_path: str) -> None:
    """Download artifacts from the training outputs volume to ./modal-outputs/ locally."""
    local_dest = REPO_ROOT / "modal-outputs" / output_rel_path
    local_dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts to {local_dest} ...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--env",
            "robotics",
            "egoverse-training-outputs",
            output_rel_path,
            str(local_dest),
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode == 0:
        print(f"Artifacts saved to: {local_dest.resolve()}")
    else:
        print(
            f"Download failed — pull manually:\n"
            f"  modal volume get --env robotics egoverse-training-outputs "
            f'"{output_rel_path}" "{local_dest}"'
        )


# ---------------------------------------------------------------------------
# Container helpers (execute inside the Modal container)
# ---------------------------------------------------------------------------


def _ssh_to_https(url: str) -> str:
    """Convert git@github.com:org/repo.git → https://github.com/org/repo.git"""
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:") :]
        return f"https://github.com/{path}"
    return url


def _prepare_repo(git_remote: str, git_commit: str) -> None:
    """Clone (or update) the repo and check out the exact commit."""
    # The container has no SSH keys — always use HTTPS for cloning
    clone_url = _ssh_to_https(git_remote)
    repo_dir = Path(CFG.remote_repo_dir)

    if (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "--all", "--tags"],
            check=True,
        )
    elif repo_dir.exists():
        # Directory already exists but has no .git — happens when a volume is
        # mounted at a subdirectory (e.g. /root/EgoVerse/logs), which causes
        # Modal to create the parent before we clone. Use init+fetch instead.
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
        subprocess.run(
            ["git", "clone", clone_url, CFG.remote_repo_dir],
            check=True,
        )

    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "checkout", git_commit],
        check=True,
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

    # --no-deps: packages come from the image; just registers egomimic in sys.path
    subprocess.run(
        [CFG.python_bin, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd=CFG.remote_repo_dir,
        check=True,
    )


# ---------------------------------------------------------------------------
# Lightweight repo setup for shard workers (no submodules needed for loading)
# ---------------------------------------------------------------------------


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
                    "git", "clone", "--depth=1", "--no-recurse-submodules",
                    clone_url, str(repo_dir),
                ],
                check=True,
            )

    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", git_commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--detach", git_commit],
        check=True,
    )
    subprocess.run(
        [CFG.python_bin, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd=str(repo_dir),
        check=True,
    )


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------


@app.function(
    gpu=CFG.gpu,
    cpu=CFG.cpu,
    memory=CFG.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_hydra_train(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> str:
    """Clone the repo at *git_commit* and run trainHydra.py with *hydra_args*.

    Returns the path relative to the output volume root where artifacts were written
    (e.g. 'mecka_modal/test_2026-05-11_20-35-51'), or empty string on failure.
    """
    import glob

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)

    hydra_args = _resolve_volume_paths(hydra_args)
    cmd = _build_train_cmd(hydra_args)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    # Expose container context so ModalAutoRestartCallback can spawn a continuation job
    import json as _json
    import time as _time

    env["MODAL_IS_REMOTE"] = "1"
    env["MODAL_TIMEOUT_SECONDS"] = str(CFG.timeout_seconds)
    env["MODAL_START_TIME"] = str(_time.time())
    env["MODAL_HYDRA_ARGS"] = _json.dumps(list(hydra_args))
    env["MODAL_GIT_REMOTE"] = git_remote
    env["MODAL_GIT_COMMIT"] = git_commit

    print(f"Running: {shlex.join(cmd)}")

    process = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

    zarr_volume.commit()

    # Find the most recently modified run directory and persist it to the volume
    all_run_dirs = sorted(
        glob.glob(f"{CFG.output_mount_path}/*/*"),
        key=os.path.getmtime,
    )
    output_rel_path = (
        os.path.relpath(all_run_dirs[-1], CFG.output_mount_path) if all_run_dirs else ""
    )
    training_outputs_volume.commit()

    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed (exit {process.returncode}): {shlex.join(cmd)}"
        )

    return output_rel_path


# ---------------------------------------------------------------------------
# DemInf curation — shard worker (one per container, parallel loading)
# ---------------------------------------------------------------------------

@app.function(
    gpu=None,
    cpu=4,
    memory=16384,
    timeout=1800,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={CFG.volume_mount_path: zarr_volume},
)
def _load_shard(
    path_hash_pairs: list,
    git_remote: str,
    git_commit: str,
) -> list:
    """Load a shard of episodes from zarr. Returns serialised episode dicts."""
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as _Path
    import sys as _sys

    _prepare_repo_light(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)

    from egomimic.curation.utils import load_episode_from_path

    def _load_one(pair):
        path_str, episode_hash = pair
        ep = load_episode_from_path(_Path(path_str), episode_hash=episode_hash)
        if ep is None:
            return None
        return {
            "episode_hash": ep.episode_hash,
            "observations": ep.observations,
            "actions": ep.actions,
            "embodiment": ep.embodiment,
        }

    n_threads = min(16, max(1, len(path_hash_pairs)))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_load_one, path_hash_pairs))

    kept = [r for r in results if r is not None]
    print(f"[shard] loaded {len(kept)}/{len(path_hash_pairs)} episodes")
    return kept


# ---------------------------------------------------------------------------
# DemInf curation orchestrator — fan-out loading + serial KSG
# GPU only needed when StateEmbedder mode="image"; override with +modal_gpu=A100
# ---------------------------------------------------------------------------

_CURATE_GPU = os.environ.get("MODAL_GPU") or None  # no GPU by default for KSG
_CURATE_CPU = float(os.environ.get("MODAL_CPU", "32"))
_CURATE_MEMORY_MB = (
    int(float(os.environ.get("MODAL_MEMORY_GB")) * 1024)
    if os.environ.get("MODAL_MEMORY_GB")
    else int(os.environ.get("MODAL_MEMORY_MB", "65536"))
)


@app.function(
    gpu=_CURATE_GPU,
    cpu=_CURATE_CPU,
    memory=_CURATE_MEMORY_MB,
    timeout=CFG.timeout_seconds,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_curate(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> str:
    """Fan-out episode loading (300 containers) + serial KSG curation."""
    import sys as _sys
    import time as _time
    import numpy as _np
    from pathlib import Path as _Path

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)

    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    # ── 1. Load Hydra config ──────────────────────────────────────────────────
    import hydra as _hydra
    from hydra import compose

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = compose("curate", overrides=list(hydra_args))

    from egomimic.rldb.zarr.utils import set_global_seed
    import lightning as _L

    if cfg.get("seed"):
        _L.seed_everything(cfg.seed, workers=True)
        set_global_seed(cfg.seed)

    from egomimic.utils.aws.aws_data_utils import load_env
    load_env()

    # ── 2. Scan for episode paths (uses existing Modal fan-out scan) ──────────
    from omegaconf import OmegaConf
    from egomimic.rldb.filters import DatasetFilter
    from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver

    train_datasets = OmegaConf.to_container(
        cfg.data.train_datasets, resolve=True, throw_on_missing=False
    )
    all_paths: list = []
    for ds_name, ds_cfg in train_datasets.items():
        folder_path = ((ds_cfg or {}).get("resolver") or {}).get("folder_path")
        if not folder_path:
            continue
        filter_lambdas = list(
            (((ds_cfg or {}).get("filters") or {}).get("filter_lambdas")) or []
        )
        try:
            pairs = LocalEpisodeResolver._get_local_filtered_paths(
                search_path=Path(folder_path),
                filters=DatasetFilter(filter_lambdas),
            )
            all_paths.extend(pairs)
            print(f"[scan] {ds_name}: {len(pairs)} paths from {folder_path}")
        except Exception as exc:
            print(f"[scan] {ds_name}: failed — {exc}")

    total = len(all_paths)
    print(f"Total episode paths: {total}")
    if total == 0:
        print("No episodes found — check resolver.folder_path in data config")
        return ""

    # ── 3. Fan-out loading: up to 300 containers, ~660 episodes each ─────────
    N_SHARDS = min(300, total)
    shard_size = (total + N_SHARDS - 1) // N_SHARDS
    shards = [all_paths[i : i + shard_size] for i in range(0, total, shard_size)]
    print(
        f"Fan-out loading: {len(shards)} containers × ~{shard_size} episodes "
        f"(16 threads each)"
    )

    t0 = _time.time()
    episode_dicts: list = []
    for shard_result in _load_shard.starmap(
        [(s, git_remote, git_commit) for s in shards]
    ):
        episode_dicts.extend(shard_result)
    print(f"Loaded {len(episode_dicts)} episodes in {_time.time() - t0:.1f}s")

    # ── 4. Reconstruct Episode objects ────────────────────────────────────────
    from egomimic.curation.utils import Episode

    episodes = [
        Episode(
            episode_hash=d["episode_hash"],
            observations=_np.asarray(d["observations"], dtype=_np.float32),
            actions=_np.asarray(d["actions"], dtype=_np.float32),
            embodiment=d["embodiment"],
        )
        for d in episode_dicts
    ]
    del episode_dicts
    print(f"Reconstructed {len(episodes)} Episode objects")

    # ── 5. WandB ─────────────────────────────────────────────────────────────
    from egomimic.utils.instantiators import instantiate_loggers

    loggers = instantiate_loggers(cfg.get("logger"))
    wandb_run = None
    for lgr in loggers:
        if hasattr(lgr, "experiment"):
            try:
                wandb_run = lgr.experiment
            except Exception:
                pass
            break

    # ── 6. Run curation pipeline ──────────────────────────────────────────────
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        _Path(CFG.output_mount_path) / cfg.name / f"{cfg.description}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Instantiating algo <{cfg.model._target_}>")
    algo = _hydra.utils.instantiate(cfg.model)

    result = algo.curate(episodes, output_dir=output_dir, wandb_run=wandb_run)
    print(
        f"Curation done — kept={len(result.kept_hashes)}  "
        f"removed={len(result.all_removed_hashes)}  output={output_dir}"
    )

    zarr_volume.commit()
    training_outputs_volume.commit()

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    return str(output_dir)


# ---------------------------------------------------------------------------
# Container health-check function
# ---------------------------------------------------------------------------


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=900,  # 15 min per shard
    volumes={CFG.volume_mount_path: zarr_volume},
)
def scan_shard(
    episode_names: list[str],
    filter_lambdas: list[str],
) -> list[tuple[str, str]]:
    """Scan a shard of zarr episode dirs and return matched (path, episode_hash) pairs.

    Self-contained: no egomimic imports — reads .zattrs JSON directly and
    re-evals filter lambda strings to reconstruct DatasetFilter behavior.
    Mirrors `_normalize_filter_row` + `DatasetFilter.matches` from the egomimic
    side; keep in sync if those change.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    zarr_volume.reload()

    predicates = [eval(expr) for expr in filter_lambdas]
    base = Path(CFG.volume_mount_path)

    def _read_metadata(name: str):
        p = base / name
        zattrs = p / ".zattrs"
        try:
            if zattrs.is_file():
                with zattrs.open("rb") as f:
                    return json.load(f)
            import zarr

            store = zarr.open_group(str(p), mode="r")
            return dict(store.attrs)
        except Exception:
            return None

    def _matches(metadata: dict, episode_hash: str) -> bool:
        row = dict(metadata)
        row["episode_hash"] = episode_hash
        v = row.get("is_deleted")
        if v is None or v == "":
            row["is_deleted"] = False
        if row.get("is_deleted"):
            return False
        for pred in predicates:
            if not pred(row):
                return False
        return True

    def _process(name: str):
        episode_hash = name[:-5] if name.endswith(".zarr") else name
        metadata = _read_metadata(name)
        if metadata is None:
            return None
        if _matches(metadata, episode_hash):
            return (str(base / name), episode_hash)
        return None

    matched: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for result in executor.map(_process, episode_names):
            if result is not None:
                matched.append(result)

    return matched


@app.function(
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={CFG.volume_mount_path: zarr_volume},
    timeout=120,
)
def _health_check() -> dict:
    """Verify secrets, DB, R2 credentials, and volume mount from inside the container."""
    results = {}

    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL"):
        results[key] = "OK" if os.environ.get(key) else "MISSING"

    results["MONGODB_URI"] = "OK" if os.environ.get("MONGODB_URI") else "MISSING"

    probe = f"{CFG.volume_mount_path}/.modal_health_probe"
    try:
        open(probe, "w").close()
        os.remove(probe)
        results["volume"] = f"OK — mounted at {CFG.volume_mount_path}"
    except Exception as e:
        results["volume"] = f"ERROR: {e}"

    r = subprocess.run(["s5cmd", "version"], capture_output=True, text=True)
    results["s5cmd"] = f"OK — {r.stdout.strip()}" if r.returncode == 0 else "MISSING"

    return results


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def verify() -> None:
    """Boot the container and verify all secrets, volume, and s5cmd."""
    print("Running container health check...")
    results = _health_check.remote()
    all_ok = True
    for k, v in results.items():
        symbol = "✓" if v.startswith("OK") else "✗"
        print(f"  {symbol}  {k}: {v}")
        if not v.startswith("OK"):
            all_ok = False
    print()
    if all_ok:
        print("All checks passed — Modal setup is ready.")
    else:
        raise SystemExit("One or more checks failed.")


@app.local_entrypoint()
def submit(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a Modal job and return immediately."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting commit {git_commit[:12]} from {git_remote}")
    handle = run_hydra_train.spawn(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Submitted Modal job: {handle.object_id}")
    print("Monitor at: https://modal.com/apps/egomimic-training")
    print(
        "After completion, download artifacts:\n"
        "  modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/"
    )


@app.local_entrypoint()
def submit_curate(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a Modal DemInf curation job and return immediately."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting curation commit {git_commit[:12]} from {git_remote}")
    handle = run_curate.spawn(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Submitted Modal curation job: {handle.object_id}")
    print("Monitor at: https://modal.com/apps/egomimic-training")
    print(
        "After completion, download artifacts:\n"
        "  modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/"
    )


@app.local_entrypoint()
def run_curate_cmd(*hydra_args: str) -> None:
    """Blocking run: streams curation logs to stdout and waits for completion."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Running curation at commit {git_commit[:12]} from {git_remote}")
    result = run_curate.remote(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Curation complete: {result}")


@app.local_entrypoint()
def run(*hydra_args: str) -> None:
    """Blocking run: streams logs and downloads artifacts to ./modal-outputs/ when complete."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Running commit {git_commit[:12]} from {git_remote}")
    output_rel_path = run_hydra_train.remote(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Remote run completed. Output path in volume: {output_rel_path}")
    if output_rel_path:
        _download_run_artifacts(output_rel_path)
