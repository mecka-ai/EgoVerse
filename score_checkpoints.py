"""Score the arm checkpoints of a task family on that family's shared eval set.

The nine runs each validate on their OWN arm's 10% split, so their Valid/* curves
cannot be compared with each other -- a bottom-arm model can post a lower loss
simply because its data is easier to fit. This scores checkpoints on one set per
task family that no arm in that family trained on (built by build_common_eval.py),
which is what actually answers whether the quality score predicts model quality.

Structure mirrors trainModal deliberately: clone the repo, then run the work in a
SUBPROCESS. _prepare_repo installs openpi's patched transformers 4.53.2 and
overlays its modeling files, and those only take effect in a fresh interpreter --
scoring in-process instead builds the model from the old modeling code and the
checkpoint fails to load with mismatched layer names like
`paligemma_with_expert.gemma_expert...input_layernorm.dense.weight`.

    modal run -e robotics score_checkpoints.py --domain dish-handling
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

# modal_setup.py is baked into the image at /root/, so a bare-name import works
# both locally and in the container -- where `egomimic` does not exist until the
# repo is cloned. Importing egomimic.modal.modal_setup fails at module load.
_HERE = str(Path(__file__).resolve().parent / "egomimic" / "modal")
for _p in (_HERE, "/root"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_setup import (  # noqa: E402
    CFG,
    _prepare_repo,
    _resolve_git_state,
    image as train_image,
)

ZARR_MOUNT = "/mnt/zarr-data"
OUT_MOUNT = "/root/EgoVerse/logs"

app = modal.App("score-checkpoints")
zarr_vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)
out_vol = modal.Volume.from_name("egoverse-training-outputs", create_if_missing=False)

N_EVAL_BATCHES = int(os.environ.get("SCORE_N_BATCHES", "60"))


@app.function(
    image=train_image,
    gpu="H200:1",
    cpu=16,
    memory=131072,
    timeout=14400,
    ephemeral_disk=600 * 1024,
    secrets=[modal.Secret.from_name(n) for n in CFG.secret_names],
    volumes={ZARR_MOUNT: zarr_vol, OUT_MOUNT: out_vol},
)
def score(domain: str, run_globs: list[str], git_remote: str, git_commit: str,
          n_batches: int) -> dict:
    _prepare_repo(git_remote=git_remote, git_commit=git_commit,
                  init_submodules=True)
    out_vol.reload()

    ckpts: list[str] = []
    for pattern in run_globs:
        ckpts.extend(sorted(str(p) for p in Path(OUT_MOUNT).glob(pattern)))
    if not ckpts:
        print(f"no checkpoints matched {run_globs}")
        return {}
    print(f"{len(ckpts)} checkpoint(s) to score for {domain}")

    manifest = f"{ZARR_MOUNT}/_arms/_eval_common__{domain}.json"
    worker = f"{CFG.remote_repo_dir}/egomimic/scripts/score_common_eval.py"
    out_json = "/tmp/score_results.json"

    cmd = [CFG.python_bin, worker,
           "--eval-manifest", manifest,
           "--zarr-dir", ZARR_MOUNT,
           "--n-batches", str(n_batches),
           "--out", out_json]
    for c in ckpts:
        cmd += ["--ckpt", c]

    env = os.environ.copy()
    openpi_src = f"{CFG.remote_repo_dir}/external/openpi/src"
    env["PYTHONPATH"] = (f"{openpi_src}:{env['PYTHONPATH']}"
                         if env.get("PYTHONPATH") else openpi_src)

    proc = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"scoring worker failed (exit {proc.returncode})")

    results = json.loads(Path(out_json).read_text())
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main(domain: str = "cleaning-sanitation", glob: str = "",
         n_batches: int = N_EVAL_BATCHES):
    globs = ([glob] if glob else
             [f"qaexp-{domain}-{arm}/*/checkpoints/*.ckpt"
              for arm in ("top", "bottom", "random")])
    git_remote, git_commit, _ = _resolve_git_state()
    print(json.dumps(
        score.remote(domain, globs, git_remote, git_commit, n_batches), indent=2))
