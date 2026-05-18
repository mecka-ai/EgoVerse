"""
Modal entrypoint for DemInf curation at scale.

The curation Modal function (run_curate) lives in run.py alongside the
training function so that both share the same self-contained image definition.

Usage
-----
Via trainHydra.py (recommended — auto-submits when trainer._modal=true):
    python egomimic/trainHydra.py --config-name=curate \\
        name=my_run description=test trainer._modal=true

Direct invocation (fire-and-forget):
    modal run --env robotics egomimic/modal/run.py::submit_curate -- \\
        name=my_run description=test model.filter_ratio=0.2

Or via this convenience entrypoint:
    modal run --env robotics egomimic/modal/curate_modal.py -- \\
        name=my_run description=test model.filter_ratio=0.2

Resources
---------
KSG mutual-information estimation is CPU-bound (scipy cKDTree, workers=-1).
Defaults: no GPU, 32 CPUs.  For StateEmbedder mode=image, override:
    +modal_gpu=A100 +modal_cpu=16
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export run_curate from run.py so `modal run curate_modal.py` works.
# The @app.local_entrypoint below is a convenience alias for submit_curate.
# ---------------------------------------------------------------------------

from egomimic.modal.run import (  # noqa: F401
    _local_wandb_key,
    _resolve_git_state,
    app,
    run_curate,
)


@app.local_entrypoint()
def main(*hydra_args: str, wandb_api_key: str = "") -> None:
    """Submit a DemInf curation run to Modal (fire-and-forget).

    All positional arguments after ``--`` are forwarded as Hydra overrides:
        modal run egomimic/modal/curate_modal.py -- \\
            name=my_run description=test model.filter_ratio=0.2
    """
    if not wandb_api_key:
        wandb_api_key = _local_wandb_key()

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )

    print(f"Submitting curation commit {git_commit[:12]} from {git_remote}")
    handle = run_curate.spawn(
        tuple(hydra_args), git_remote, git_commit, wandb_api_key
    )
    print(f"Submitted Modal curation job: {handle.object_id}")
    print("Monitor at: https://modal.com/apps/egomimic-training")
    print(
        "After completion, download artifacts:\n"
        "  modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/"
    )
