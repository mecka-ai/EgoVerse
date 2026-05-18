"""
DemInf curation algorithm.

Integrates DemInf (Hejna et al. 2025) into the EgoVerse Hydra+Lightning
training pipeline.  Configured via hydra_configs/model/deminf_*.yaml and
launched through trainHydra.py with ``mode: curate``:

    python egomimic/trainHydra.py --config-name=curate \\
        name=my_run description=test data=mecka_all_zarr model=deminf_default

All sub-components (embedders, filters) are Hydra-instantiated from the model
config automatically.  The TrajectoryScorer is assembled from the pre-built
embedders and the ``scorer`` config dict (which has no _target_).

Results are written to the Hydra output directory and a DatasetFilter-compatible
filter.yaml is exported for direct injection into subsequent training runs:

    +data.train_datasets.mecka_bimanual.filters.filter_lambdas=[<lambda>]
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from egomimic.curation.curator import CurationResult
from egomimic.curation.embedders import ActionEmbedder, StateEmbedder
from egomimic.curation.filters import apply_filters
from egomimic.curation.scorer import TrajectoryScorer
from egomimic.curation.utils import Episode

logger = logging.getLogger(__name__)


class DemInfAlgo:
    """
    DemInf curation algorithm, driven by Hydra config.

    Not a LightningModule — curation does not require gradient updates.
    Launched via ``mode: curate`` in trainHydra.py.

    Args:
        state_embedder: Pre-instantiated StateEmbedder (Hydra creates this).
        action_embedder: Pre-instantiated ActionEmbedder (Hydra creates this).
        scorer: DictConfig with ``k_range`` list — built into TrajectoryScorer
            internally so embedder instances can be wired in.
        filter_ratio: Bottom fraction of MI-scored episodes to discard.
        preprocessing: List of pre-instantiated EpisodeFilter objects (PauseFilter,
            ActionClipFilter, MinLengthFilter) — Hydra instantiates each entry.
        cross_embodiment_mode: "independent" scores each embodiment separately
            (recommended for heterogeneous datasets); "shared" scores jointly.
        device: Torch device string forwarded to image-mode embedders.
    """

    def __init__(
        self,
        state_embedder: StateEmbedder,
        action_embedder: ActionEmbedder,
        scorer: Any,
        filter_ratio: float = 0.3,
        preprocessing: list | None = None,
        cross_embodiment_mode: str = "independent",
        grouping: str = "task",
        device: str = "cpu",
    ) -> None:
        self.state_embedder = state_embedder
        self.action_embedder = action_embedder
        self.filter_ratio = filter_ratio
        self.preprocessing = list(preprocessing) if preprocessing else []
        self.cross_embodiment_mode = cross_embodiment_mode
        self.grouping = grouping
        self.device = device

        k_range = tuple(int(k) for k in scorer.get("k_range", [3, 7]))
        self._scorer = TrajectoryScorer(
            state_embedder=state_embedder,
            action_embedder=action_embedder,
            k_range=k_range,
            cross_embodiment_mode=cross_embodiment_mode,
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def curate(
        self,
        episodes: list[Episode],
        output_dir: Path | str | None = None,
        wandb_run: Any = None,
    ) -> CurationResult:
        """
        Run the full DemInf curation pipeline.

        Steps:
            1. Apply preprocessing filters (pauses, clipping, min-length).
            2. Fit embedders on all retained episodes.
            3. KSG-score every episode by its mean per-timestep MI.
            4. Keep the top ``(1 - filter_ratio)`` by score.
            5. Log metrics to WandB and write outputs to ``output_dir``.

        Args:
            episodes: Raw episodes loaded by trainHydra.py from the data config.
            output_dir: Hydra runtime output directory for all artifacts.
            wandb_run: Active ``wandb.Run`` for metric logging (optional; falls
                back to ``wandb.run`` if None).

        Returns:
            CurationResult with kept/removed hashes, per-episode MI scores,
            and aggregate statistics.
        """
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("DemInf curation: %d input episodes (grouping=%s)", len(episodes), self.grouping)

        # Step 1: preprocessing
        filtered_episodes, pre_removed = apply_filters(episodes, self.preprocessing)
        logger.info(
            "Preprocessing: %d / %d episodes retained",
            len(filtered_episodes),
            len(episodes),
        )

        if not filtered_episodes:
            logger.warning("All episodes removed by preprocessing filters!")
            result = CurationResult(
                kept_hashes=[],
                removed_hashes=pre_removed,
                low_mi_hashes=[],
                scores={},
                stats={"error": "all_episodes_filtered"},
            )
            if output_dir:
                self._save(result, output_dir, scores_by_task={})
            return result

        # Step 2 + 3: embed + score (per-task or global)
        if self.grouping == "task":
            scores, scores_by_task = self._score_per_task(filtered_episodes)
        else:
            self._scorer.fit(filtered_episodes)
            scores = self._scorer.get_scores()
            scores_by_task = {}

        # No filtering — keep all scored episodes; filtering is a separate step.
        kept_hashes = list(scores.keys())
        all_score_vals = np.array([s for s in scores.values() if np.isfinite(s)])

        # Per-task stats
        per_task_stats: dict[str, dict] = {}
        for task_name, task_scores in scores_by_task.items():
            vals = np.array([s for s in task_scores.values() if np.isfinite(s)])
            per_task_stats[task_name] = {
                "count": len(task_scores),
                "mi_mean": float(np.nanmean(vals)) if len(vals) else float("nan"),
                "mi_std": float(np.nanstd(vals)) if len(vals) else float("nan"),
                "mi_median": float(np.nanmedian(vals)) if len(vals) else float("nan"),
                "mi_min": float(np.nanmin(vals)) if len(vals) else float("nan"),
                "mi_max": float(np.nanmax(vals)) if len(vals) else float("nan"),
            }

        stats = {
            "total_input": len(episodes),
            "pre_filter_removed": len(pre_removed),
            "scored": len(scores),
            "n_tasks": len(scores_by_task),
            "grouping": self.grouping,
            "cross_embodiment_mode": self.cross_embodiment_mode,
            "mi_mean": float(all_score_vals.mean()) if len(all_score_vals) else float("nan"),
            "mi_std": float(all_score_vals.std()) if len(all_score_vals) else float("nan"),
            "mi_median": float(np.median(all_score_vals)) if len(all_score_vals) else float("nan"),
            "mi_min": float(all_score_vals.min()) if len(all_score_vals) else float("nan"),
            "mi_max": float(all_score_vals.max()) if len(all_score_vals) else float("nan"),
            "per_task": per_task_stats,
        }

        result = CurationResult(
            kept_hashes=kept_hashes,
            removed_hashes=pre_removed,
            low_mi_hashes=[],
            scores=scores,
            stats=stats,
        )

        logger.info(
            "Scoring done — %d episodes across %d tasks  pre_filter_removed=%d",
            len(scores),
            len(scores_by_task),
            len(pre_removed),
        )

        # Step 5: log + save
        self._log_wandb(stats, per_task_stats, wandb_run)
        if output_dir:
            self._save(result, output_dir, scores_by_task=scores_by_task)

        return result

    # ------------------------------------------------------------------
    # Per-task scoring
    # ------------------------------------------------------------------

    def _score_per_task(
        self, episodes: list[Episode]
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Score each episode relative to others in the same task group."""
        from collections import defaultdict

        by_task: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            task = ep.metadata.get("task_name", "unknown")
            by_task[task].append(ep)

        logger.info("Scoring %d tasks independently …", len(by_task))

        flat_scores: dict[str, float] = {}
        scores_by_task: dict[str, dict[str, float]] = {}

        for task_name, group in sorted(by_task.items()):
            if len(group) < 2:
                # KSG needs at least 2 points; assign nan
                task_scores = {ep.episode_hash: float("nan") for ep in group}
                logger.warning("Task '%s' has only %d episode(s) — skipping KSG", task_name, len(group))
            else:
                self._scorer.fit(group)
                task_scores = self._scorer.get_scores()
                logger.info("Task '%s': scored %d episodes", task_name, len(task_scores))

            flat_scores.update(task_scores)
            scores_by_task[task_name] = task_scores

        return flat_scores, scores_by_task

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save(
        self,
        result: CurationResult,
        output_dir: Path,
        scores_by_task: dict | None = None,
    ) -> None:
        """Write scores.json, scores_by_task.json, kept_hashes.json, curation_stats.json."""
        with open(output_dir / "scores.json", "w") as f:
            json.dump(result.scores, f, indent=2)

        if scores_by_task:
            with open(output_dir / "scores_by_task.json", "w") as f:
                json.dump(scores_by_task, f, indent=2)

        with open(output_dir / "kept_hashes.json", "w") as f:
            json.dump(result.kept_hashes, f, indent=2)

        with open(output_dir / "curation_stats.json", "w") as f:
            json.dump(result.stats, f, indent=2)

        logger.info("Curation outputs written to %s", output_dir)

    def _export_filter_yaml(self, result: CurationResult, path: Path) -> None:
        """
        Write a DatasetFilter-compatible YAML for direct use in training configs.

        Inject the filter into a training run:
            +data.train_datasets.<name>.filters.filter_lambdas=[<lambda>]
        """
        kept = result.kept_hashes
        stats = result.stats or {}
        mi_mean = stats.get("mi_mean")
        mi_median = stats.get("mi_median")

        kept_set_repr = repr(set(kept))
        lambda_str = f"lambda row: row.get('episode_hash') in {kept_set_repr}"
        mi_line = (
            f"# MI mean/median: {mi_mean:.4f} / {mi_median:.4f}"
            if mi_mean is not None and mi_median is not None and np.isfinite(mi_mean)
            else "# MI stats: unavailable"
        )
        lines = [
            "# DemInf curation filter",
            f"# Generated: {datetime.now(tz=timezone.utc).isoformat()}",
            f"# Git hash:  {_get_git_hash()}",
            f"# Episodes:  {len(kept)} kept / {stats.get('total_input', 0)} total",
            f"# Filter ratio: {self.filter_ratio:.0%}",
            mi_line,
            "#",
            "# Usage — add to a training data config's filters section:",
            "#   +data.train_datasets.<name>.filters.filter_lambdas=[<lambda>]",
            "",
            "_target_: egomimic.rldb.filters.DatasetFilter",
            "filter_lambdas:",
            f"  - {json.dumps(lambda_str)}",
            "",
            "kept_hashes:",
        ]
        for h in kept:
            lines.append(f"  - {json.dumps(h)}")
        lines.append("")
        path.write_text("\n".join(lines))

    # ------------------------------------------------------------------
    # WandB logging
    # ------------------------------------------------------------------

    def _log_wandb(
        self,
        stats: dict,
        per_task_stats: dict,
        wandb_run: Any = None,
    ) -> None:
        """Log curation metrics to WandB if a run is active."""
        try:
            import wandb as _wandb

            run = wandb_run or _wandb.run
            if run is None:
                return

            metrics: dict[str, float] = {
                "curation/total_input": stats.get("total_input", 0),
                "curation/scored": stats.get("scored", 0),
                "curation/pre_filter_removed": stats.get("pre_filter_removed", 0),
                "curation/n_tasks": stats.get("n_tasks", 0),
                "curation/mi_mean": stats.get("mi_mean", float("nan")),
                "curation/mi_std": stats.get("mi_std", float("nan")),
                "curation/mi_median": stats.get("mi_median", float("nan")),
            }

            for task_name, ts in per_task_stats.items():
                safe = task_name.replace(".", "_").replace("/", "_")[:64]
                metrics[f"curation/task/{safe}/count"] = ts.get("count", 0)
                metrics[f"curation/task/{safe}/mi_mean"] = ts.get("mi_mean", float("nan"))
                metrics[f"curation/task/{safe}/mi_median"] = ts.get("mi_median", float("nan"))

            run.log(metrics)
            logger.info("Logged curation metrics to WandB run '%s'", run.name)

        except ImportError:
            logger.debug("wandb not installed — skipping WandB logging")
        except Exception as exc:
            logger.warning("WandB logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
