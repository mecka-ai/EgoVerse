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
        device: str = "cpu",
    ) -> None:
        self.state_embedder = state_embedder
        self.action_embedder = action_embedder
        self.filter_ratio = filter_ratio
        self.preprocessing = list(preprocessing) if preprocessing else []
        self.cross_embodiment_mode = cross_embodiment_mode
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

        logger.info("DemInf curation: %d input episodes", len(episodes))

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
                self._save(result, output_dir)
            return result

        # Step 2 + 3: embed + score
        self._scorer.fit(filtered_episodes)
        scores = self._scorer.get_scores()

        # Step 4: rank + threshold
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        n_total = len(ranked)
        n_remove = int(np.floor(n_total * self.filter_ratio))
        n_keep = n_total - n_remove

        kept_hashes = [h for h, _ in ranked[:n_keep]]
        low_mi_hashes = [h for h, _ in ranked[n_keep:]]
        threshold_score = ranked[n_keep - 1][1] if n_keep > 0 else float("nan")

        all_scores = np.array([s for _, s in ranked if np.isfinite(s)])
        ep_by_hash = {ep.episode_hash: ep for ep in filtered_episodes}
        kept_set = set(kept_hashes)

        embodiment_scores: dict[str, list[float]] = {}
        embodiment_kept: dict[str, int] = {}
        embodiment_total: dict[str, int] = {}
        for h, s in scores.items():
            emb = ep_by_hash[h].embodiment if h in ep_by_hash else "unknown"
            embodiment_scores.setdefault(emb, []).append(s)
            embodiment_total[emb] = embodiment_total.get(emb, 0) + 1
            if h in kept_set:
                embodiment_kept[emb] = embodiment_kept.get(emb, 0) + 1

        per_embodiment_stats = {
            emb: {
                "count": len(vals),
                "kept": embodiment_kept.get(emb, 0),
                "total": embodiment_total.get(emb, 0),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "median": float(np.nanmedian(vals)),
            }
            for emb, vals in embodiment_scores.items()
        }

        stats = {
            "total_input": len(episodes),
            "pre_filter_removed": len(pre_removed),
            "scored": n_total,
            "kept": n_keep,
            "low_mi_removed": n_remove,
            "filter_ratio": self.filter_ratio,
            "threshold_score": float(threshold_score),
            "mi_mean": float(all_scores.mean()) if len(all_scores) else float("nan"),
            "mi_std": float(all_scores.std()) if len(all_scores) else float("nan"),
            "mi_median": float(np.median(all_scores)) if len(all_scores) else float("nan"),
            "mi_min": float(all_scores.min()) if len(all_scores) else float("nan"),
            "mi_max": float(all_scores.max()) if len(all_scores) else float("nan"),
            "per_embodiment": per_embodiment_stats,
            "cross_embodiment_mode": self.cross_embodiment_mode,
        }

        result = CurationResult(
            kept_hashes=kept_hashes,
            removed_hashes=pre_removed,
            low_mi_hashes=low_mi_hashes,
            scores=scores,
            stats=stats,
        )

        logger.info(
            "Curation done — kept=%d  pre_filter_removed=%d  low_mi_removed=%d",
            n_keep,
            len(pre_removed),
            n_remove,
        )

        # Step 5: log + save
        self._log_wandb(stats, per_embodiment_stats, wandb_run)
        if output_dir:
            self._save(result, output_dir)

        return result

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save(self, result: CurationResult, output_dir: Path) -> None:
        """Write scores.json, kept_hashes.json, curation_stats.json, filter.yaml."""
        with open(output_dir / "scores.json", "w") as f:
            json.dump(result.scores, f, indent=2)

        with open(output_dir / "kept_hashes.json", "w") as f:
            json.dump(result.kept_hashes, f, indent=2)

        with open(output_dir / "curation_stats.json", "w") as f:
            json.dump(result.stats, f, indent=2)

        self._export_filter_yaml(result, output_dir / "filter.yaml")
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
        per_embodiment_stats: dict,
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
                "curation/kept": stats.get("kept", 0),
                "curation/pre_filter_removed": stats.get("pre_filter_removed", 0),
                "curation/low_mi_removed": stats.get("low_mi_removed", 0),
                "curation/keep_rate": stats.get("kept", 0)
                / max(stats.get("scored", 1), 1),
                "curation/mi_mean": stats.get("mi_mean", float("nan")),
                "curation/mi_std": stats.get("mi_std", float("nan")),
                "curation/mi_median": stats.get("mi_median", float("nan")),
                "curation/threshold_score": stats.get("threshold_score", float("nan")),
            }

            for emb, emb_stats in per_embodiment_stats.items():
                safe = emb.replace(".", "_")
                metrics[f"curation/{safe}/kept"] = emb_stats.get("kept", 0)
                metrics[f"curation/{safe}/total"] = emb_stats.get("total", 0)
                metrics[f"curation/{safe}/mi_mean"] = emb_stats.get(
                    "mean", float("nan")
                )
                metrics[f"curation/{safe}/mi_median"] = emb_stats.get(
                    "median", float("nan")
                )

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
