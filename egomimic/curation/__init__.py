"""DemInf data curation (mutual-information episode scoring)."""

from egomimic.curation.config import (
    CurationLoaderSettings,
    EmbedderSettings,
    StateImageSettings,
    apply_curation_seed,
    load_action_norm_stats,
    select_curation_loader,
    select_embedder_settings,
    select_seed,
    select_state_image_settings,
    select_tensor_keys,
)
from egomimic.curation.embedders import ActionEmbedder, StateEmbedder
from egomimic.curation.episode_pipeline import (
    build_embedders,
    run_pass2_embed_episodes,
)
from egomimic.curation.ksg import ksg_mi, ksg_mi_averaged
from egomimic.curation.scoring import (
    TrajectoryScorer,
    aggregate_scores,
    trajectory_scorer_from_cfg,
)

__all__ = [
    "ActionEmbedder",
    "CurationLoaderSettings",
    "EmbedderSettings",
    "StateImageSettings",
    "apply_curation_seed",
    "StateEmbedder",
    "TrajectoryScorer",
    "aggregate_scores",
    "build_embedders",
    "ksg_mi",
    "ksg_mi_averaged",
    "load_action_norm_stats",
    "run_pass2_embed_episodes",
    "select_curation_loader",
    "select_embedder_settings",
    "select_seed",
    "select_state_image_settings",
    "select_tensor_keys",
    "trajectory_scorer_from_cfg",
]
