"""Train-set visualization evaluator.

Wraps a concrete EvalVideo (HPTEvalVideo, PIEvalVideo, …) so the same
forward/metric/viz logic can run a second time against a separate
``train_viz`` dataloader. Videos go to ``<root>/videos_train_viz/`` and
metric keys are prefixed with ``train_viz/`` so they don't collide with the
canonical ``Valid/...`` keys.

Instantiated via Hydra from a config like
``hydra_configs/evaluator/train_viz_hpt.yaml``.
"""

from __future__ import annotations

import os

from egomimic.eval.eval_video import EvalVideo


class TrainVizEvalVideo(EvalVideo):
    def __init__(
        self,
        base: EvalVideo,
        limit_val_batches: int = 400,
        metric_prefix: str | None = None,
        video_dirname: str = "videos_train_viz",
    ):
        """
        Args:
            base: concrete EvalVideo whose forward/metric/viz logic to reuse.
            metric_prefix: if None (default), legacy behavior — every metric key
                from ``base`` is prefixed with ``train_viz/``. If set (e.g.
                ``"Valid_oph"``), the canonical ``Valid/`` family emitted by
                ``base`` is REWRITTEN onto this prefix (``Valid/x`` →
                ``Valid_oph/x``), producing a fully separate val-metric family.
            video_dirname: output subdir under the run root for this pass's
                videos (default preserves the legacy ``videos_train_viz``;
                e.g. ``videos_oph`` for a held-out-operator val set).
        """
        self.base = base
        self.metric_prefix = metric_prefix
        self.video_dirname = video_dirname
        super().__init__(limit_val_batches=limit_val_batches)

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, value):
        self._trainer = value
        self.base.trainer = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.base.model = value

    def video_dir(self):
        return os.path.join(self.root_dir(), self.video_dirname)

    def compute_metrics_and_viz(self, batch):
        metrics, images_dict = self.base.compute_metrics_and_viz(batch)
        if self.metric_prefix is None:
            # Legacy: additive train_viz/ prefix (train_viz/Valid/...).
            metrics = {f"train_viz/{k}": v for k, v in metrics.items()}
        else:
            prefix = self.metric_prefix.rstrip("/") + "/"
            metrics = {
                (prefix + k[len("Valid/") :] if k.startswith("Valid/") else prefix + k): v
                for k, v in metrics.items()
            }
        return metrics, images_dict
