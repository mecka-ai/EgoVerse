"""Validation evaluator for the InfoNCE action-segment encoder.

Its presence re-enables ModelWrapper's Valid/ logging (validation_step
early-returns when evaluator is None), which surfaces the held-out InfoNCE loss
plus the contrastive telemetry (pos_sim / neg_sim / retrieval_acc / emb_std) on
val spans. No videos: embeddings have no image-space rendering; offline t-SNE /
NMI analysis runs on checkpoints instead.
"""

from egomimic.eval.eval import Eval


class EvalActionContrastive(Eval):
    def __init__(self):
        super().__init__()
        self.trainer = None
        self.model = None
        self.override_dict = {
            "limit_train_batches": 0,
            "limit_val_batches": 50,
            "check_val_every_n_epoch": 1,
            "max_epochs": 1,
            "min_epochs": 1,
            "num_sanity_val_steps": 0,
        }

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass
