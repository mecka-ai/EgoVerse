"""Index bounds-checking mixin shared by the prefetch datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

class _BoundsCheckMixin:
    """Shared bounds-check logic for map-style and iterable datasets."""

    def _check_bounds(
        self, data: dict, dataset, idx: int, dataset_name: str
    ) -> str | None:
        if self.data_schematic is None:
            return None

        embodiment_id = data.get("embodiment")
        if embodiment_id is None:
            raise ValueError("data has no embodiment metadata")

        norm_stats = self.data_schematic.norm_stats.get(embodiment_id, {})
        if not norm_stats:
            return None

        self._n_samples_checked += 1
        prefix: str | None = None

        for key_name, stats in norm_stats.items():
            zarr_key = self.data_schematic.keyname_to_zarr_key(key_name, embodiment_id)
            if zarr_key is None or zarr_key not in data:
                continue

            v = data[zarr_key]
            if isinstance(v, torch.Tensor):
                arr = v.float()
            elif isinstance(v, np.ndarray):
                arr = torch.from_numpy(v).float()
            else:
                continue

            slack = getattr(self, "bounds_slack", 0.0)
            q_low = torch.as_tensor(
                stats.get("quantile_0_01", stats.get("quantile_0_1", stats["quantile_1"])),
                dtype=torch.float32,
            ) - slack
            q_high = torch.as_tensor(
                stats.get("quantile_99_99", stats.get("quantile_99_9", stats["quantile_99"])),
                dtype=torch.float32,
            ) + slack

            try:
                q_low = torch.broadcast_to(q_low, arr.shape)
                q_high = torch.broadcast_to(q_high, arr.shape)
            except RuntimeError:
                key_sig = (str(zarr_key), tuple(arr.shape), tuple(q_low.shape))
                if not hasattr(self, "_shape_mismatch_warned"):
                    self._shape_mismatch_warned = set()
                if key_sig not in self._shape_mismatch_warned:
                    self._shape_mismatch_warned.add(key_sig)
                    logger.warning(
                        "Skipping bounds check for key=%s: value=%s q_low=%s",
                        zarr_key, tuple(arr.shape), tuple(q_low.shape),
                    )
                continue

            if torch.any(torch.isnan(arr)) or torch.any(torch.isinf(arr)):
                ep_name = Path(getattr(dataset, "episode_path", dataset_name)).name
                prefix = f"NaN/Inf violation ep={ep_name} frame={idx} key={zarr_key}"
                break

            if torch.any(arr < q_low) or torch.any(arr > q_high):
                ep_name = Path(getattr(dataset, "episode_path", dataset_name)).name
                prefix = f"Bounds violation ep={ep_name} frame={idx} key={zarr_key}"
                break

        if prefix is not None:
            self._n_violation_samples += 1

        return prefix


