"""CurationResult dataclass for DemInf curation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CurationResult:
    """Output of a single curation run."""

    kept_hashes: list[str]
    removed_hashes: list[str]  # filtered out by preprocessing
    low_mi_hashes: list[str]  # discarded for low MI score
    scores: dict[str, float]  # {episode_hash: mi_score} for scored episodes
    stats: dict = field(default_factory=dict)

    @property
    def all_removed_hashes(self) -> list[str]:
        return self.removed_hashes + self.low_mi_hashes
