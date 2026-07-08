"""Naive rule-based language clustering: exact (verb, hand, direction) triples.

Instead of Qwen3 embeddings + k-means, parse each annotation for the three motion
factors and put every span with an identical parsed triple in the same cluster —
within a cluster all three words align perfectly by construction. Vocabulary from
the elmo GT annotation analysis (pick up/put/dump verbs; from-the-top / side-grip /
diagonal / lip-grip approach phrases; facing-* placement orientations).
"""

from __future__ import annotations

import re
from collections import Counter

_VERBS = [
    ("pick_up", r"^\s*(pick\s+up|grasp|lift|take)\b"),
    ("put", r"^\s*(put|place|set|return)\b"),
    ("dump", r"^\s*(dump|pour)\b"),
]

# Priority-ordered: the first matching phrase wins (texts often carry several,
# e.g. "diagonal lip grasp at the bottom" → diagonal).
_DIRECTIONS = [
    ("top", r"from the top|top gras|from above"),
    ("side", r"from the side|side gri"),
    ("diagonal", r"diagonal"),
    ("front", r"from the front"),
    ("handle", r"by the handle|handle gri"),
    ("lip", r"lip gri|lip gras"),
    ("bottom", r"at the bottom"),
    ("facing_forward", r"facing (the )?(forward|front)"),
    ("facing_backward", r"facing (the )?backward"),
    ("facing_left", r"facing (the )?left"),
    ("facing_right", r"facing (the )?right"),
    ("facing_top", r"facing (the )?top"),
    ("facing_bottom", r"facing (the )?bottom"),
    ("right_side_up", r"right side up"),
    ("upside_down", r"upside down"),
]


def parse_motion_triple(text: str) -> tuple[str, str, str]:
    """Parse (verb, hand, direction) from an annotation. Unmatched factors → 'other'/'none'."""
    tl = str(text).lower()
    verb = next((v for v, p in _VERBS if re.search(p, tl)), "other")
    left, right, both = "left hand" in tl, "right hand" in tl, "both hands" in tl
    hand = "both" if (both or (left and right)) else "left" if left else "right" if right else "none"
    direction = next((d for d, p in _DIRECTIONS if re.search(p, tl)), "none")
    return verb, hand, direction


def naive_language_clusters(span_ids: list[str], span_texts: list[str],
                            span_meta: list[dict]) -> dict:
    """Group spans by exact parsed triple → clustered-scores dict (viewer schema).

    Cluster ids are assigned by descending size; labels are the triple itself,
    e.g. ``pick_up | left | diagonal``.
    """
    triples = [parse_motion_triple(t) for t in span_texts]
    order = [t for t, _ in Counter(triples).most_common()]
    cid_of = {t: i for i, t in enumerate(order)}

    clustered: dict = {
        f"cluster_{i}": {"label": " | ".join(t), "spans": {}} for t, i in cid_of.items()
    }
    for sid, text, m, t in zip(span_ids, span_texts, span_meta, triples):
        clustered[f"cluster_{cid_of[t]}"]["spans"][sid] = {
            "score": None, "episode": m["episode"],
            "start": int(m["start"]), "end": int(m["end"]), "text": text,
        }
    return clustered
