"""Rating engine: aggregation (mean ± 95% CI), bulk-rate logic, Krippendorff's α.

Every aggregate is recomputed from the ``ratings`` table on demand — means/CIs
are never stored as columns. The ``ratings`` log is the single source of truth.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

import numpy as np
from scipy import stats

try:  # krippendorff is optional at import time; /iota degrades gracefully.
    import krippendorff

    _HAVE_KRIPP = True
except Exception:  # pragma: no cover
    _HAVE_KRIPP = False


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------
def _span_filter(scene: Optional[str], operator: Optional[str]) -> tuple[str, list]:
    """Build a WHERE clause fragment + params for spans, on scene/operator."""
    clauses, params = [], []
    if scene:
        clauses.append("scene = ?")
        params.append(scene)
    if operator:
        clauses.append("operator = ?")
        params.append(operator)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def filtered_span_ids(
    conn: sqlite3.Connection, scene: Optional[str], operator: Optional[str]
) -> list[str]:
    where, params = _span_filter(scene, operator)
    rows = conn.execute(f"SELECT span_id FROM spans{where}", params).fetchall()
    return [r["span_id"] for r in rows]


# ---------------------------------------------------------------------------
# Rater helper
# ---------------------------------------------------------------------------
def ensure_rater(conn: sqlite3.Connection, rater_id: int, name: Optional[str] = None) -> None:
    """Create the rater row if absent (keeps the FK satisfied). No-op otherwise."""
    conn.execute(
        "INSERT INTO raters (rater_id, name) VALUES (?, ?) ON CONFLICT(rater_id) DO NOTHING",
        (rater_id, name),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Single-rating upsert (one rating per (span, rater), enforced here not in DB)
# ---------------------------------------------------------------------------
def upsert_rating(
    conn: sqlite3.Connection, span_id: str, rater_id: int, score: int, is_bulk: int
) -> None:
    """Insert or replace this rater's rating for this span.

    Re-rating via /rate replaces the score and resets is_bulk. The
    (span_id, rater_id) pair is kept unique at the API layer rather than with a
    DB constraint, so corrections are a plain UPDATE.
    """
    existing = conn.execute(
        "SELECT rating_id FROM ratings WHERE span_id = ? AND rater_id = ?",
        (span_id, rater_id),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE ratings SET score = ?, is_bulk = ?, ts = CURRENT_TIMESTAMP WHERE rating_id = ?",
            (score, is_bulk, existing["rating_id"]),
        )
    else:
        conn.execute(
            "INSERT INTO ratings (span_id, rater_id, score, is_bulk) VALUES (?, ?, ?, ?)",
            (span_id, rater_id, score, is_bulk),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Bulk-rate
# ---------------------------------------------------------------------------
def bulk_rate(
    conn: sqlite3.Connection,
    group_by: str,
    group_value: str,
    rater_id: int,
    score: int,
) -> dict[str, int]:
    """Assign an is_bulk=1 rating to every span in the group this rater has not
    rated *individually* (is_bulk=0). Existing individual ratings are skipped
    and never overwritten; a prior bulk rating is updated to the new score.
    """
    if group_by not in ("scene", "operator"):
        raise ValueError("group_by must be 'scene' or 'operator'")

    span_ids = [
        r["span_id"]
        for r in conn.execute(
            f"SELECT span_id FROM spans WHERE {group_by} = ?", (group_value,)
        ).fetchall()
    ]

    rated_count = 0
    skipped_count = 0
    for span_id in span_ids:
        existing = conn.execute(
            "SELECT rating_id, is_bulk FROM ratings WHERE span_id = ? AND rater_id = ?",
            (span_id, rater_id),
        ).fetchone()
        if existing and existing["is_bulk"] == 0:
            # An individual rating exists — never clobber it.
            skipped_count += 1
            continue
        # No rating yet, or a previous bulk rating: (re)assign as bulk.
        upsert_rating(conn, span_id, rater_id, score, is_bulk=1)
        rated_count += 1

    return {"rated_count": rated_count, "skipped_count": skipped_count}


# ---------------------------------------------------------------------------
# Aggregation: mean ± 95% CI per span
# ---------------------------------------------------------------------------
def aggregate(
    conn: sqlite3.Connection, scene: Optional[str], operator: Optional[str]
) -> list[dict[str, Any]]:
    """Per-span aggregate over the filtered span set, sorted by mean desc.

    Includes *every* span matching the filter, even unrated ones (n=0). For
    n < 2 the CI is null (it is undefined); for n = 0 the mean is null too.
    """
    where, params = _span_filter(scene, operator)
    spans = conn.execute(
        f"SELECT span_id, scene, operator FROM spans{where} ORDER BY span_id", params
    ).fetchall()

    # One pass over the relevant ratings, grouped by span.
    by_span: dict[str, list[sqlite3.Row]] = {s["span_id"]: [] for s in spans}
    if by_span:
        placeholders = ",".join("?" for _ in by_span)
        for r in conn.execute(
            f"SELECT span_id, score, is_bulk FROM ratings WHERE span_id IN ({placeholders})",
            list(by_span.keys()),
        ).fetchall():
            by_span[r["span_id"]].append(r)

    rows: list[dict[str, Any]] = []
    for s in spans:
        rlist = by_span[s["span_id"]]
        scores = [r["score"] for r in rlist]
        n = len(scores)
        mean = float(np.mean(scores)) if n >= 1 else None
        ci_low, ci_high = _confidence_interval(scores)
        n_bulk = sum(1 for r in rlist if r["is_bulk"] == 1)
        rows.append(
            {
                "span_id": s["span_id"],
                "scene": s["scene"],
                "operator": s["operator"],
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
                "bulk_fraction": (n_bulk / n) if n > 0 else None,
            }
        )

    # Sort by mean descending; spans with no mean (n=0) sink to the bottom.
    rows.sort(key=lambda r: (r["mean"] is not None, r["mean"] or 0.0), reverse=True)
    return rows


def _confidence_interval(scores: list[int]) -> tuple[Optional[float], Optional[float]]:
    """95% t-interval for the mean, clamped to the [1, 5] score range.

    Returns (None, None) when n < 2. The raw t-interval (mean ± t·sem) can
    extend past the 1–5 scale, but the population mean of values in [1, 5] is
    itself provably in [1, 5], so a bound outside that range is meaningless —
    we clamp it. Individual ratings are always 1–5 regardless.
    """
    n = len(scores)
    if n < 2:
        return None, None
    arr = np.asarray(scores, dtype=float)
    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))  # ddof=1
    if sem == 0.0:
        # All raters agree — interval collapses to the point estimate.
        return mean, mean
    low, high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return max(1.0, float(low)), min(5.0, float(high))


# ---------------------------------------------------------------------------
# Krippendorff's α (ordinal)
# ---------------------------------------------------------------------------
def krippendorff_alpha(
    conn: sqlite3.Connection, scene: Optional[str], operator: Optional[str]
) -> Optional[float]:
    """Ordinal Krippendorff's α across all raters over the filtered span set.

    Returns None when fewer than 2 raters have rated at least one common span
    (i.e. no unit is co-rated), or when the package is unavailable / α is
    mathematically undefined.
    """
    if not _HAVE_KRIPP:
        return None

    span_ids = filtered_span_ids(conn, scene, operator)
    if not span_ids:
        return None

    placeholders = ",".join("?" for _ in span_ids)
    ratings = conn.execute(
        f"SELECT span_id, rater_id, score FROM ratings WHERE span_id IN ({placeholders})",
        span_ids,
    ).fetchall()
    if not ratings:
        return None

    raters = sorted({r["rater_id"] for r in ratings})
    units = sorted({r["span_id"] for r in ratings})
    if len(raters) < 2:
        return None

    rater_idx = {rid: i for i, rid in enumerate(raters)}
    unit_idx = {sid: j for j, sid in enumerate(units)}

    matrix = np.full((len(raters), len(units)), np.nan)
    for r in ratings:
        matrix[rater_idx[r["rater_id"]], unit_idx[r["span_id"]]] = r["score"]

    # Require at least one unit co-rated by >= 2 raters (a "pairable" value).
    co_rated = np.sum(~np.isnan(matrix), axis=0) >= 2
    if not np.any(co_rated):
        return None

    try:
        alpha = krippendorff.alpha(
            reliability_data=matrix, level_of_measurement="ordinal"
        )
    except Exception:
        return None
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return None
    return float(alpha)
