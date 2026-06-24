"""FastAPI app for the ego-rating tool — all routes.

Run from the ego-rating/ directory:

    uvicorn backend.main:app --reload

Serves the SPA at ``/`` and (if present) video clips under ``/videos``.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import db, rating

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
VIDEOS_DIR = BASE_DIR / "videos"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_app()  # create tables, read config.yaml, upsert spans
    yield


app = FastAPI(title="ego-rating", lifespan=lifespan)

# Permissive CORS — this is a single-user dev tool; the SPA is same-origin but
# this keeps things working if the frontend is opened from elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class RateBody(BaseModel):
    span_id: str
    rater_id: int
    score: int = Field(ge=1, le=5)


class BulkRateBody(BaseModel):
    group_by: str = Field(pattern="^(scene|operator)$")
    group_value: str
    rater_id: int
    score: int = Field(ge=1, le=5)


# ---------------------------------------------------------------------------
# Config / metadata
# ---------------------------------------------------------------------------
@app.get("/config")
def get_config(conn: sqlite3.Connection = Depends(db.get_db)):
    scenes = [
        r["scene"]
        for r in conn.execute(
            "SELECT DISTINCT scene FROM spans ORDER BY scene"
        ).fetchall()
    ]
    operators = [
        r["operator"]
        for r in conn.execute(
            "SELECT DISTINCT operator FROM spans ORDER BY operator"
        ).fetchall()
    ]
    return {
        "annotation": db.get_annotation(),
        "scenes": scenes,
        "operators": operators,
    }


# ---------------------------------------------------------------------------
# Rating queue
# ---------------------------------------------------------------------------
@app.get("/next")
def get_next(
    rater_id: int,
    scene: Optional[str] = None,
    operator: Optional[str] = None,
    conn: sqlite3.Connection = Depends(db.get_db),
):
    """Next unrated span for this rater under the filter, randomly ordered.

    "Unrated" = no row in ratings for (span_id, rater_id). Returns null when
    none remain.
    """
    where, params = rating._span_filter(scene, operator)
    not_rated = (
        "span_id NOT IN (SELECT span_id FROM ratings WHERE rater_id = ?)"
    )
    if where:
        sql = f"SELECT * FROM spans{where} AND {not_rated}"
    else:
        sql = f"SELECT * FROM spans WHERE {not_rated}"
    sql += " ORDER BY RANDOM() LIMIT 1"
    row = conn.execute(sql, [*params, rater_id]).fetchone()
    if row is None:
        return None
    return {
        "span_id": row["span_id"],
        "video_uri": row["video_uri"],
        "start": row["start"],
        "end": row["end"],
        "scene": row["scene"],
        "operator": row["operator"],
    }


@app.post("/rate")
def post_rate(body: RateBody, conn: sqlite3.Connection = Depends(db.get_db)):
    # Span must exist (config is truth for spans).
    if conn.execute(
        "SELECT 1 FROM spans WHERE span_id = ?", (body.span_id,)
    ).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"unknown span_id: {body.span_id}")
    rating.ensure_rater(conn, body.rater_id)
    rating.upsert_rating(conn, body.span_id, body.rater_id, body.score, is_bulk=0)
    return {"ok": True}


@app.post("/bulk-rate")
def post_bulk_rate(body: BulkRateBody, conn: sqlite3.Connection = Depends(db.get_db)):
    rating.ensure_rater(conn, body.rater_id)
    try:
        result = rating.bulk_rate(
            conn, body.group_by, body.group_value, body.rater_id, body.score
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, **result}


# ---------------------------------------------------------------------------
# Aggregates / leaderboard
# ---------------------------------------------------------------------------
@app.get("/ratings")
def get_ratings(
    scene: Optional[str] = None,
    operator: Optional[str] = None,
    conn: sqlite3.Connection = Depends(db.get_db),
):
    return {"rows": rating.aggregate(conn, scene, operator)}


@app.get("/spans")
def get_spans(
    scene: Optional[str] = None,
    operator: Optional[str] = None,
    rater_id: Optional[int] = None,
    conn: sqlite3.Connection = Depends(db.get_db),
):
    """Spans matching the filter, each with a `rated` bool for the given rater.

    `rated` is True if the rater has any rating (individual or bulk) for the
    span — consistent with /next, which excludes those same spans.
    """
    where, params = rating._span_filter(scene, operator)
    spans = conn.execute(
        f"SELECT span_id, video_uri, start, end, scene, operator FROM spans{where} ORDER BY span_id",
        params,
    ).fetchall()

    rated_ids: set[str] = set()
    if rater_id is not None:
        rated_ids = {
            r["span_id"]
            for r in conn.execute(
                "SELECT DISTINCT span_id FROM ratings WHERE rater_id = ?", (rater_id,)
            ).fetchall()
        }

    return {
        "rows": [
            {
                "span_id": s["span_id"],
                "video_uri": s["video_uri"],
                "start": s["start"],
                "end": s["end"],
                "scene": s["scene"],
                "operator": s["operator"],
                "rated": s["span_id"] in rated_ids,
            }
            for s in spans
        ]
    }


@app.get("/iota")
def get_iota(
    scene: Optional[str] = None,
    operator: Optional[str] = None,
    conn: sqlite3.Connection = Depends(db.get_db),
):
    """Krippendorff's α (ordinal) over the filtered span set, or null."""
    return {"alpha": rating.krippendorff_alpha(conn, scene, operator)}


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@app.get("/raw-ratings")
def get_raw_ratings(conn: sqlite3.Connection = Depends(db.get_db)):
    """All rating rows (joined to span scene/operator) for the admin screen."""
    rows = conn.execute(
        """
        SELECT r.rating_id, r.span_id, r.rater_id, r.score, r.is_bulk, r.ts,
               s.scene, s.operator
        FROM ratings r
        LEFT JOIN spans s ON s.span_id = r.span_id
        ORDER BY r.ts DESC, r.rating_id DESC
        """
    ).fetchall()
    return {"rows": [dict(r) for r in rows]}


@app.post("/reload-config")
def post_reload_config(conn: sqlite3.Connection = Depends(db.get_db)):
    """Re-read config.yaml and re-upsert spans (admin; no auth this pass)."""
    try:
        n = db.load_and_upsert_spans(conn)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "span_count": n}


# ---------------------------------------------------------------------------
# Static SPA + videos (registered last so API routes take precedence)
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if VIDEOS_DIR.exists():
    app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")
