"""SQLite init + config-driven span upsert.

Startup behavior (see ``init_app``):
  1. Create tables if they don't exist.
  2. Read ``config.yaml``.
  3. Upsert every span from config into the ``spans`` table (insert or replace
     on ``span_id``).
  4. Expose ``config["annotation"]`` to all routes (via :func:`get_annotation`).

The schema is intentionally plain (no DB-level UNIQUE on ratings) so it ports
cleanly to Postgres later; the "one rating per (span, rater)" rule is enforced
at the API layer instead (see ``rating.upsert_rating``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # the ego-rating/ root
CONFIG_PATH = BASE_DIR / "config.yaml"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "ego_rating.db"

# ---------------------------------------------------------------------------
# Schema (kept verbatim from the spec; types chosen to map cleanly to Postgres)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
  span_id   TEXT PRIMARY KEY,
  video_uri TEXT NOT NULL,
  start     REAL NOT NULL,
  end       REAL NOT NULL,
  scene     TEXT NOT NULL,
  operator  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raters (
  rater_id INTEGER PRIMARY KEY,
  name     TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
  rating_id INTEGER PRIMARY KEY,
  span_id   TEXT    REFERENCES spans(span_id),
  rater_id  INTEGER REFERENCES raters(rater_id),
  score     INTEGER CHECK(score BETWEEN 1 AND 5),
  is_bulk   INTEGER DEFAULT 0,
  ts        DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Speeds up the (span_id, rater_id) existence checks used everywhere.
CREATE INDEX IF NOT EXISTS idx_ratings_span_rater ON ratings(span_id, rater_id);
"""

# Module-level annotation, refreshed by ``load_and_upsert_spans``.
_ANNOTATION: str = ""


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------
def connect() -> sqlite3.Connection:
    """Open a connection with row access by name and FK enforcement on."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db() -> Iterable[sqlite3.Connection]:
    """FastAPI dependency: yields a connection and always closes it."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def read_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
    if "annotation" not in config:
        raise ValueError("config.yaml must define a top-level 'annotation' string")
    spans = config.get("spans") or []
    seen: set[str] = set()
    for span in spans:
        sid = span.get("id")
        if not sid:
            raise ValueError(f"every span needs an 'id': {span!r}")
        if sid in seen:
            raise ValueError(f"duplicate span id in config: {sid!r}")
        seen.add(sid)
    return config


def upsert_spans(conn: sqlite3.Connection, spans: list[dict[str, Any]]) -> int:
    """Insert-or-replace each config span keyed on span_id. Returns count."""
    rows = [
        (
            s["id"],
            s["video"],
            float(s["start"]),
            float(s["end"]),
            s["scene"],
            s["operator"],
        )
        for s in spans
    ]
    conn.executemany(
        """
        INSERT INTO spans (span_id, video_uri, start, end, scene, operator)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(span_id) DO UPDATE SET
          video_uri = excluded.video_uri,
          start     = excluded.start,
          end       = excluded.end,
          scene     = excluded.scene,
          operator  = excluded.operator
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_and_upsert_spans(conn: sqlite3.Connection) -> int:
    """Read config.yaml, refresh the module annotation, upsert spans."""
    global _ANNOTATION
    config = read_config()
    _ANNOTATION = str(config["annotation"])
    return upsert_spans(conn, config.get("spans") or [])


def get_annotation() -> str:
    """The single shared instruction string for the session."""
    return _ANNOTATION


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def init_app() -> None:
    """Create tables and load config — call once at process startup."""
    conn = connect()
    try:
        init_db(conn)
        n = load_and_upsert_spans(conn)
        print(f"[ego-rating] initialized DB at {DB_PATH}; upserted {n} spans.")
    finally:
        conn.close()
