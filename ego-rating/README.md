# ego-rating

A 1–5 direct-rating web app for scoring action spans by quality. A rater watches
a single video clip of an action span, sees the shared annotation above it, and
submits a 1–5 score. Scores aggregate into a per-span quality leaderboard
(mean ± 95% CI), with inter-rater reliability via Krippendorff's α. Supports
filtering the queue by **scene** or **operator**, and bulk-assigning a score to
every unrated span in a scene/operator group.

Spans are defined entirely in `config.yaml` — they are never created or edited
through the UI.

## Quick start

```bash
cd ego-rating
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
# open http://127.0.0.1:8000
```

On startup the backend creates `data/ego_rating.db`, reads `config.yaml`, and
upserts every span.

Drop video clips into `videos/` (paths in `config.yaml` are served from there,
e.g. `videos/alice_bench1_take1.mp4` → `/videos/alice_bench1_take1.mp4`).

## Config (`config.yaml`)

```yaml
annotation: "Pick up the wrench from the table"   # ONE shared string per session
spans:
  - id: "span_001"                                # unique across all spans
    video: "videos/alice_bench1_take1.mp4"
    start: 4.2
    end: 11.0
    scene: "lab_bench_1"                          # grouping axis 1
    operator: "alice"                             # grouping axis 2
```

Edit `config.yaml`, then **Admin → Reload config** (or `POST /reload-config`) to
sync changes into the DB.

## Screens

- **#rate** — annotation, scene/operator filters, progress, looping clip, 1–5
  buttons, and bulk-score links for the current span's scene/operator.
- **#leaderboard** — per-span mean ± 95% CI, N, bulk%, sorted by mean desc, with
  Krippendorff's α below.
- **#admin** — raw ratings log + Reload config.

The `Rater` field in the header is the active `rater_id` (no auth this pass).

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/config` | annotation + distinct scenes/operators |
| GET | `/next?rater_id=&scene=&operator=` | next unrated span (random), or `null` |
| POST | `/rate` | `{span_id, rater_id, score}` — upsert individual rating |
| POST | `/bulk-rate` | `{group_by, group_value, rater_id, score}` — bulk-assign |
| GET | `/ratings?scene=&operator=` | per-span mean ± CI, n, bulk_fraction |
| GET | `/spans?scene=&operator=&rater_id=` | spans + `rated` bool for the rater |
| GET | `/iota?scene=&operator=` | Krippendorff's α (ordinal) or `null` |
| GET | `/raw-ratings` | full ratings log (admin screen) |
| POST | `/reload-config` | re-read config.yaml, re-upsert spans |

## Notes

- Ratings are the source of truth; means/CIs are recomputed on demand, never
  stored.
- One rating per `(span, rater)`, enforced at the API layer — re-rating via
  `/rate` replaces the score and resets `is_bulk` to 0.
- Bulk-rate never overwrites an existing individual rating (it skips it).
- SQLite for dev; the schema maps cleanly to Postgres later.
