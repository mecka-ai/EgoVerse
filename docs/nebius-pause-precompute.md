# Pause-filter precompute on Nebius / SLURM

This is the SLURM-native equivalent of the Modal-based pause-precompute
fan-out. It produces the same JSON cache file that
`egomimic.rldb.zarr.zarr_dataset_multi._apply_pause_precompute_cache`
already knows how to consume — so the training process is unchanged.

## Why a precompute step?

Pause filtering decides, per episode, which raw frame indices to keep. A
"pause step" is a frame where every available motion signal moved less
than `pause_removal_epsilon` since the previous frame:

- left/right `obs_ee_pose` L2 delta < ε  (wrist motion), **and**
- if `obs_keypoints` are present, max-over-landmarks delta < ε on each
  hand (catches finger flexion when the wrist is still).

The first frame of each pause run is kept (the transition into pause);
subsequent in-pause frames are dropped until motion resumes.
Implementation: `_build_pause_keep_mask` in
`egomimic/rldb/zarr/zarr_dataset_multi.py`.

Running this in-process at the start of every training job is fine for
small datasets (the existing `_inprocess_pause_precompute` thread pool
handles this) but at ~70k mecka episodes it adds ~10–30 min of CPU work
before the first GPU step. The fan-out moves that work onto cheap CPU
nodes, in parallel, and caches the result. The training job then reads a
pre-built JSON in ~1 s.

## How the SLURM version works

```
login node                              compute nodes (array)              login node
─────────────                           ─────────────────────              ─────────────
pause_precompute_driver.py
   │
   ├── hydra-compose your training config
   ├── walk train_datasets + valid_datasets
   │   for every resolver with pause_removal_epsilon
   ├── ModalEpisodeResolver.discover_episode_paths(filters)
   │     → [(hash, /shared/zarr/<hash>), ...]
   ├── group by epsilon, partition into N shards
   ├── write <out-dir>/manifest.jsonl                        sbatch array task
   ├── sbatch --array=0-(N-1)%C pause_precompute.sbatch ─────▶  reads its
   │                                                            manifest line,
   │                                                            processes shard,
   │                                                            writes
   │                                                            shards/shard_<i>.json
   │                                                                  │
   ├── poll sacct until tasks finish ◀────────────────────────────────┘
   └── aggregate shards/*.json → cache.json                  ──▶ cache.json
                                                                        │
                                                            training job reads it
                                                            via $EGOMIMIC_PAUSE_PRECOMPUTE_CACHE
```

Cache JSON shape (consumer contract):

```json
{
  "<episode_hash>": {
    "raw_total": 1234,
    "keep_indices": [0, 1, 2, 5, 7, ...]
  },
  ...
}
```

`raw_total == 0` is the miss sentinel — the consumer treats those as if
they were never cached and falls back to in-process precompute for those
episodes.

## One-shot run

On the cluster login node, in an environment that can `import egomimic`
and reach the SQL episode table:

```bash
# 1. Run the driver. It composes the config, walks resolvers, sbatches
#    an array job, polls until done, and aggregates the cache.
python -m egomimic.scripts.nebius.pause_precompute_driver \
    --config-name train_zarr_cartesian \
    --overrides data=mecka_50k_20k \
    --out-dir /shared/pause/run-$(date +%Y%m%d_%H%M%S) \
    --shards 100 --concurrency 50 \
    --partition cpu --time 00:30:00 --mem 16G --cpus-per-task 8

# 2. Driver prints the env var to set. Export it before training.
export EGOMIMIC_PAUSE_PRECOMPUTE_CACHE=/shared/pause/run-.../cache.json

# 3. Launch training as normal. The resolver auto-detects the env var.
sbatch your-training-job.sbatch  # or python egomimic/trainHydra.py ...
```

## Action-chunk filtering (no code change required)

The consumer side does this for free. `ZarrDataset.__getitem__` checks
`self.keep_indices` for both:

1. Single-frame reads (`real_idx = keep_indices[idx]`).
2. Horizon-H chunk reads — when `keep_indices` is set, the chunk
   fancy-indexes `keep_indices[idx : idx + horizon]` and reads only those
   filtered frames from zarr. So an action chunk of length 100 starting
   at a frame near a pause region will still contain 100 non-paused
   frames, with raw paused frames stitched out.

This is the "fully-filtered chunk" behavior added in
`pause-filter-keypoints-and-filtered-chunks`. The Nebius port doesn't
need to re-implement it — it just produces the `keep_indices` that
make it work.

## Re-runs and idempotency

- Each worker writes `shards/shard_<id>.json` atomically (write to
  `.tmp`, rename). If the file already exists, it's skipped — pass
  `--force` to the worker to rebuild.
- The aggregator picks up whatever shards exist on disk. Partial array
  completion (e.g. one node timed out) still produces a usable cache;
  the missing episodes fall back to in-process precompute at training time.
- The driver writes the SLURM job id to `<out-dir>/slurm_job_id`.
  `sacct -j $(cat slurm_job_id)` after the fact tells you what happened.

## Cluster-specific setup

The bundled `pause_precompute.sbatch` is a starting template. Real
clusters need site-specific glue:

### venv-based clusters

```bash
export EGOMIMIC_VENV_ACTIVATE=/shared/envs/emimic/bin/activate
export EGOMIMIC_REPO_ROOT=/shared/repos/EgoVerse
python -m egomimic.scripts.nebius.pause_precompute_driver ...
```

`pause_precompute.sbatch` sources `$EGOMIMIC_VENV_ACTIVATE` and prepends
`$EGOMIMIC_REPO_ROOT` to `PYTHONPATH`.

### enroot/pyxis (typical Nebius managed Slurm)

Add these `#SBATCH` lines to the template or pass via `--extra-sbatch`:

```
#SBATCH --container-image=/shared/images/egomimic.sqsh
#SBATCH --container-mounts=/shared:/shared
```

Inside the container, `python` resolves to the baked image's interpreter
and the `egomimic` package is already installed — no venv activation
needed.

### Account / QOS overrides

Anything sbatch accepts can pass through:

```bash
python -m egomimic.scripts.nebius.pause_precompute_driver \
    ... \
    --extra-sbatch -- --account=ai-research --qos=normal --gres=none
```

## Comparison to the Modal version

| concern              | Modal (`egomimic-scan::pause_precompute_shard`)        | SLURM/Nebius (this)                                      |
|----------------------|---------------------------------------------------------|----------------------------------------------------------|
| Producer trigger     | `_modal_fanout_pause_precompute` inside hydrated worker | `pause_precompute_driver` on login node                  |
| Work discovery       | `resolver._resolve_episode_meta(filters)`               | `resolver.discover_episode_paths(filters)`               |
| Fan-out unit         | `fn.map(shards, epsilons)` — Modal containers           | sbatch array task, one per shard                         |
| Per-shard worker     | `@app.function pause_precompute_shard`                  | `pause_precompute_worker.py` invoked from sbatch         |
| Cache file location  | `/tmp/pause_precompute_cache.json` (container scratch) | `<out-dir>/cache.json` on shared FS                      |
| Consumer             | `_apply_pause_precompute_cache` via `$EGOMIMIC_PAUSE_PRECOMPUTE_CACHE` | **same** |
| Filter algorithm     | `_build_pause_keep_mask` (ee_pose + keypoints L/R)      | **same** — imported, not duplicated                      |

The consumer side is unchanged, so a single training script can run
under either backend just by pointing `EGOMIMIC_PAUSE_PRECOMPUTE_CACHE`
at the right file.

## When not to use this

- **Few episodes** (< ~1k): in-process precompute is fast enough that
  fan-out scheduling overhead dominates. Just let `trainHydra.py` do it
  in-process.
- **Iterating on the filter algorithm**: re-running fan-out for every
  tweak is slow. Develop in-process first; precompute when the algorithm
  is stable.
- **No shared filesystem**: this assumes manifest + shard outputs live
  on a path visible from both the login node and every compute node.
  Without that, fall back to the in-process path.
