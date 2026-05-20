# idle-filter-precomp — moving pause/idle filtering out of the training transform

## What changed

The pause/idle filter used to be a per-sample transform
(`PauseRemovalTransform` in `egomimic/rldb/zarr/action_chunk_transforms.py`)
inserted into the action-chunk transform pipeline. It re-computed the pause
mask on every `__getitem__` call, every epoch, for every chunk.

It is now an **episode-level precompute** that runs once on CPU after the
resolver loads episodes and before training starts.

### Before

```yaml
# mecka_all_zarr.yaml — old
resolver:
  _target_: ...LocalSQLEpisodeResolver
  folder_path: /mnt/zarr-data
  key_map: ...
  transform_list:
    _target_: ...Mecka.get_transform_list
    mode: cartesian
    pause_removal_epsilon: 0.005          # ← passed to the transform builder
```

`_build_aria_cartesian_bimanual_transform_list` would insert a
`PauseRemovalTransform(epsilon=0.005)` between the coordinate-frame
transforms and `InterpolatePose`. Every sampled chunk got its own pause
compression.

### After

```yaml
# mecka_all_zarr.yaml — new
resolver:
  _target_: ...LocalSQLEpisodeResolver
  folder_path: /mnt/zarr-data
  pause_removal_epsilon: 0.005            # ← now on the resolver
  key_map: ...
  transform_list:
    _target_: ...Mecka.get_transform_list
    mode: cartesian
```

`pause_removal_epsilon` is plumbed through the resolver into
`ZarrDataset(..., pause_removal_epsilon=...)`. After all datasets are
constructed, the resolver calls `_run_pause_precompute(datasets)`, which
fans out `precompute_pause_filter()` across episodes in a thread pool.
Each call:

1. Reads the full `left.obs_ee_pose` and `right.obs_ee_pose` arrays
   (small — ~24 KB each for a 3000-frame episode).
2. Builds a coordinated boolean keep-mask via `_build_pause_keep_mask`
   (a frame is a "pause step" iff **both** hands' consecutive deltas have
   L2 norm < `pause_removal_epsilon`). The first frame of a pause run is
   kept (transition); subsequent in-pause frames are dropped until motion
   resumes — same logic as the old `PauseRemovalTransform._compress`,
   applied at the whole-episode level.
3. Stores `keep_indices` on the dataset.

After precompute, `ZarrDataset.__len__` returns `len(keep_indices)` and
`__getitem__(idx)` translates `idx → keep_indices[idx]` before any zarr
read, so the **dataset PyTorch sees is the filtered one**. Chunks remain
contiguous reads from the original episode starting at the kept frame
(see the "design tradeoff" note below).

The aggregate kept/total summary is logged once per resolver, e.g.

```
INFO  Pause precompute (epsilon=0.005): kept 1832145/2017804 frames (90.8%)
      across 14 episodes in 0.6s (errors=0)
```

## Why

1. **CPU savings.** Pause detection ran on every sample, every epoch. For
   N samples × E epochs, that's `O(N·E)` repeated work; the new path is
   `O(N)` at startup.
2. **Correctness.** The old transform applied independently per key (left
   vs right). Chunks for the left hand could collapse different frames
   than the right hand, then both got padded to length H — silently
   de-aligning the two action streams. The precompute uses a single
   coordinated mask, so both hands share the same kept-frame timeline.
3. **Visible dataset alteration.** The size of the dataset PyTorch
   iterates over now actually drops, which is straightforward to assert
   in tests and observe in logs.

## How to verify

### Unit tests

```
source emimic/bin/activate
pytest egomimic/rldb/zarr/test_pause_precompute.py -v
```

The suite (8 tests) covers:

- `test_build_pause_keep_mask_matches_reference` (4 parametrizations) —
  the new helper produces the same mask as a fresh re-implementation of
  the old `PauseRemovalTransform._compress` loop, on synthetic episodes
  with known pause spans.
- `test_build_pause_keep_mask_short_episode` — 0- and 1-frame edges.
- `test_zarr_dataset_precompute_alters_length` — `__len__` falls from
  raw `total_frames` to `len(keep_indices)` after
  `precompute_pause_filter()` is invoked. **This proves the dataset is
  actually altered, not just the per-sample chunk content.**
- `test_zarr_dataset_getitem_uses_keep_indices` — a logical idx resolves
  through `keep_indices` to a non-paused real frame.
- `test_zarr_dataset_precompute_is_idempotent` — repeat calls are no-ops.

### End-to-end Modal smoke — completed

```
source emimic/bin/activate
export MODAL_ENVIRONMENT=robotics

modal run egomimic/modal/run.py::submit -- \
    data=mecka_all_zarr_smoke \
    trainer=debug_modal \
    model=hpt_bc_flow_mecka \
    logger=tensorboard \
    name=idle_filter_smoke \
    description=precompute_verify
```

`debug_modal` runs 2 train batches × 3 val batches × 4 epochs — total
cost is dominated by image build and cold start. Look for the
`Pause precompute (epsilon=...)` line in the logs to confirm the new
path ran, and for a kept-fraction < 100 % to confirm it altered the
dataset.

Result from the run on commit `53b670b` (Modal app
`ap-LUXQEISLXFuVw11C1zcjn4`):

```
Loaded 14 episodes from local volume
Pause precompute (epsilon=0.005): kept 25433/25470 frames (99.9%) across 14 episodes in 1.7s (errors=0)
... (logged once for the train resolver, once for the valid resolver)
Trainer.fit stopped: max_epochs=4 reached.
```

37 frames dropped across 14 episodes. The chosen smoke episodes happen
to be near-fully-active, so the dropped fraction is small; the
mechanism is exercised end-to-end. The
"`Found zero norm quaternions in 'quat'`" warnings logged during
training are a pre-existing data issue at episode endings (the H=30
horizon chunk reads past the last frame and pads with the last raw
value, which is zero in some episodes) and are handled by the existing
random-idx fallback in `ZarrDataset.__getitem__`. Not a regression
from this change.

## Design tradeoff: start-frame filter, not within-chunk gather

Chunks are still read **contiguously** from the original episode starting
at `keep_indices[idx]`. This means:

- Pauses are removed from the set of **sample start frames** (the dataset
  shrinks by the pause fraction).
- Pauses **inside** an action chunk are still read as-is from the
  original episode.

The old per-sample transform compressed pauses inside each chunk and
padded back with the last frame. The new precompute does not — pauses
that fall mid-chunk pass through. If we want strict within-chunk pause
removal we'd need fancy zarr indexing (`arr.get_coordinate_selection`)
to gather kept frames; that's a larger change and was deferred. See
`_run_pause_precompute` in `egomimic/rldb/zarr/zarr_dataset_multi.py`
and the comment in `__getitem__`.

## Open items

- **`offline_norm_stats.py` no longer applies any pause filter** when
  collecting normalization stats. Stats are collected on the raw
  (pre-precompute) frame distribution, which can drift slightly from
  what training actually sees. The comment in the file documents this;
  mirror `precompute_pause_filter` there if exact alignment is needed.
- **Modal SQL fast-path opens zarr stores when `pause_removal_epsilon`
  is set.** The previous optimization that deferred `zarr.open` (when
  `_total_frames` and `_embodiment` were known from SQL) is lost in this
  mode, because the precompute needs to read `obs_ee_pose` arrays. For
  the full 198k-episode dataset this adds noticeable startup time; the
  thread-pool fan-out in `_run_pause_precompute` mitigates it but
  doesn't eliminate it.
