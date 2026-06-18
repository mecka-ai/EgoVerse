# MSE viewer test harness

No node required — the viewer JS is tested under **quickjs** (a tiny embeddable
JS engine) driven from Python, mirroring `egomimic/scripts/knn_grading/viewer_harness/`.

## Run

```bash
# 1) Flow test (stdlib only): builds /tmp/mse_viewer.html from fake scores and
#    round-trips a selection through mse_apply_selection.py.
python egomimic/scripts/mse_viewer_harness/test_mse_flow.py

# 2) JS test: execute the viewer's selection logic under quickjs.
pip install --target /tmp/jsengine2 quickjs      # one-time
python3.12 egomimic/scripts/mse_viewer_harness/test_mse_viewer_js.py
```

`test_mse_flow.py` must run first — it writes `/tmp/mse_viewer.html` that the JS
test loads.

## What's covered

- `selectKept` keeps the lowest-MSE / highest-MSE top-k% correctly, with the
  k=0 / k=100 edges, deterministic hash tie-break, and **non-finite (unscored)
  episodes always kept** (both `higher_is_worse` directions exercised).
- `countKept`, `keptHashesAllTasks` (union across tasks + keep-all-uncovered
  universe), and `percentileNorm` goodness direction.
- `render()` and the exports (`exportEpsToUse`, `exportSelection`) run without
  throwing and produce the expected allowlist / selection schema.
- `mse_apply_selection.py apply` unions kept episodes, keeps uncovered universe
  episodes, and writes a `.provenance.json` sidecar.
