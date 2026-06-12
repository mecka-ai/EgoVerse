# Viewer test harnesses (no node needed)

- `test_prune_flow.py` — venv-side end-to-end: fake grading run + viz run →
  `review_tools pair` → `build_html` (writes `/tmp/prune_viewer.html`) →
  `apply` → `calibrate --selection-labels`.
  Run: `source <repo>/emimic/bin/activate && python test_prune_flow.py`
- `test_viewer_prune_js.py` — executes the generated viewer JS under quickjs
  (cluster determinism, prune ordering, hidden removed points, click cost,
  exports). Run: `pip install --target /tmp/jsengine2 quickjs` once, then
  `/usr/bin/python3.12 test_viewer_prune_js.py` (the wheel is cpython-3.12).
  Requires `/tmp/prune_viewer.html` from the flow test first.
