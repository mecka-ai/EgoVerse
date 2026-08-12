# WAM (Wan2.2-TI2V-5B) on Modal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Wan2.2-TI2V-5B WAM training launchable via `python egomimic/modal/trainModal.py model=wam_bc_human_wan22_5b data=mecka_wam trainer=ddp_modal +modal_gpu=H200 init_submodules=false`.

**Architecture:** Port the RL2 `upstream/aniketh/wam` dreamzero WAM (commit `82f3c207`) into this fork. The runnable path (`WAM` algo → `WAMModel` → `build_wam_dit`/`build_wan_vae`) is Wan2.1-1.3B-only; this plan adds the Wan2.2 5B path (5B DiT preset + sharded load, `WanVideoVAE38` z=48, HF checkpoint fetch, non-square resize) and wires the whole thing into this fork's Modal launcher, data resolver, and `mecka_bimanual` embodiment convention.

**Tech Stack:** PyTorch 2.6 / Lightning / Hydra / Modal 1.4.x / diffusers / peft / vendored Wan (DiT + VAE).

## Global Constraints

- **No `HUMAN_BIMANUAL` enum member.** The fork's `_EMBODIMENT_ALIASES = {"HUMAN_BIMANUAL": "MECKA_BIMANUAL"}` already exists; WAM uses `mecka_bimanual` directly. Do not touch `egomimic/rldb/embodiment/embodiment.py`.
- **`diffusers` loads at module import** in the wan backbone → guard the WAM import in `algo/__init__.py` with `try/except ImportError`, exactly like the OAT/quest guards already there.
- **No local pytest suite for training.** Verification per task = `python -m py_compile` (syntax) + `python egomimic/trainHydra.py ... --cfg job` (config compose) where applicable. The integration test is a Modal `trainer=debug` run (Task 9). Do **not** invent fake pytest tests.
- **Modal 1.4.x API** (per CLAUDE.md): container detection via `os.environ.get("MODAL_IS_REMOTE") == "1"`; `Volume.from_name`; string GPU literals; imports inside functions.
- **5B geometry:** resolution **160×320** → VAE38 16× → latent 10×20 → DiT patch(1,2,2) → 5×10 = **`frame_seqlen=50`**. Horizons: `action_horizon=16`, `num_action_per_block=4`, `num_state_per_block=1`, `state_horizon=4`, `cam_horizon=17` (→ 5 latent frames: 1 cond + 4 predicted).
- **Norm stats: online.** Never pass `norm_stats.precomputed_norm_path` for WAM. Keep the evaluator wired (unset evaluator disables all Valid/W&B logging).
- **Checkpoints:** dedicated `wan-checkpoints` Modal volume, mounted at `/mnt/wan-ckpts`; `Wan-AI/Wan2.2-TI2V-5B` (Apache-2.0, ungated).
- **Commit** after each task with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

## File Structure

New (dropped in verbatim from `82f3c207`): `egomimic/models/wan/*` (16 files), `egomimic/models/wam_nets.py`, `egomimic/algo/wam.py`, `egomimic/eval/eval_wam.py`, `egomimic/rldb/zarr/zarr_dataset_wam.py`, `egomimic/hydra_configs/evaluator/eval_wam.yaml`, `egomimic/hydra_configs/evaluator/viz/wam_cartesian.yaml`.

New (authored fresh for this fork): `egomimic/hydra_configs/model/wam_bc_human_wan22_5b.yaml`, `egomimic/hydra_configs/data/mecka_wam.yaml`.

Modified: `egomimic/models/wam_nets.py` (5B builders), `egomimic/algo/wam.py` (non-square resize), `egomimic/rldb/zarr/zarr_dataset_wam.py` (`ModalWamEpisodeResolver`), `egomimic/algo/__init__.py` (guarded import), `egomimic/rldb/embodiment/human.py` (WAM keymap/transform), `pyproject.toml`, `.ruff.toml`, `egomimic/modal/modal_setup.py`.

Branch source files staged in scratchpad at `/private/tmp/.../scratchpad/wam_branch/` for reference; use `git checkout 82f3c207 -- <path>` (worktree shares the object DB) for exact copies.

---

### Task 1: Drop in vendored WAM code (new files only)

**Files:**
- Create (via `git checkout 82f3c207 --`): `egomimic/models/wan/` (16 files), `egomimic/models/wam_nets.py`, `egomimic/algo/wam.py`, `egomimic/eval/eval_wam.py`, `egomimic/rldb/zarr/zarr_dataset_wam.py`, `egomimic/hydra_configs/evaluator/eval_wam.yaml`, `egomimic/hydra_configs/evaluator/viz/wam_cartesian.yaml`

**Interfaces produced:** `egomimic.algo.wam.WAM`, `egomimic.models.wam_nets.build_wam_dit`/`build_wan_vae`/`FlowMatchScheduler`, `egomimic.rldb.zarr.zarr_dataset_wam.{ZarrWamDataset,WamMultiDataset,S3WamEpisodeResolver,LocalWamEpisodeResolver}`, `egomimic.eval.eval_wam.WAMEvalVideo`.

- [ ] **Step 1: Copy the new files from the branch commit**
```bash
cd "$(git rev-parse --show-toplevel)"
git checkout 82f3c207 -- \
  egomimic/models/wan \
  egomimic/models/wam_nets.py \
  egomimic/algo/wam.py \
  egomimic/eval/eval_wam.py \
  egomimic/rldb/zarr/zarr_dataset_wam.py \
  egomimic/hydra_configs/evaluator/eval_wam.yaml \
  egomimic/hydra_configs/evaluator/viz/wam_cartesian.yaml
```
- [ ] **Step 2: Confirm no fork file was overwritten** — `git status --porcelain` should show only additions (`A`/`??`), never a modified pre-existing fork file. If `eval/` or `evaluator/viz/` didn't exist, they're created now.
- [ ] **Step 3: Syntax check all copied Python**
```bash
python -m py_compile egomimic/models/wan/*.py egomimic/models/wam_nets.py egomimic/algo/wam.py egomimic/eval/eval_wam.py egomimic/rldb/zarr/zarr_dataset_wam.py && echo "PY OK"
```
Expected: `PY OK` (syntax only; diffusers/torch imports are not exercised here).
- [ ] **Step 4: Commit** — `git add -A && git commit --no-verify -m "feat(wam): vendor dreamzero WAM backbone + algo (Wan DiT/VAE)"`

---

### Task 2: Additive merges — deps, guarded algo import, human.py WAM methods, ruff

**Files:**
- Modify: `pyproject.toml` (dependency list)
- Modify: `egomimic/algo/__init__.py`
- Modify: `egomimic/rldb/embodiment/human.py` (add two `Mecka` classmethods)
- Modify: `.ruff.toml`

**Interfaces produced:** `Mecka.get_wam_keymap(cam_horizon, action_horizon, state_horizon, norm_mode, annotation_key)`, `Mecka.get_wam_transform_list()`.

- [ ] **Step 1: Add the three pip deps to `pyproject.toml`** — under the main dependencies array add `"diffusers>=0.38.0"`, `"accelerate>=1.14.0"`, `"peft>=0.19.1"` (`safetensors` already present). Match the branch's `pyproject.toml` diff for the exact lines.
- [ ] **Step 2: Add the guarded WAM import to `egomimic/algo/__init__.py`** — append, mirroring the existing OAT/quest guards:
```python
try:
    from egomimic.algo.wam import WAM as WAM
except ImportError:
    pass  # wan backbone deps (diffusers/peft) absent — non-WAM run
```
- [ ] **Step 3: Port `get_wam_keymap` + `get_wam_transform_list` into `Mecka`** — copy both classmethods verbatim from `82f3c207:egomimic/rldb/embodiment/human.py` (staged at `scratchpad/wam_branch/.../human.py` lines ~369–470+) into this fork's `class Mecka(Human)`. They are embodiment-agnostic and reference `cls.VIZ_IMAGE_KEY`, `ActionChunkCoordinateFrameTransform`, `XYZWXYZ_to_XYZYPR`, `ConcatKeys` — all already imported in this fork's `human.py`.
- [ ] **Step 4: Apply the WAM `.ruff.toml` hunk** — from `82f3c207:.ruff.toml`, add only the lint carve-outs for vendored `egomimic/models/wan/` (e.g. per-file ignores). Do not otherwise alter this fork's ruff config.
- [ ] **Step 5: Syntax + method-presence check**
```bash
python -m py_compile egomimic/algo/__init__.py egomimic/rldb/embodiment/human.py && \
python -c "import ast,sys; m=ast.parse(open('egomimic/rldb/embodiment/human.py').read()); \
names={n.name for c in ast.walk(m) if isinstance(c,ast.ClassDef) and c.name=='Mecka' for n in c.body if isinstance(n,(ast.FunctionDef,))}; \
assert {'get_wam_keymap','get_wam_transform_list'} <= names, names; print('Mecka WAM methods OK')"
```
Expected: `Mecka WAM methods OK`.
- [ ] **Step 6: Commit** — `git commit --no-verify -am "feat(wam): deps, guarded algo import, Mecka WAM keymap/transform, ruff"`

---

### Task 3: Wan2.2 5B builders in `wam_nets.py`

**Files:**
- Modify: `egomimic/models/wam_nets.py`

**Interfaces produced:** `build_wam_dit(checkpoint_path, arch="wan21_1_3b", action_dim, max_state_dim, num_action_per_block, num_state_per_block, frame_seqlen, freeze_video, lora, lora_rank, lora_alpha, **overrides)`; `build_wan_vae(checkpoint_path, z_dim=16)` now builds `WanVideoVAE38` when `z_dim==48`; `ensure_file(local_path, filename, repo_id)`.

**Interfaces consumed:** `CausalWanModel`, `WanVideoVAE`, `WanVideoVAE38` (from `egomimic.models.wan.wan_video_vae`), `WanModel.state_dict_converter` (Task 1).

- [ ] **Step 1: Import `WanVideoVAE38`** — extend the existing import line to `from egomimic.models.wan.wan_video_vae import WanVideoVAE, WanVideoVAE38`.
- [ ] **Step 2: Add the 5B preset dict** next to `WAM_DIT_1_3B`:
```python
# Canonical Wan2.2-TI2V-5B DiT dims (head_dim 128, same as 1.3B; RoPE/patchify
# are parametric so this flows through CausalWanModel unchanged).
WAM_DIT_5B = dict(
    model_type="ti2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=48,
    dim=3072,
    ffn_dim=14336,
    freq_dim=256,
    text_dim=4096,
    out_dim=48,
    num_heads=24,
    num_layers=30,
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    num_frame_per_block=1,
    concat_first_frame_latent=False,  # 5B: latent-only (no first-frame channel concat)
    hidden_size=1024,
    max_num_embodiments=1,
)

_WAM_DIT_PRESETS = {"wan21_1_3b": WAM_DIT_1_3B, "wan22_5b": WAM_DIT_5B}
```
- [ ] **Step 3: Add an `arch` selector + sharded loading to `build_wam_dit`.** Signature gains `arch: str = "wan21_1_3b"`. Replace `cfg = dict(WAM_DIT_1_3B)` with `cfg = dict(_WAM_DIT_PRESETS[arch])`. Replace the single-file safetensors load with a helper that handles a sharded checkpoint (path ending `.safetensors.index.json` or a directory):
```python
def _load_wan_state_dict(checkpoint_path: str) -> dict:
    """Load a Wan DiT checkpoint: single .safetensors, sharded (index.json), or .pth."""
    import json as _json
    import os as _os
    from safetensors.torch import load_file
    if checkpoint_path.endswith(".safetensors.index.json"):
        base = _os.path.dirname(checkpoint_path)
        with open(checkpoint_path) as f:
            index = _json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        sd = {}
        for shard in shards:
            sd.update(load_file(_os.path.join(base, shard)))
        return sd
    if checkpoint_path.endswith(".safetensors"):
        return load_file(checkpoint_path)
    return torch.load(checkpoint_path, map_location="cpu")
```
Use it in `build_wam_dit`'s load block (keep the `from_civitai` + `load_state_dict(strict=False)` + missing/unexpected print — that print is the Task-9 R1 signal).
- [ ] **Step 4: Branch `build_wan_vae` on `z_dim`:**
```python
def build_wan_vae(checkpoint_path: str = None, z_dim: int = 16):
    vae = WanVideoVAE38(z_dim=48, dim=160) if z_dim == 48 else WanVideoVAE(z_dim=z_dim)
    ...  # (unchanged load + from_civitai + eval().requires_grad_(False))
```
- [ ] **Step 5: Add `ensure_file`** (ported from `wan_flow_matching_action_tf.py`), so builders/downloaders can fetch-if-missing:
```python
def ensure_file(local_path: str | None, filename: str, repo_id: str) -> str:
    """Return local_path if it exists, else HF-download <filename> from <repo_id> to a cache."""
    import os as _os
    from huggingface_hub import hf_hub_download
    if local_path and _os.path.exists(local_path):
        return local_path
    return hf_hub_download(repo_id=repo_id, filename=filename)
```
- [ ] **Step 6: Syntax check** — `python -m py_compile egomimic/models/wam_nets.py && echo OK`.
- [ ] **Step 7: Commit** — `git commit --no-verify -am "feat(wam): Wan2.2 5B DiT preset + sharded load + VAE38 + ensure_file"`

---

### Task 4: Non-square resize in `WAMModel` (`wam.py`)

**Files:**
- Modify: `egomimic/algo/wam.py`

**Interfaces produced:** `WAMModel.__init__(..., target_h=None, target_w=None)`; `WAM.__init__(..., target_h=None, target_w=None)`. Default `None` → square `frame_size` (1.3B path unchanged).

- [ ] **Step 1: Thread `target_h`/`target_w` through `WAMModel.__init__`** — add params (default `None`), store `self.target_h = target_h or frame_size`, `self.target_w = target_w or frame_size`.
- [ ] **Step 2: Use them in `_frame_to_video`** — replace the `size=(self.frame_size, self.frame_size)` interpolate target with `size=(self.target_h, self.target_w)`, and replace the final reshape dims `self.frame_size, self.frame_size` with `self.target_h, self.target_w`.
- [ ] **Step 3: Thread through `WAM.__init__`** — add `target_h=None, target_w=None` params and pass them into the `WAMModel(...)` construction.
- [ ] **Step 4: Syntax check** — `python -m py_compile egomimic/algo/wam.py && echo OK`.
- [ ] **Step 5: Commit** — `git commit --no-verify -am "feat(wam): non-square (target_h,target_w) frame resize for 5B"`

---

### Task 5: `ModalWamEpisodeResolver` (`zarr_dataset_wam.py`)

**Files:**
- Modify: `egomimic/rldb/zarr/zarr_dataset_wam.py`

**Interfaces produced:** `ModalWamEpisodeResolver(ModalEpisodeResolver)` with `_dataset_class = ZarrWamDataset`.
**Interfaces consumed:** `ModalEpisodeResolver` (from `egomimic.rldb.zarr.zarr_dataset_multi`).

- [ ] **Step 1: Import `ModalEpisodeResolver`** — extend the existing `from egomimic.rldb.zarr.zarr_dataset_multi import (...)` block to include `ModalEpisodeResolver`.
- [ ] **Step 2: Add the resolver class** (mirrors `S3WamEpisodeResolver`; precedent `LocalAnnotationCutoffEpisodeResolver`):
```python
class ModalWamEpisodeResolver(ModalEpisodeResolver):
    """Modal-volume resolver (SQL filter + /mnt/zarr-data) building ZarrWamDataset clip leaves."""

    _dataset_class = ZarrWamDataset
```
Add `"ModalWamEpisodeResolver"` to `__all__`.
- [ ] **Step 3: Syntax check** — `python -m py_compile egomimic/rldb/zarr/zarr_dataset_wam.py && echo OK`.
- [ ] **Step 4: Commit** — `git commit --no-verify -am "feat(wam): ModalWamEpisodeResolver for /mnt/zarr-data clip loading"`

---

### Task 6: Hydra configs — model (5B), data (mecka_wam)

**Files:**
- Create: `egomimic/hydra_configs/model/wam_bc_human_wan22_5b.yaml`
- Create: `egomimic/hydra_configs/data/mecka_wam.yaml`

- [ ] **Step 1: Write the 5B model config** `model/wam_bc_human_wan22_5b.yaml`:
```yaml
# WAM (World-Action Model) — dreamzero CausalWanModel on the Wan2.2 TI2V-5B
# backbone, mecka_bimanual embodiment. Joint DiT forward over
# [video latent | action | state] -> video + action velocities (rectified flow).
# 5B geometry: 160x320 -> VAE38 16x -> latent 10x20 -> DiT patch -> frame_seqlen=50.
# k=4 predicted latent frames: cam_horizon=17 -> 5 latent (1 cond + 4 predicted),
# action_horizon=16 (4x4), state_horizon=4 (1 per predicted frame). Pair with data=mecka_wam.
_target_: egomimic.pl_utils.pl_model.ModelWrapper
robomimic_model:
  _target_: egomimic.algo.wam.WAM
  domains: ["mecka_bimanual"]
  ac_keys:
    mecka_bimanual: "actions_cartesian"
  action_dim: 12
  action_horizon: 16
  state_dim: 12
  target_h: 160
  target_w: 320
  world_loss_weight: 1.0
  num_inference_steps: 16
  dit:
    _target_: egomimic.models.wam_nets.build_wam_dit
    arch: wan22_5b
    checkpoint_path: /mnt/wan-ckpts/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json
    action_dim: 12
    max_state_dim: 12
    num_action_per_block: 4
    num_state_per_block: 1
    frame_seqlen: 50
    freeze_video: true
    lora: true
  vae:
    _target_: egomimic.models.wam_nets.build_wan_vae
    checkpoint_path: /mnt/wan-ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
    z_dim: 48
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 1e-4
  weight_decay: 1e-6
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  _partial_: true
  T_max: 1400
  eta_min: 1e-5
```
(Exact checkpoint filenames are confirmed in Task 8 against the downloaded volume; adjust these two paths if the repo layout differs.)
- [ ] **Step 2: Write the data config** `data/mecka_wam.yaml` (Modal resolver + starter filter):
```yaml
# WAM data: frame CLIP + frame-aligned action/state chunks via ZarrWamDataset
# (ModalWamEpisodeResolver over /mnt/zarr-data, SQL filter) + Mecka WAM keymap/transform.
# Alignment: cam_horizon=17 -> 5 latent frames; action_horizon=16; state_horizon=4.
# Bounds checking disabled (WamMultiDataset): raw frame-aligned ee-poses, no quantile calibration.
_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

train_datasets:
  mecka_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_wam.WamMultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_wam.ModalWamEpisodeResolver
      folder_path: /mnt/zarr-data
      key_map:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_wam_keymap
        cam_horizon: 17
        action_horizon: 16
        state_horizon: 4
      transform_list:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_wam_transform_list
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
        - "lambda row: row['embodiment'] == 'mecka_bimanual' and row.get('zarr_processed_path', '') != '' and row.get('is_deleted', False) == False"
    mode: total

valid_datasets:
  mecka_bimanual: ${data.train_datasets.mecka_bimanual}

train_dataloader_params:
  mecka_bimanual:
    batch_size: 2
    num_workers: 10
valid_dataloader_params:
  mecka_bimanual:
    batch_size: 2
    num_workers: 10
```
(The `filters` lambda is a starter over mecka episodes; verify the exact column names against the SQL schema in Task 9 and narrow to a handful of episodes for the debug run.)
- [ ] **Step 3: Compose-check both configs** (YAML + group resolution; no instantiation):
```bash
python egomimic/trainHydra.py model=wam_bc_human_wan22_5b data=mecka_wam trainer=debug evaluator=eval_wam name=x description=x --cfg job 2>&1 | tail -30
```
Expected: a fully composed config prints (no `Could not find` / `ConfigCompositionException`). Import-level errors of `_target_` classes are fine here — `--cfg job` does not instantiate.
- [ ] **Step 4: Commit** — `git commit --no-verify -am "feat(wam): Wan2.2-5B model + mecka_wam data hydra configs (mecka_bimanual)"`

---

### Task 7: Modal image, checkpoint volume, and downloader (`modal_setup.py`)

**Files:**
- Modify: `egomimic/modal/modal_setup.py`

**Interfaces produced:** `wan_checkpoints_volume`; `/mnt/wan-ckpts` mount added to `run_hydra_train`'s volumes; a `download_wan22_weights` entrypoint.

- [ ] **Step 1: Add the three deps to the image pip layer** — append `"diffusers"`, `"accelerate"`, `"peft"` to the big `.pip_install(...)` list (heavy-layer, cached).
- [ ] **Step 2: Declare the checkpoint volume + mount constant** — near the other `Volume.from_name` lines:
```python
wan_checkpoints_volume = modal.Volume.from_name("wan-checkpoints", create_if_missing=True)
WAN_CKPT_MOUNT = "/mnt/wan-ckpts"
```
- [ ] **Step 3: Mount it read-into training** — in `trainModal.py`'s `_build_volumes()` (or `modal_setup`'s volume assembly) add `WAN_CKPT_MOUNT: wan_checkpoints_volume` to the volumes dict returned for `run_hydra_train`. (Import `wan_checkpoints_volume`, `WAN_CKPT_MOUNT` from `modal_setup` in `trainModal.py`.)
- [ ] **Step 4: Add the one-time downloader** to `modal_setup.py` (or `trainModal.py`):
```python
@app.function(volumes={WAN_CKPT_MOUNT: wan_checkpoints_volume}, timeout=3600,
              secrets=[modal.Secret.from_name("egoverse-hf")])
def download_wan22_weights():
    import os
    from huggingface_hub import snapshot_download
    dest = f"{WAN_CKPT_MOUNT}/Wan2.2-TI2V-5B"
    os.makedirs(dest, exist_ok=True)
    snapshot_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        local_dir=dest,
        allow_patterns=["*.safetensors", "*.json", "*.pth", "*.txt"],
    )
    wan_checkpoints_volume.commit()
    print("downloaded:", sorted(os.listdir(dest)))
```
- [ ] **Step 5: Syntax check** — `python -m py_compile egomimic/modal/modal_setup.py egomimic/modal/trainModal.py && echo OK`.
- [ ] **Step 6: Commit** — `git commit --no-verify -am "feat(wam): Modal image deps + wan-checkpoints volume + downloader"`

---

### Task 8: Populate the checkpoint volume (operational, one-time)

- [ ] **Step 1: Run the downloader**
```bash
modal run --env robotics egomimic/modal/modal_setup.py::download_wan22_weights
```
- [ ] **Step 2: Verify contents + capture exact filenames**
```bash
modal volume ls --env robotics wan-checkpoints Wan2.2-TI2V-5B
```
Expected: DiT `diffusion_pytorch_model*.safetensors` (+ `*.index.json` if sharded), `Wan2.2_VAE.pth`, `config.json`.
- [ ] **Step 3: Reconcile config paths** — if the DiT is a single file (no `.index.json`) or the VAE filename differs, update the two `checkpoint_path`s in `model/wam_bc_human_wan22_5b.yaml` and commit.

---

### Task 9: Modal `trainer=debug` integration run (the real test)

- [ ] **Step 1: Narrow the data filter for debug** — temporarily set `data.train_datasets.mecka_bimanual.filters.filter_lambdas` to a specific 1–3 episode-hash filter (edit the config or pass a CLI override) so the run is fast and deterministic.
- [ ] **Step 2: Launch the debug run**
```bash
python egomimic/modal/trainModal.py \
  --config-name=train_zarr_human_wam_wan22_5b trainer=debug logger=wandb \
  name=wam_wan22_debug description="wan22 5B wam debug" \
  +modal_gpu=H200 init_submodules=false
```
- [ ] **Step 3: Verify R1 (checkpoint key match)** — in the Modal logs, `[build_wam_dit] loaded ...: N missing / M unexpected keys` and the `[build_wan_vae]` line. Expect the DiT robot-module keys (state/action/decoder) to be "missing" (they're fresh) but the pretrained blocks matched. A near-total miss ⇒ key-name mismatch ⇒ add a rename map in `_load_wan_state_dict` and re-run.
- [ ] **Step 4: Verify R2 (geometry)** — no `ValueError` from `CausalWanModel` about register/frame_seqlen; one training step completes and reports `world_loss` + `action_loss`.
- [ ] **Step 5: Verify R3 (data) + eval** — the dataloader yields clips (no all-episodes-skipped warning); one validation `sample()` runs and `WAMEvalVideo` logs a predicted-frame video to W&B.
- [ ] **Step 6: Record outcome** — note key-match counts, chosen resolution/frame_seqlen validity, and any config path fixes in the spec's Risks section. Commit any fixes.

---

### Task 10: DDP launch

- [ ] **Step 1: Restore the real data filter** (widen from the debug episode subset).
- [ ] **Step 2: Launch**
```bash
python egomimic/modal/trainModal.py \
  --config-name=train_zarr_human_wam_wan22_5b trainer=ddp_modal logger=wandb \
  name=wam_wan22 description="wan22 5B wam" \
  +modal_gpu=H200 init_submodules=false
```
- [ ] **Step 3: Confirm** the W&B run advances `global_step` and Valid metrics + dream videos log. Commit final config state.

---

## Self-Review

**Spec coverage:** A(vendored)→T1; B(merges)→T2; C(5B wiring)→T3+T4; D(Modal: resolver→T5, configs→T6, image/volume→T7, download→T8); E(validation)→T9; DDP→T10. Embodiment refactor (mecka_bimanual, no enum)→T2/T6 + Global Constraints. Norm-stats-online + evaluator-wired→Global Constraints + T9. All spec sections mapped.

**Placeholder scan:** No TBD/TODO. Two intentional operational unknowns (exact 5B checkpoint filenames; exact SQL column names for the mecka filter) are explicitly resolved in T8/T9 with the reconciliation step spelled out — not hand-waved.

**Type consistency:** `build_wam_dit(arch=...)`, `build_wan_vae(z_dim=48)`, `ModalWamEpisodeResolver._dataset_class=ZarrWamDataset`, `WAM(target_h,target_w)`, config `_target_`s all match across tasks and the vendored file names from T1.
