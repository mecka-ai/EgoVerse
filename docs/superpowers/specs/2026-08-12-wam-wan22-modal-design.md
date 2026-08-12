# WAM (Wan2.2-TI2V-5B) on the EgoVerse Modal pipeline — Design

**Date:** 2026-08-12
**Branch:** `aniketh/wam-wan22` (worktree off `aniketh/extra-features`)
**Author:** Aniketh + Claude

## Goal

Make Wan2.2 WAM (World-Action Model) training launchable through this fork's Modal
pipeline:

```bash
python egomimic/modal/trainModal.py \
  model=wam_bc_human_wan22_5b data=mecka_wam trainer=ddp_modal logger=wandb \
  name=<run> description=<desc> +modal_gpu=H200 init_submodules=false
```

## Background (what already exists, and where)

The WAM code lives on the RL2 upstream branch `upstream/aniketh/wam` as a single commit
`82f3c207` ("added dreamzero support into egoverse", +13.7K lines). It is a **dreamzero
port**: one joint DiT forward over `[video latent | action | state]` predicts the video
flow-matching velocity and the action flow-matching velocity (rectified flow). The
pretrained video DiT is loaded + frozen (optionally LoRA-adapted); the state/action/decoder
register modules train.

Key fact discovered during investigation:

- **The runnable path is Wan2.1-1.3B only.** `egomimic/algo/wam.py` (`WAMModel`) builds a
  `CausalWanModel` via `build_wam_dit` + a Wan VAE via `build_wan_vae`
  (`egomimic/models/wam_nets.py`). Both builders are hardcoded to the **1.3B Wan2.1 T2V**
  preset (`WAM_DIT_1_3B`, `WanVideoVAE(z_dim=16)`).
- **The Wan2.2 5B logic exists but is unwired.** All the 2.2 auto-selection lives in
  `egomimic/models/wan/wan_flow_matching_action_tf.py` (`WANPolicyHead`), which is
  **reference code — never imported at runtime**. It reads back `self.vae.z_dim == 48` /
  `self.model.in_dim == 48` to pick `Wan2.2_VAE.pth` + the `Wan-AI/Wan2.2-TI2V-5B` DiT and
  to resize video for the 5B latent geometry.
- The vendored building blocks needed for 2.2 **are** present:
  `WanVideoVAE38(z_dim=48, dim=160)` (16× spatial VAE) in `wan_video_vae.py`, and
  `CausalWanModel` accepts arbitrary `dim/ffn_dim/num_heads/num_layers/in_dim/out_dim`
  (RoPE/patchify fully parametric; head_dim 128 for both 1.3B and 5B).

So "launch Wan2.2 WAM" = **port the four 2.2 pieces from `WANPolicyHead` into the live
`wam_nets.py`/`wam.py`/config path**, then wire the whole thing into this fork's Modal
launcher and data resolver.

## This fork vs. the WAM branch (integration deltas)

The WAM branch was cut from `upstream/main` (merge-base `21647be0`); this fork
(`aniketh/extra-features`) is 295 commits ahead of `origin/main` with a diverged file
layout and its own Modal pipeline. Concrete deltas:

| Area | WAM branch | This fork | Action |
|------|-----------|-----------|--------|
| Modal launcher | none (upstream has no Modal) | `modal_setup.py` + `trainModal.py` | wire WAM into fork's launcher |
| Data resolver | `S3WamEpisodeResolver` / `LocalWamEpisodeResolver` | `ModalEpisodeResolver` over `/mnt/zarr-data` (SQL) | add `ModalWamEpisodeResolver` |
| Embodiment | added universal `HUMAN_BIMANUAL=15` | separate `MECKA_BIMANUAL=9` + alias `HUMAN_BIMANUAL→MECKA_BIMANUAL` | **do NOT add enum member**; WAM configs use `mecka_bimanual` |
| `algo/__init__.py` | `+from ...wam import WAM` (no OAT/quest) | has OAT/quest guards | add the one WAM import line only |
| deps | `diffusers/safetensors/accelerate/peft` | image has `safetensors` only | add `diffusers/accelerate/peft` |

### Embodiment refactor (per user directive)

The WAM branch came from a refactor that treated `human_bimanual` as a **universal** human
embodiment. This fork does not: it keeps `MECKA_BIMANUAL`, `ARIA_BIMANUAL`, `EVA_BIMANUAL`
distinct and already ships `_EMBODIMENT_ALIASES = {"HUMAN_BIMANUAL": "MECKA_BIMANUAL", ...}`
(applied inside `get_embodiment_id`). Investigation confirmed the string `human_bimanual`
appears in the branch's **code** only in the `embodiment.py` enum addition; the WAM
keymap/transforms/algo are embodiment-agnostic (they key off zarr fields
`images.front_1`, `right.obs_ee_pose`, `left.obs_ee_pose`, `obs_head_pose` and emit
`actions_cartesian`).

**Refactor decision:** WAM adopts this fork's convention — `mecka_bimanual` as the dataset
key / `domains` / `ac_keys`. Do **not** introduce `HUMAN_BIMANUAL=15`. The existing alias
remains the safety net for any DB rows literally labeled `human_bimanual` (e.g. ELMO aria).

## Locked decisions

1. **Checkpoints:** dedicated `wan-checkpoints` Modal volume, fetched once. An `ensure_file`
   helper HF-downloads `Wan-AI/Wan2.2-TI2V-5B` (Apache-2.0, ungated) into the volume on
   first use and reuses it thereafter; mounted read-only in training. No local download.
2. **Embodiment:** refactor WAM to `mecka_bimanual`; no new enum member.
3. **GPU:** default `+modal_gpu=H200` (scale to `H200:N` for DDP).
4. **Norm stats:** WAM's embodiment isn't covered by the `mecka_all_zarr` precomputed set and
   its bounds calibration is disabled → compute norm stats **online**; do **not** pass
   `norm_stats.precomputed_norm_path`. Evaluator stays wired (`eval_wam`) so Valid/W&B
   logging is active (an unset evaluator silently disables all validation logging).

## Work breakdown

### A. Drop-in vendored code (new files, no conflicts)
- `egomimic/models/wan/*` (16 files)
- `egomimic/models/wam_nets.py`
- `egomimic/algo/wam.py`
- `egomimic/eval/eval_wam.py`
- `egomimic/rldb/zarr/zarr_dataset_wam.py`
- `egomimic/hydra_configs/evaluator/eval_wam.yaml`, `evaluator/viz/wam_cartesian.yaml`
- `.ruff.toml` change (vendored-code lint carve-outs) — apply only the WAM-relevant hunk.

### B. Targeted additive merges
- `egomimic/rldb/embodiment/embodiment.py`: **no change** (alias already present; do not add
  the enum member).
- `egomimic/rldb/embodiment/human.py`: add `Mecka.get_wam_keymap` + `Mecka.get_wam_transform_list`
  (verbatim from the branch; embodiment-agnostic).
- `egomimic/algo/__init__.py`: add `from egomimic.algo.wam import WAM as WAM`.
- `pyproject.toml`: add `diffusers>=0.38.0`, `accelerate>=1.14.0`, `peft>=0.19.1`
  (`safetensors` already present).

### C. Wan2.2 5B wiring (the actual new engineering)
- `egomimic/models/wam_nets.py`:
  - Add `WAM_DIT_5B` preset: `dim=3072, ffn_dim=14336, num_heads=24, num_layers=30,
    in_dim=48, out_dim=48, model_type="ti2v", concat_first_frame_latent=False`, plus the
    shared `patch_size/text_len/freq_dim/text_dim/qk_norm/cross_attn_norm/eps/hidden_size`.
  - `build_wam_dit`: add an `arch`/`preset` selector (`"wan21_1_3b"` default, `"wan22_5b"`);
    select the preset dict accordingly; **load sharded safetensors** when the checkpoint is
    a `*.safetensors.index.json` (the 5B DiT ships sharded) — iterate shards, merge, then
    `from_civitai` + `load_state_dict(strict=False)`.
  - `build_wan_vae`: build `WanVideoVAE38(z_dim=48, dim=160)` when `z_dim == 48`, else the
    existing `WanVideoVAE(z_dim=z_dim)`. Import `WanVideoVAE38`.
  - Add an `ensure_file(local_path, filename, repo_id)` HF-fetch helper (ported from
    `wan_flow_matching_action_tf.py`) so builders can fetch-if-missing into the volume cache.
- `egomimic/algo/wam.py`: `WAMModel._frame_to_video` gains `target_h`/`target_w` (default to
  the existing square `frame_size` so the 1.3B path is byte-for-byte unchanged); 5B config
  passes `target_h=160, target_w=320`. `WAMModel.__init__` / `WAM.__init__` thread the two
  params through.
- New `egomimic/hydra_configs/model/wam_bc_human_wan22_5b.yaml`: `domains: [mecka_bimanual]`,
  `ac_keys: {mecka_bimanual: actions_cartesian}`, `dit` → `build_wam_dit(arch=wan22_5b,
  in_dim/out_dim implied by preset, num_action_per_block/num_state_per_block, frame_seqlen=50,
  freeze_video/lora)`, `vae` → `build_wan_vae(z_dim=48, checkpoint_path=<vol>/Wan2.2_VAE.pth)`,
  `frame_size`→`target_h=160/target_w=320`, checkpoint paths pointing at the volume mount.

### D. Modal integration
- `egomimic/modal/modal_setup.py`:
  - Image pip layer: add `diffusers`, `accelerate`, `peft`.
  - Declare `wan_checkpoints_volume = modal.Volume.from_name("wan-checkpoints",
    create_if_missing=True)`; mount it read-only at e.g. `/mnt/wan-ckpts` in `run_hydra_train`
    (extend `_build_volumes`).
  - Add a one-time `download_wan22_weights` `@app.function` (or local entrypoint) that calls
    the `ensure_file` helper to populate the volume, so the first training run doesn't pay
    the ~10GB download inside the DDP job.
- `egomimic/rldb/zarr/zarr_dataset_wam.py`: add `ModalWamEpisodeResolver(ModalEpisodeResolver)`
  with `_dataset_class = ZarrWamDataset`, alongside the existing `S3WamEpisodeResolver` /
  `LocalWamEpisodeResolver` (import `ModalEpisodeResolver` from `zarr_dataset_multi`; exact
  precedent for the pattern: `LocalAnnotationCutoffEpisodeResolver`). Use `WamMultiDataset`
  (skips quantile bounds rejection) as the dataset wrapper. No edit to `zarr_dataset_multi.py`
  is required beyond it already exporting `ModalEpisodeResolver`.
- New `egomimic/hydra_configs/data/mecka_wam.yaml` (Modal variant): `WamMultiDataset._from_resolver`
  over `ModalWamEpisodeResolver(folder_path=/mnt/zarr-data)`, `Mecka.get_wam_keymap` (cam=17,
  action=16, state=4), `Mecka.get_wam_transform_list`, dataset key `mecka_bimanual`, a small
  starter DB filter, train/valid split.

### E. Validation (before any DDP run)
Run `trainer=debug` on Modal against a **tiny** episode filter to shake out:
1. **Checkpoint key match** — print `build_wam_dit`/`build_wan_vae` missing/unexpected counts;
   a near-total miss means the 5B checkpoint key names don't match `CausalWanModel` and a
   rename map is required (the state-dict converter has no 5B hash entry).
2. **Resolution ↔ `frame_seqlen` invariant** — `CausalWanModel` has a runtime `ValueError`
   guard tying latent frame_seqlen to register length; vendored comments disagree
   (320×176 vs 160×320). Confirm 160×320 → latent 10×20 → DiT patch → 5×10 = 50 tokens/frame
   and that `num_image_blocks == num_action_blocks == num_state_blocks`.
3. One training step + one validation `sample()` + one `WAMEvalVideo` frame render succeed.

Only after green: relaunch `trainer=ddp_modal` with `+modal_gpu=H200 init_submodules=false`.

## Risks / unknowns

- **R1 (high): 5B checkpoint key naming.** No converter hash entry for TI2V-5B → weights pass
  through unchanged and load `strict=False`. If names don't match `CausalWanModel`, the model
  silently trains from ~random. Mitigation: assert on missing/unexpected key counts in the
  debug run; add a rename map if needed.
- **R2 (medium): resolution/register invariant.** Wrong `frame_seqlen`/resolution trips the
  DiT's `ValueError` or misaligns action/frame blocks. Mitigation: derive from the VAE 16×
  factor + DiT patch 2, validate empirically in debug.
- **R3 (low): data schema.** WAM needs `images.front_1` clips + `obs_head_pose` +
  `{left,right}.obs_ee_pose` on the mounted episodes. Mecka zarrs on `mecka_data_v2` should
  have these (the cartesian pipeline uses the same fields); confirm in debug.
- **R4 (low): H200 availability/memory.** 5B frozen (~10GB bf16) + VAE + LoRA/robot-module
  grads + video-token activations at small batch should fit one H200; confirm at debug.

## Files touched (summary)
New: `models/wan/*`, `models/wam_nets.py`, `algo/wam.py`, `eval/eval_wam.py`,
`rldb/zarr/zarr_dataset_wam.py`, model config `wam_bc_human_wan22_5b.yaml`, data config
`mecka_wam.yaml`, evaluator configs `eval_wam.yaml` + `viz/wam_cartesian.yaml`.
Edited: `algo/__init__.py`, `rldb/embodiment/human.py`, `pyproject.toml`, `.ruff.toml`,
`modal/modal_setup.py`. (`rldb/zarr/zarr_dataset_wam.py` is a new file and carries
`ModalWamEpisodeResolver`, so `zarr_dataset_multi.py` needs no edit.)
Explicitly NOT edited: `rldb/embodiment/embodiment.py` (alias already present).

## Non-goals
- Keeping/porting the Wan2.1 1.3B path runnable is a free side effect (its builder stays the
  default), but it is not a target and won't be separately validated.
- Text-conditioned generation (T5 encoder) — vendored but out of scope.
