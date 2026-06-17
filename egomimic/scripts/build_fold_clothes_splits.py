"""Build eps_to_use JSON splits + data configs for the folding_clothes
scored-episode HPT experiment suite.

Inputs:
  /tmp/fc_scores_by_task.json  -- scores_by_task.json fetched from the Modal
                                  volume egoverse-training-outputs at
                                  deminf_fold_clothes/resnet_32d_2026-06-16_18-28-43/

Produces (in egomimic/hydra_configs/data/extra/ and egomimic/hydra_configs/data/):
  Set A (val = 5 median-score episodes, removed from pool first):
    folding_clothes_top50_train.json / bot50 / rand50 / all_midval
    folding_clothes_mid5_val.json
  Set B (val = held-out operator, ~5 episodes, excluded from pool):
    folding_clothes_op_all_train.json / op_top40 / op_bot40
    folding_clothes_op_val.json
  Per-run in-domain train_viz JSONs (3 evenly-spaced episodes from each train set).
  7 data config YAMLs (mecka_fc_*_zarr.yaml).

Run:  python egomimic/scripts/build_fold_clothes_splits.py
"""

import json
import math
import os
import random
from pathlib import Path

from sqlalchemy import URL, create_engine
from egomimic.utils.aws.aws_sql import episode_table_to_df, load_env

TASK = "folding_clothes"
SCORES = Path("/tmp/fc_scores_by_task.json")
REPO = Path(__file__).resolve().parents[2]
EXTRA = REPO / "egomimic/hydra_configs/data/extra"
DATA = REPO / "egomimic/hydra_configs/data"
RNG_SEED = 42
N_VIZ = 3


def engine_psycopg2():
    """Build engine using the locally-installed psycopg2 (psycopg v3 absent)."""
    load_env()
    url = os.environ.get("DATABASE_URL")
    if url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1).replace(
            "postgres://", "postgresql+psycopg2://", 1
        )
        return create_engine(url, pool_pre_ping=True)
    return create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=os.environ["PG_USER"],
            password=os.environ["PG_PASSWORD"],
            host=os.environ["PG_HOST"],
            port=int(os.environ.get("PG_PORT", "5432")),
            database=os.environ.get("PG_DATABASE", "defaultdb"),
            query={"sslmode": "require"},
        ),
        pool_pre_ping=True,
    )


def write_json(name, hashes):
    path = EXTRA / name
    path.write_text(json.dumps(list(hashes), indent=2))
    print(f"  wrote {name:42s} {len(hashes):4d} eps")
    return path


def viz_slice(train_hashes):
    """3 evenly-spaced in-domain episodes from a train list."""
    n = len(train_hashes)
    if n <= N_VIZ:
        return list(train_hashes)
    idx = [round(i * (n - 1) / (N_VIZ - 1)) for i in range(N_VIZ)]
    return [train_hashes[i] for i in idx]


DATA_CFG_TEMPLATE = """_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

# {header}
# train     -> {train_json}
# valid     -> {val_json}
# train_viz -> {viz_json}
# val n train = empty; train_viz subset of train.

train_datasets:
  mecka_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.ModalEpisodeResolver
      folder_path: /mnt/zarr-data
      key_map:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_keymap
        mode: cartesian
      transform_list:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_transform_list
        mode: cartesian
      eps_to_use: egomimic/hydra_configs/data/extra/{train_json}
    mode: total

valid_datasets:
  mecka_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.ModalEpisodeResolver
      folder_path: /mnt/zarr-data
      key_map:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_keymap
        mode: cartesian
      transform_list:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_transform_list
        mode: cartesian
      eps_to_use: egomimic/hydra_configs/data/extra/{val_json}
    mode: total

train_viz_datasets:
  mecka_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
    resolver:
      _target_: egomimic.rldb.zarr.zarr_dataset_multi.ModalEpisodeResolver
      folder_path: /mnt/zarr-data
      key_map:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_keymap
        mode: cartesian
      transform_list:
        _target_: egomimic.rldb.embodiment.human.Mecka.get_transform_list
        mode: cartesian
      eps_to_use: egomimic/hydra_configs/data/extra/{viz_json}
    mode: total

train_dataloader_params:
  mecka_bimanual:
    batch_size: 64
    num_workers: 12
    persistent_workers: true

valid_dataloader_params:
  mecka_bimanual:
    batch_size: 64
    num_workers: 12
    persistent_workers: true

train_viz_dataloader_params:
  mecka_bimanual:
    batch_size: 8
    num_workers: 4
    persistent_workers: true
"""


def write_cfg(cfg_name, header, train_json, val_json, viz_json):
    path = DATA / cfg_name
    path.write_text(
        DATA_CFG_TEMPLATE.format(
            header=header,
            train_json=train_json,
            val_json=val_json,
            viz_json=viz_json,
        )
    )
    print(f"  wrote {cfg_name}")


def main():
    sbt = json.loads(SCORES.read_text())
    fc = sbt[TASK]
    finite = [(h, s) for h, s in fc.items() if s is not None and math.isfinite(s)]
    S = [h for h, _ in sorted(finite, key=lambda x: -x[1])]  # desc by score
    N = len(S)
    print(f"folding_clothes finite-scored episodes: N={N}")

    # ---- operator map ----
    df = episode_table_to_df(engine_psycopg2())
    fcset = set(S)
    sub = df[df["episode_hash"].isin(fcset)]
    op_of = dict(zip(sub["episode_hash"], sub["operator"]))
    missing = [h for h in S if h not in op_of]
    if missing:
        print(f"WARNING: {len(missing)} hashes not found in DB (no operator)")

    # ================= SET A =================
    mid = N // 2
    val_A = S[mid - 2 : mid + 3]
    assert len(val_A) == 5, val_A
    pool_A = [h for h in S if h not in set(val_A)]  # preserves score order
    M = len(pool_A)
    top50 = pool_A[: M // 2]
    bot50 = pool_A[M // 2 :]
    rand50 = random.Random(RNG_SEED).sample(pool_A, M // 2)
    all_A = pool_A

    print(f"\n=== SET A (val = 5 middle, M={M}) ===")
    write_json("folding_clothes_mid5_val.json", val_A)
    write_json("folding_clothes_top50_train.json", top50)
    write_json("folding_clothes_bot50_train.json", bot50)
    write_json("folding_clothes_rand50_train.json", rand50)
    write_json("folding_clothes_all_midval_train.json", all_A)
    write_json("folding_clothes_top50_viz.json", viz_slice(top50))
    write_json("folding_clothes_bot50_viz.json", viz_slice(bot50))
    write_json("folding_clothes_rand50_viz.json", viz_slice(rand50))
    write_json("folding_clothes_all_midval_viz.json", viz_slice(all_A))

    # asserts
    vset = set(val_A)
    for nm, tr in [("top50", top50), ("bot50", bot50), ("rand50", rand50), ("all_A", all_A)]:
        assert not (set(tr) & vset), f"{nm} overlaps val_A"
    assert set(top50) | set(bot50) == set(pool_A) and not (set(top50) & set(bot50))
    assert len(rand50) == M // 2

    # ================= SET B =================
    # operator counts among scored folding_clothes
    from collections import Counter
    counts = Counter(op_of[h] for h in S if h in op_of)
    # pick operator closest to 5
    best_op = min(counts, key=lambda o: (abs(counts[o] - 5), o))
    val_B = [h for h in S if op_of.get(h) == best_op]
    pool_B = [h for h in S if op_of.get(h) != best_op]
    K = len(pool_B)
    k40 = math.ceil(0.4 * K)
    top40 = pool_B[:k40]
    bot40 = pool_B[-k40:]
    all_B = pool_B

    print(f"\n=== operator counts (top 12) ===")
    for op, c in counts.most_common(12):
        print(f"  {op!r}: {c}")
    print(f"chosen held-out operator: {best_op!r} ({counts[best_op]} eps)")

    print(f"\n=== SET B (val = operator {best_op!r}, K={K}, k40={k40}) ===")
    write_json("folding_clothes_op_val.json", val_B)
    write_json("folding_clothes_op_all_train.json", all_B)
    write_json("folding_clothes_op_top40_train.json", top40)
    write_json("folding_clothes_op_bot40_train.json", bot40)
    write_json("folding_clothes_op_all_viz.json", viz_slice(all_B))
    write_json("folding_clothes_op_top40_viz.json", viz_slice(top40))
    write_json("folding_clothes_op_bot40_viz.json", viz_slice(bot40))

    bset = set(val_B)
    for nm, tr in [("all_B", all_B), ("top40", top40), ("bot40", bot40)]:
        assert not (set(tr) & bset), f"{nm} overlaps val_B (operator leak!)"

    # ================= DATA CONFIGS =================
    print(f"\n=== DATA CONFIGS ===")
    write_cfg("mecka_fc_score_top50_zarr.yaml", "folding_clothes TOP 50% by score, val=5 middle",
              "folding_clothes_top50_train.json", "folding_clothes_mid5_val.json", "folding_clothes_top50_viz.json")
    write_cfg("mecka_fc_score_bot50_zarr.yaml", "folding_clothes BOTTOM 50% by score, val=5 middle",
              "folding_clothes_bot50_train.json", "folding_clothes_mid5_val.json", "folding_clothes_bot50_viz.json")
    write_cfg("mecka_fc_rand50_zarr.yaml", "folding_clothes RANDOM 50% (seed 42), val=5 middle",
              "folding_clothes_rand50_train.json", "folding_clothes_mid5_val.json", "folding_clothes_rand50_viz.json")
    write_cfg("mecka_fc_all_midval_zarr.yaml", "folding_clothes ALL (minus 5 middle), val=5 middle",
              "folding_clothes_all_midval_train.json", "folding_clothes_mid5_val.json", "folding_clothes_all_midval_viz.json")
    write_cfg("mecka_fc_op_all_zarr.yaml", f"folding_clothes ALL (minus operator {best_op}), val=operator",
              "folding_clothes_op_all_train.json", "folding_clothes_op_val.json", "folding_clothes_op_all_viz.json")
    write_cfg("mecka_fc_op_top40_zarr.yaml", f"folding_clothes TOP 40% (operator {best_op} held out), val=operator",
              "folding_clothes_op_top40_train.json", "folding_clothes_op_val.json", "folding_clothes_op_top40_viz.json")
    write_cfg("mecka_fc_op_bot40_zarr.yaml", f"folding_clothes BOTTOM 40% (operator {best_op} held out), val=operator",
              "folding_clothes_op_bot40_train.json", "folding_clothes_op_val.json", "folding_clothes_op_bot40_viz.json")

    print("\nALL ASSERTS PASSED. Summary:")
    print(f"  Set A: top50={len(top50)} bot50={len(bot50)} rand50={len(rand50)} all={len(all_A)} val={len(val_A)}")
    print(f"  Set B: all={len(all_B)} top40={len(top40)} bot40={len(bot40)} val={len(val_B)} (op={best_op})")


if __name__ == "__main__":
    main()
