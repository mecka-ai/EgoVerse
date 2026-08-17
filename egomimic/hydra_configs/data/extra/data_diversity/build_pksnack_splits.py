"""Build the packaging_snacks splits for the two `pksnack_*` experiment families.

Ingested pool = episodes in the union of D{1..5}.json (what actually exists on the
zarr volume) restricted to task == "packaging_snacks".

Outputs (all under this directory, prefix `pksnack_`):
  pksnack_val_ophold5.json        5 eps of ONE fully held-out operator (shared val,
                                  both families). The operator is excluded from
                                  EVERY train set.
  pksnack_all_train.json          model-size family train = pool - ophold operator
                                  - opscale_val_indom8 (see NOTE below).
  pksnack_opscale_val_indom8.json 8 eps, one per L1 operator (median length),
                                  removed from the pool BEFORE any allocation.
  pksnack_opscale_L{1..4}_train.json
                                  operator-scaling levels: ops ranked by total
                                  frames desc, L1=top8 L2=top24 L3=top72 L4=top160
                                  (strict superset chain), round-robin episode
                                  allocation to a fixed 1,080,000-frame (10 h) budget.
  pksnack_trainviz4.json          4 trained-on (in-distribution) eps present in ALL
                                  FOUR level train sets (hence also in all_train),
                                  one per distinct operator.

NOTE: the model-size train set also excludes the 8 in-domain-operator val episodes
(0.4% of the pool) so that BOTH val sets are clean for BOTH families — this keeps the
global invariant "every train set is disjoint from every val set".

Run:  python egomimic/hydra_configs/data/extra/data_diversity/build_pksnack_splits.py
"""

import json
import sys
from collections import deque

REPO = "/Users/anikethcheluva/Documents/CS Projects/EgoVerse"
DD = f"{REPO}/egomimic/hydra_configs/data/extra/data_diversity"
sys.path.insert(0, REPO)

import pandas as pd
from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine

TASK = "packaging_snacks"
FPS = 30.0
BUDGET = 1_080_000  # 10 h @ 30 fps
LEVELS = {1: 8, 2: 24, 3: 72, 4: 160}
N_OPHOLD_VAL = 5
N_TRAINVIZ = 4


def hours(frames):
    return frames / FPS / 3600.0


# ---- ingested pool ----------------------------------------------------------
pool_ids = set()
for i in range(1, 6):
    pool_ids |= set(json.load(open(f"{DD}/D{i}.json")))
print(f"ingested pool (D1-D5 union): {len(pool_ids)} eps")

engine = create_default_engine()
with engine.connect() as conn:
    df = pd.read_sql(
        text(
            "SELECT episode_hash, operator, task, num_frames FROM app.episodes "
            "WHERE episode_hash = ANY(:hashes)"
        ),
        conn,
        params={"hashes": sorted(pool_ids)},
    )
assert len(df) == len(pool_ids), f"SQL rows {len(df)} != pool {len(pool_ids)}"

pk = df[df.task == TASK].copy()
assert pk.episode_hash.is_unique
assert pk.num_frames.notna().all() and (pk.num_frames > 0).all()
POOL = set(pk.episode_hash)
frames = dict(zip(pk.episode_hash, pk.num_frames.astype(int)))
op_of = dict(zip(pk.episode_hash, pk.operator))
print(
    f"{TASK} n pool: {len(pk)} eps  {int(pk.num_frames.sum()):,} frames  "
    f"{hours(pk.num_frames.sum()):.2f} h  {pk.operator.nunique()} operators"
)

by_op_all = {op: sorted(g.episode_hash) for op, g in pk.groupby("operator")}
tot_all = pk.groupby("operator").num_frames.sum()
for n in (8, 24, 72, 160):
    top = sorted(tot_all.index, key=lambda o: (-int(tot_all[o]), o))[:n]
    f = int(tot_all[top].sum())
    print(f"  top-{n:<3} operators: {f:,} frames  {hours(f):.2f} h")

# ---- held-out operator (shared val for both families) -----------------------
# deterministic: among operators with >= N_OPHOLD_VAL episodes, the SMALLEST total
# footprint (tie-break operator id) so holding it out costs the least train data.
cand = [o for o in by_op_all if len(by_op_all[o]) >= N_OPHOLD_VAL]
oph_op = sorted(cand, key=lambda o: (int(tot_all[o]), o))[0]
oph_eps = sorted(by_op_all[oph_op], key=lambda h: (frames[h], h))[:N_OPHOLD_VAL]
oph_set = set(oph_eps)
print(
    f"\nheld-out operator: {oph_op}  total_eps={len(by_op_all[oph_op])} "
    f"total_frames={int(tot_all[oph_op]):,} ({hours(tot_all[oph_op]):.3f} h)"
)
print(
    f"  val_ophold5: {len(oph_eps)} eps  {sum(frames[h] for h in oph_eps):,} frames "
    f"({hours(sum(frames[h] for h in oph_eps)):.3f} h)"
)

# base pool: the held-out OPERATOR is removed entirely (not just its 5 val eps)
BASE = {h for h in POOL if op_of[h] != oph_op}
base_df = pk[pk.episode_hash.isin(BASE)]
print(
    f"base pool (pool - operator {oph_op}): {len(BASE)} eps  "
    f"{int(base_df.num_frames.sum()):,} frames  {hours(base_df.num_frames.sum()):.2f} h  "
    f"{base_df.operator.nunique()} operators"
)

by_op = {op: sorted(g.episode_hash) for op, g in base_df.groupby("operator")}
op_totals = base_df.groupby("operator").num_frames.sum()
op_rank = sorted(op_totals.index, key=lambda o: (-int(op_totals[o]), o))

# ---- in-domain-operator val: median-length ep of each L1 operator ------------
l1_ops = op_rank[: LEVELS[1]]
val8 = []
for op in l1_ops:
    eps = sorted(by_op[op], key=lambda h: (frames[h], h))
    val8.append(eps[(len(eps) - 1) // 2])  # lower median, deterministic
assert len(set(val8)) == LEVELS[1]
val8_set = set(val8)
print(
    f"\nopscale_val_indom8: {len(val8)} eps  {sum(frames[h] for h in val8):,} frames "
    f"({hours(sum(frames[h] for h in val8)):.3f} h)  ops = L1 top-8"
)

# every train set is built from the pool with BOTH val sets already removed
avail_by_op = {op: [h for h in hs if h not in val8_set] for op, hs in by_op.items()}
ALL_TRAIN = sorted(BASE - val8_set)
at_frames = sum(frames[h] for h in ALL_TRAIN)
print(
    f"all_train (model-size family): {len(ALL_TRAIN)} eps  {at_frames:,} frames  "
    f"{hours(at_frames):.2f} h  {len({op_of[h] for h in ALL_TRAIN})} operators"
)


# ---- per-level round-robin train sets ---------------------------------------
def build_level(ops):
    queues = {op: deque(avail_by_op[op]) for op in ops}
    total, chosen, contributed = 0, [], {op: 0 for op in ops}
    tolerance_used = None
    done = False
    while not done:
        progressed = False
        for op in ops:
            if not queues[op]:
                continue
            h = queues[op][0]
            f = frames[h]
            if total + f > BUDGET:
                if contributed[op] == 0 and tolerance_used is None:
                    tolerance_used = (op, h, f)  # coverage tolerance: +-1 episode
                else:
                    done = True
                    break
            queues[op].popleft()
            chosen.append(h)
            total += f
            contributed[op] += 1
            progressed = True
        if not progressed:
            break  # all queues exhausted
    return chosen, total, contributed, tolerance_used


results = {}
for lvl, n_ops in LEVELS.items():
    ops = op_rank[:n_ops]
    assert len(ops) == n_ops, f"L{lvl}: only {len(ops)} operators available"
    chosen, total, contributed, tol = build_level(ops)
    results[lvl] = (ops, chosen, total, contributed, tol)
    per_op = [contributed[o] for o in ops]
    print(
        f"L{lvl}: ops={n_ops}  eps={len(chosen)}  frames={total:,} ({hours(total):.2f} h, "
        f"{100 * total / BUDGET:.2f}% of budget)  eps/op min={min(per_op)} max={max(per_op)}"
        + (f"  TOLERANCE: {tol}" if tol else "")
    )

# ---- train-viz set: trained-on eps present in ALL FOUR levels ---------------
sets = {lvl: set(results[lvl][1]) for lvl in LEVELS}
inter = sets[1] & sets[2] & sets[3] & sets[4]
print(f"\nepisodes in ALL FOUR level train sets: {len(inter)}")
# one episode per distinct operator, operators in rank order, median-length pick
tv, used_ops = [], set()
for op in op_rank:
    if len(tv) == N_TRAINVIZ:
        break
    if op in used_ops:
        continue
    cands = sorted([h for h in inter if op_of[h] == op], key=lambda h: (frames[h], h))
    if not cands:
        continue
    tv.append(cands[(len(cands) - 1) // 2])
    used_ops.add(op)
assert len(tv) == N_TRAINVIZ, f"only found {len(tv)} train-viz candidates"
print(
    f"trainviz4: {len(tv)} eps  {sum(frames[h] for h in tv):,} frames "
    f"({hours(sum(frames[h] for h in tv)):.3f} h)  ops={len({op_of[h] for h in tv})}"
)
for h in tv:
    print(f"  {h}  op={op_of[h]}  frames={frames[h]:,}")

# ---- sanity checks -----------------------------------------------------------
VALS = {"ophold5": oph_set, "indom8": val8_set}
TRAINS = {"all_train": set(ALL_TRAIN), **{f"L{l}": sets[l] for l in LEVELS}}

assert not oph_set & val8_set, "the two val sets overlap"
assert {op_of[h] for h in val8_set} == set(l1_ops)
assert {op_of[h] for h in oph_set} == {oph_op}

prev_ops = None
for lvl in (1, 2, 3, 4):
    ops, chosen, total, contributed, tol = results[lvl]
    ops_set = set(ops)
    assert len(set(chosen)) == len(chosen), f"L{lvl}: dup episodes"
    if prev_ops is not None:
        assert prev_ops < ops_set, f"L{lvl - 1} ops not a STRICT subset of L{lvl}"
    prev_ops = ops_set
    assert {op_of[h] for h in chosen} == ops_set, f"L{lvl}: operator set mismatch"
    assert all(contributed[o] >= 1 for o in ops), f"L{lvl}: some op contributed 0 eps"
    if tol is None:
        assert total <= BUDGET, f"L{lvl} over budget w/o tolerance"
    else:
        assert total - tol[2] <= BUDGET, f"L{lvl} over budget beyond tolerance"
    assert total >= BUDGET * 0.99 or tol is not None, f"L{lvl} >1% under budget: {total}"

for tname, tset in TRAINS.items():
    assert tset <= POOL, f"{tname} escaped the ingested pool"
    for vname, vset in VALS.items():
        assert not tset & vset, f"{tname} overlaps val set {vname}"
    assert oph_op not in {op_of[h] for h in tset}, f"held-out operator present in {tname}"
    assert set(tv) <= tset, f"train-viz episodes missing from {tname}"

print("\nALL SANITY CHECKS PASSED")


# ---- write jsons --------------------------------------------------------------
def write_json(path, hashes):
    with open(path, "w") as f:
        json.dump(sorted(hashes), f, indent=0)
        f.write("\n")
    fr = sum(frames[h] for h in hashes)
    print(f"wrote {path}  ({len(hashes)} eps, {fr:,} frames, {hours(fr):.3f} h)")


write_json(f"{DD}/pksnack_val_ophold5.json", oph_eps)
write_json(f"{DD}/pksnack_opscale_val_indom8.json", val8)
write_json(f"{DD}/pksnack_trainviz4.json", tv)
write_json(f"{DD}/pksnack_all_train.json", ALL_TRAIN)
for lvl in (1, 2, 3, 4):
    write_json(f"{DD}/pksnack_opscale_L{lvl}_train.json", results[lvl][1])

# ---- report table -------------------------------------------------------------
print("\n== SPLIT TABLE ==")
print(f"{'split':<26} {'eps':>6} {'frames':>12} {'hours':>7} {'ops':>5}")


def row(name, hs):
    fr = sum(frames[h] for h in hs)
    print(
        f"{name:<26} {len(hs):>6} {fr:>12,} {hours(fr):>7.2f} "
        f"{len({op_of[h] for h in hs}):>5}"
    )


row("val_ophold5 (idx: Valid_oph)", oph_eps)
row("opscale_val_indom8 (Valid)", val8)
row("trainviz4 (Valid_trainviz)", tv)
row("all_train (model-size)", ALL_TRAIN)
for lvl in (1, 2, 3, 4):
    row(f"opscale_L{lvl}_train", results[lvl][1])
