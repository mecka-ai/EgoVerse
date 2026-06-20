"""Build a random allowlist of mecka_data_v2 episodes summing to ~250h total, so
the default 80/20 (by-episode, seed 42) MultiDataset split yields ~200h train.

- intersect SQL (num_frames, non-deleted, mecka_bimanual) with episodes actually
  present on the mecka_data_v2 volume (so the resolver won't skip → hours hold).
- random selection (fixed seed) until the hour budget.
- replicate the runtime split to report exact train/valid hours.
"""
import json, os, random, re, subprocess, sys

REPO = "/mnt/c/Users/aidan/Desktop/Mecka/EgoVerse"
sys.path.insert(0, REPO)
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df
from egomimic.rldb.zarr.zarr_dataset_multi import split_dataset_names

FPH = 30 * 3600           # frames per hour (30 fps convention)
TARGET_TOTAL_H = 250.0    # 80% ≈ 200h train
SELECT_SEED = 7           # selection shuffle (split uses seed 42 at runtime)
OUT = f"{REPO}/egomimic/hydra_configs/data/extra/mecka_random_250h.json"

# 1) episodes physically present on the mecka_data_v2 volume
env = os.environ.copy(); env["MODAL_ENVIRONMENT"] = "robotics"
r = subprocess.run(["python", "-m", "modal", "volume", "ls", "mecka_data_v2", "/"],
                   capture_output=True, text=True, env=env, timeout=300)
present = set(re.findall(r"([0-9a-fA-F]{24})\.zarr", r.stdout))
print("volume present .zarr episodes:", len(present))

# 2) SQL durations
df = episode_table_to_df(create_default_engine())
df = df[(df["is_deleted"] != True) & (df["num_frames"] > 0)]  # noqa: E712
frames = {h: int(n) for h, n in zip(df["episode_hash"], df["num_frames"])}
print("sql non-deleted w/ frames:", len(frames))

# 3) intersect
cand = [h for h in frames if h in present]
print(f"candidates (sql ∩ volume): {len(cand)} eps = {sum(frames[h] for h in cand)/FPH:.0f}h")
assert sum(frames[h] for h in cand) / FPH > TARGET_TOTAL_H, "not enough present hours!"

# 4) random accumulate to budget
rng = random.Random(SELECT_SEED); rng.shuffle(cand)
budget = TARGET_TOTAL_H * FPH
pool, acc = [], 0
for h in cand:
    if acc >= budget:
        break
    pool.append(h); acc += frames[h]
print(f"pool: {len(pool)} eps = {acc/FPH:.2f}h total")

# 5) replicate runtime default split (valid_ratio=0.2, seed=42, by episode)
train, valid = split_dataset_names(pool, valid_ratio=0.2, seed=42)
th = sum(frames[h] for h in train) / FPH
vh = sum(frames[h] for h in valid) / FPH
print(f"default split -> TRAIN {len(train)} eps {th:.2f}h | VALID {len(valid)} eps {vh:.2f}h")

# 6) write sorted allowlist
json.dump(sorted(pool), open(OUT, "w"))
print("wrote", OUT, "(", len(pool), "hashes )")
