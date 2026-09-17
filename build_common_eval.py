"""Build a per-domain eval set that no arm in that domain trained on.

Why this is needed: each of the 9 runs validates on its OWN arm's 10% split, so
train/val curves are not comparable across arms -- a bottom-arm model can post a
lower loss simply because its data is easier to fit, while being a worse policy.
Answering "does the quality score predict model performance" requires scoring
every checkpoint on one common, never-trained-on set.

There is no slack in the corpus to carve that from: top and bottom are disjoint
halves whose union is the whole domain, and random is drawn from the same pool.
So the common set is exactly the episodes that every arm containing them held
out, i.e. those in NO arm's train split.

split_catalog is deterministic -- random.Random(seed).shuffle over the catalog
order, first valid_ratio as valid -- so the split can be replicated here exactly
rather than guessed. The catalog order is the order of entries in
_catalog_cache.json after filtering by the arm's eps_to_use, which is what
ZarrDirEpisodeResolver.load_catalog produces.
"""
import json
import random
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("build-common-eval", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)

DOMAINS = ["cleaning-sanitation", "organization-stocking", "dish-handling"]
ARMS = ["top", "bottom", "random"]
SEED = 42
VALID_RATIO = 0.1


def split_for(catalog_hashes, seed=SEED, valid_ratio=VALID_RATIO):
    """Replicate ZipEpisodeResolver.split_catalog exactly."""
    rng = random.Random(seed)
    shuffled = list(catalog_hashes)
    rng.shuffle(shuffled)
    n_valid = max(1, int(len(shuffled) * valid_ratio))
    return set(shuffled[:n_valid]), set(shuffled[n_valid:])  # valid, train


@app.function(volumes={MOUNT: vol}, cpu=4, memory=8192, timeout=3600)
def build():
    vol.reload()
    root = Path(MOUNT)
    cat = json.loads((root / "_catalog_cache.json").read_text())
    order = [e["episode_hash"] for e in cat]          # resolver's catalog order
    n_frames = {e["episode_hash"]: int(e["n_frames"]) for e in cat}
    rank = {h: i for i, h in enumerate(order)}

    out = {}
    for dom in DOMAINS:
        arm_hashes = {}
        for arm in ARMS:
            f = root / "_arms" / f"{dom}__{arm}.json"
            hs = set(json.load(open(f)))
            # catalog order, filtered -- exactly what load_catalog returns
            arm_hashes[arm] = [h for h in order if h in hs]

        trained, held = set(), set()
        for arm in ARMS:
            v, t = split_for(arm_hashes[arm])
            trained |= t
            held |= v
            print(f"  {dom}__{arm}: {len(arm_hashes[arm])} eps -> "
                  f"train {len(t)}, valid {len(v)}")

        common = sorted(held - trained, key=lambda h: rank[h])
        frames = sum(n_frames.get(h, 0) for h in common)
        out[dom] = {"n_episodes": len(common), "n_frames": frames}
        (root / "_arms" / f"_eval_common__{dom}.json").write_text(json.dumps(common))
        print(f"{dom}: common eval = {len(common)} episodes, {frames:,} frames "
              f"(union of holdouts {len(held)}, trained-anywhere {len(trained)})\n")

    vol.commit()
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(build.remote(), indent=2))
