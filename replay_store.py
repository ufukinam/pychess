# replay_store.py
from __future__ import annotations

import os
import glob
import time
import numpy as np


def save_shard(samples, out_dir: str = "replay", prefix: str = "shard") -> str:
    """
    samples: list of (state(18,8,8), pi(ACTION_SIZE,), v)
    Saves a compressed .npz shard to disk.
    """
    os.makedirs(out_dir, exist_ok=True)

    states = np.stack([s for (s, _, _) in samples]).astype(np.float32)
    pis = np.stack([p for (_, p, _) in samples]).astype(np.float32)
    vs = np.array([v for (_, _, v) in samples], dtype=np.float32)

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}_{np.random.randint(100000):05d}.npz")
    np.savez_compressed(path, states=states, pis=pis, vs=vs)
    return path


def iter_shard_paths(out_dir: str = "replay") -> list[str]:
    return sorted(glob.glob(os.path.join(out_dir, "*.npz")))


def load_shards_into_buffer(replay_buffer, out_dir: str = "replay", max_samples: int | None = None) -> int:
    """
    Loads shards and pushes into replay_buffer.add_many(...)

    If max_samples is set, loads newest-first until reaching max_samples.
    Returns number of samples loaded.
    """
    paths = iter_shard_paths(out_dir)
    if not paths:
        return 0

    if max_samples is not None:
        paths = list(reversed(paths))  # newest first

    loaded = 0
    for path in paths:
        data = np.load(path)
        states = data["states"]
        pis = data["pis"]
        vs = data["vs"]

        n = int(states.shape[0])
        samples = [(states[i], pis[i], float(vs[i])) for i in range(n)]

        if max_samples is not None and loaded + n > max_samples:
            take = max_samples - loaded
            samples = samples[:take]
            replay_buffer.add_many(samples)
            loaded += take
            break

        replay_buffer.add_many(samples)
        loaded += n

    return loaded
