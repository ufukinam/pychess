from __future__ import annotations

"""
Puzzle-data loading and puzzle-policy training helpers.

External library usage:
- `numpy`: numeric arrays, shuffling, NPZ serialization helpers.
- `torch`: tensors, model execution, and optimization.
- `torch.utils.data.Dataset/DataLoader`: mini-batch input pipeline.
- `torch.nn.functional.cross_entropy`: supervised classification loss.
"""

import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from encode import ACTION_SIZE
from net import AlphaZeroNet
from puzzles import PuzzleExample


def mask_illegal_logits(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    """Mask illegal actions so loss/argmax only consider legal moves."""
    return logits.masked_fill(~legal_masks, -1e9)


def _filter_valid_targets(
    x: torch.Tensor,
    target_idx: torch.Tensor,
    legal_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Keep only rows where the supervised target action is legal per mask.
    Invalid rows can exist in stale/corrupt cache shards and would explode CE loss.
    """
    valid = legal_masks.gather(1, target_idx.unsqueeze(1)).squeeze(1).bool()
    invalid_n = int((~valid).sum().item())
    if invalid_n == 0:
        return x, target_idx, legal_masks, 0
    return x[valid], target_idx[valid], legal_masks[valid], invalid_n


class PuzzleDataset(Dataset):
    """Simple Dataset over in-memory `PuzzleExample` objects."""
    def __init__(self, examples: list[PuzzleExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return (
            torch.from_numpy(ex.state).float(),
            torch.tensor(ex.target_index, dtype=torch.long),
            torch.from_numpy(ex.legal_mask),
        )


class CachedPuzzleDataset(Dataset):
    """Dataset view over preloaded shard arrays."""
    def __init__(self, states: np.ndarray, target_idx: np.ndarray, legal_masks: np.ndarray):
        self.states = states
        self.target_idx = target_idx
        self.legal_masks = legal_masks

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.states[idx]).float(),
            torch.tensor(int(self.target_idx[idx]), dtype=torch.long),
            torch.from_numpy(self.legal_masks[idx]),
        )


def list_cache_shards(cache_dir: str, split: str) -> list[str]:
    """Return sorted shard paths for a split (`train` or `val`)."""
    pattern = os.path.join(cache_dir, f"{split}_shard_*.npz")
    return sorted(glob.glob(pattern))


def _shard_has_keys(path: str, required_keys: tuple[str, ...]) -> bool:
    try:
        data = np.load(path)
    except Exception:
        return False
    files = set(data.files)
    return all(k in files for k in required_keys)


def filter_valid_shards(
    shard_paths: list[str],
    required_keys: tuple[str, ...],
    split_name: str,
) -> list[str]:
    valid: list[str] = []
    dropped = 0
    for p in shard_paths:
        if _shard_has_keys(p, required_keys):
            valid.append(p)
        else:
            dropped += 1
            print(f"[Cache] skipping invalid {split_name} shard: {p}")
    if dropped > 0:
        print(f"[Cache] dropped {dropped} invalid {split_name} shard(s)")
    return valid


def load_cached_shard(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one cached shard and unpack its packed legality mask bits."""
    data = np.load(path)
    states = data["states"].astype(np.float32, copy=False)
    target_idx = data["target_idx"].astype(np.int64, copy=False)
    packed = data["legal_masks_packed"]
    legal_masks = np.unpackbits(packed, axis=1, count=ACTION_SIZE).astype(np.bool_, copy=False)
    return states, target_idx, legal_masks


def shard_num_samples(path: str) -> int:
    data = np.load(path)
    return int(data["target_idx"].shape[0])


def load_cached_val_shard_with_meta(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    states = data["states"].astype(np.float32, copy=False)
    target_idx = data["target_idx"].astype(np.int64, copy=False)
    packed = data["legal_masks_packed"]
    legal_masks = np.unpackbits(packed, axis=1, count=ACTION_SIZE).astype(np.bool_, copy=False)
    fens = data["fens"] if "fens" in data else np.asarray([], dtype=str)
    moves = data["moves"] if "moves" in data else np.asarray([], dtype=str)
    puzzle_ids = data["puzzle_ids"] if "puzzle_ids" in data else np.asarray([], dtype=str)
    return states, target_idx, legal_masks, fens, moves, puzzle_ids


def train_one_epoch(
    net: AlphaZeroNet,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    label_smoothing: float = 0.0,
    progress_every_batches: int = 0,
) -> float:
    """Train for one epoch using a DataLoader (CSV/in-memory path)."""
    net.train()
    total = 0
    loss_sum = 0.0
    invalid_target_rows = 0
    total_batches = len(loader)
    t0 = time.time()
    for batch_idx, (x, target_idx, legal_masks) in enumerate(loader, start=1):
        x = x.to(device)
        target_idx = target_idx.to(device)
        legal_masks = legal_masks.to(device)
        x, target_idx, legal_masks, invalid_n = _filter_valid_targets(x, target_idx, legal_masks)
        invalid_target_rows += invalid_n
        if x.size(0) == 0:
            continue
        logits, _ = net(x)
        masked_logits = mask_illegal_logits(logits, legal_masks)
        loss = F.cross_entropy(
            masked_logits, target_idx, label_smoothing=label_smoothing
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        b = x.size(0)
        total += b
        loss_sum += float(loss.item()) * b
        if progress_every_batches > 0 and (
            batch_idx % progress_every_batches == 0 or batch_idx == total_batches
        ):
            avg_loss = loss_sum / max(1, total)
            elapsed = time.time() - t0
            print(
                f"[Train] batch {batch_idx}/{total_batches} "
                f"({100.0*batch_idx/max(1,total_batches):.1f}%) "
                f"avg_loss={avg_loss:.4f} invalid_targets={invalid_target_rows} elapsed={elapsed:.1f}s"
            )
    if invalid_target_rows > 0:
        print(f"[Train] skipped {invalid_target_rows} rows with illegal target labels")
    return loss_sum / max(1, total)


def train_one_epoch_from_shards(
    net: AlphaZeroNet,
    shard_paths: list[str],
    opt: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    label_smoothing: float = 0.0,
    progress_every_batches: int = 0,
) -> float:
    """Train for one epoch by streaming batches from NPZ shards."""
    if not shard_paths:
        return 0.0

    net.train()
    total = 0
    loss_sum = 0.0
    invalid_target_rows = 0
    total_shards = len(shard_paths)
    epoch_t0 = time.time()
    bs = max(1, int(batch_size))
    total_load_s = 0.0
    total_train_s = 0.0
    for shard_idx, path in enumerate(shard_paths, start=1):
        load_t0 = time.time()
        states, target_idx, legal_masks = load_cached_shard(path)
        total_load_s += time.time() - load_t0
        shard_samples = int(states.shape[0])
        shard_batches = max(1, (shard_samples + bs - 1) // bs)
        order = np.random.permutation(shard_samples)
        shard_t0 = time.time()
        print(
            f"[Train] shard {shard_idx}/{total_shards} "
            f"samples={shard_samples} batches={shard_batches}"
        )
        for batch_idx, start in enumerate(range(0, shard_samples, bs), start=1):
            idx = order[start : start + bs]
            x = torch.from_numpy(states[idx]).float().to(device)
            t = torch.from_numpy(target_idx[idx]).long().to(device)
            mask = torch.from_numpy(legal_masks[idx]).to(device)
            x, t, mask, invalid_n = _filter_valid_targets(x, t, mask)
            invalid_target_rows += invalid_n
            if x.size(0) == 0:
                continue
            logits, _ = net(x)
            masked_logits = mask_illegal_logits(logits, mask)
            loss = F.cross_entropy(
                masked_logits, t, label_smoothing=label_smoothing
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            b = x.size(0)
            total += b
            loss_sum += float(loss.item()) * b
            if progress_every_batches > 0 and (
                batch_idx % progress_every_batches == 0 or batch_idx == shard_batches
            ):
                avg_loss = loss_sum / max(1, total)
                elapsed = time.time() - epoch_t0
                print(
                    f"[Train] shard {shard_idx}/{total_shards} batch {batch_idx}/{shard_batches} "
                    f"avg_loss={avg_loss:.4f} invalid_targets={invalid_target_rows} elapsed={elapsed:.1f}s"
                )
        print(
            f"[Train] shard {shard_idx}/{total_shards} done in {time.time()-shard_t0:.1f}s"
        )
        total_train_s += time.time() - shard_t0
    if invalid_target_rows > 0:
        print(f"[Train] skipped {invalid_target_rows} rows with illegal target labels")
    print(
        f"[TrainTiming] total_load_s={total_load_s:.2f} "
        f"total_train_s={total_train_s:.2f} total_epoch_s={time.time()-epoch_t0:.2f}"
    )
    return loss_sum / max(1, total)
