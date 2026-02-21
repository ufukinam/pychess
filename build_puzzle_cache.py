from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import glob
from concurrent.futures import ProcessPoolExecutor

import chess
import numpy as np

from encode import ACTION_SIZE, board_to_tensor, legal_mask, move_to_index


class _ShardWriter:
    def __init__(
        self,
        out_dir: str,
        split: str,
        shard_size: int,
        with_meta: bool = False,
        compression: str = "compressed",
    ):
        self.out_dir = out_dir
        self.split = split
        self.shard_size = max(1, int(shard_size))
        self.with_meta = bool(with_meta)
        self.compression = str(compression)
        self.shard_idx = 0

        self.states: list[np.ndarray] = []
        self.target_idx: list[int] = []
        self.legal_masks: list[np.ndarray] = []
        self.fens: list[str] = []
        self.moves: list[str] = []
        self.puzzle_ids: list[str] = []

        self.total_samples = 0
        self.paths: list[str] = []

    @property
    def buffered_samples(self) -> int:
        return len(self.states)

    @property
    def total_samples_with_buffer(self) -> int:
        return int(self.total_samples + self.buffered_samples)

    def add(
        self,
        state: np.ndarray,
        target_idx: int,
        legal_mask: np.ndarray,
        fen: str = "",
        move_uci: str = "",
        puzzle_id: str = "",
    ) -> None:
        self.states.append(state.astype(np.float16, copy=False))
        self.target_idx.append(int(target_idx))
        self.legal_masks.append(legal_mask.astype(np.bool_, copy=False))
        if self.with_meta:
            self.fens.append(fen)
            self.moves.append(move_uci)
            self.puzzle_ids.append(puzzle_id)
        if len(self.states) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.states:
            return

        states = np.stack(self.states, axis=0)  # (N, 18, 8, 8), float16
        target_idx = np.asarray(self.target_idx, dtype=np.int32)  # (N,)
        masks = np.stack(self.legal_masks, axis=0)  # (N, ACTION_SIZE), bool
        masks_packed = np.packbits(masks, axis=1)  # (N, ceil(ACTION_SIZE/8)), uint8

        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(
            self.out_dir,
            f"{self.split}_shard_{self.shard_idx:05d}.npz",
        )
        payload = {
            "states": states,
            "target_idx": target_idx,
            "legal_masks_packed": masks_packed,
        }
        if self.with_meta:
            payload["fens"] = np.asarray(self.fens)
            payload["moves"] = np.asarray(self.moves)
            payload["puzzle_ids"] = np.asarray(self.puzzle_ids)
        if self.compression == "none":
            np.savez(path, **payload)
        else:
            np.savez_compressed(path, **payload)

        self.paths.append(path)
        self.total_samples += int(states.shape[0])
        self.shard_idx += 1

        self.states.clear()
        self.target_idx.clear()
        self.legal_masks.clear()
        self.fens.clear()
        self.moves.clear()
        self.puzzle_ids.clear()


def _is_val_key(fen_key: str, seed: int, val_ratio: float) -> bool:
    key = f"{seed}|{fen_key}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    bucket = int(digest[:12], 16) / float(16**12)  # [0, 1)
    return bucket < val_ratio


def _cap_reached(cap_samples: int | None, writer: _ShardWriter) -> bool:
    return cap_samples is not None and writer.total_samples_with_buffer >= cap_samples


def _trim_buffer_to_cap(
    writer: _ShardWriter,
    cap_samples: int | None,
    with_meta: bool,
) -> None:
    if cap_samples is None or writer.total_samples_with_buffer <= cap_samples:
        return
    keep = max(0, cap_samples - writer.total_samples)
    writer.states = writer.states[:keep]
    writer.target_idx = writer.target_idx[:keep]
    writer.legal_masks = writer.legal_masks[:keep]
    if with_meta:
        writer.fens = writer.fens[:keep]
        writer.moves = writer.moves[:keep]
        writer.puzzle_ids = writer.puzzle_ids[:keep]


def _extract_best_move_uci(row: dict[str, str]) -> str | None:
    moves = (row.get("Moves") or row.get("moves") or "").strip()
    if not moves:
        return None
    parts = moves.split()
    return parts[0] if parts else None


def _extract_fen(row: dict[str, str]) -> str | None:
    fen = (row.get("FEN") or row.get("fen") or "").strip()
    return fen or None


def _canonical_fen_key(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


def _iter_csv_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = _extract_fen(row)
            best_uci = _extract_best_move_uci(row)
            if not fen or not best_uci:
                continue
            puzzle_id = (row.get("PuzzleId") or row.get("id") or "").strip()
            yield (fen, best_uci, puzzle_id)


def _process_row(payload: tuple[str, str, str]):
    fen, best_uci, puzzle_id = payload
    try:
        board = chess.Board(fen=fen)
        move = chess.Move.from_uci(best_uci)
    except ValueError:
        return None

    target_index = move_to_index(move)
    if target_index < 0 or target_index >= ACTION_SIZE:
        return None

    lm = legal_mask(board)
    if not bool(lm[target_index]):
        return None

    return (
        board_to_tensor(board),
        int(target_index),
        lm,
        f"{_canonical_fen_key(board)}|{target_index}",
        fen,
        best_uci,
        puzzle_id,
    )


def _iter_batches(it, batch_size: int):
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NPZ cache shards from puzzle CSV for fast repeated training."
    )
    parser.add_argument("--puzzles_csv", type=str, required=True, help="Input puzzle CSV path.")
    parser.add_argument("--out_dir", type=str, default="puzzle_cache", help="Cache output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max examples to process; 0 disables.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic split hashing.")
    parser.add_argument(
        "--shard_size",
        type=int,
        default=2048,
        help="Samples per shard (tradeoff: IO overhead vs RAM).",
    )
    parser.add_argument(
        "--max_train_shards",
        type=int,
        default=0,
        help="Optional cap on number of train shards (0 disables cap).",
    )
    parser.add_argument(
        "--max_val_shards",
        type=int,
        default=0,
        help="Optional cap on number of val shards (0 disables cap).",
    )
    parser.add_argument(
        "--clean_out_dir",
        action="store_true",
        help="If set, delete existing train_shard_*.npz / val_shard_*.npz in out_dir before writing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process workers for CSV->tensor conversion. 0 = auto (cpu_count-1), 1 = single-process.",
    )
    parser.add_argument(
        "--row_batch_size",
        type=int,
        default=1024,
        help="Rows submitted together to worker pool.",
    )
    parser.add_argument(
        "--worker_chunksize",
        type=int,
        default=64,
        help="Chunksize used in ProcessPoolExecutor.map.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=("compressed", "none"),
        default="compressed",
        help="Shard write mode. 'none' is much faster but larger on disk.",
    )
    args = parser.parse_args()

    val_ratio = float(min(0.9, max(0.0, args.val_ratio)))
    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    row_batch_size = max(64, int(args.row_batch_size))
    worker_chunksize = max(1, int(args.worker_chunksize))
    limit = int(args.limit) if int(args.limit) > 0 else None
    train_writer = _ShardWriter(
        args.out_dir,
        split="train",
        shard_size=args.shard_size,
        with_meta=False,
        compression=args.compression,
    )
    val_writer = _ShardWriter(
        args.out_dir,
        split="val",
        shard_size=args.shard_size,
        with_meta=True,
        compression=args.compression,
    )

    train_cap_samples = (
        int(args.max_train_shards) * int(args.shard_size)
        if int(args.max_train_shards) > 0
        else None
    )
    val_cap_samples = (
        int(args.max_val_shards) * int(args.shard_size)
        if int(args.max_val_shards) > 0
        else None
    )
    if args.clean_out_dir and os.path.isdir(args.out_dir):
        stale = glob.glob(os.path.join(args.out_dir, "train_shard_*.npz"))
        stale += glob.glob(os.path.join(args.out_dir, "val_shard_*.npz"))
        for p in stale:
            os.remove(p)
        if stale:
            print(f"Removed {len(stale)} existing shard file(s) from {args.out_dir}")

    def _consume_one(rec) -> bool:
        nonlocal seen
        state, target_index, lm, fen_key, fen, best_uci, puzzle_id = rec
        seen += 1
        is_val = _is_val_key(fen_key, seed=args.seed, val_ratio=val_ratio)

        if is_val:
            if not _cap_reached(val_cap_samples, val_writer):
                val_writer.add(
                    state,
                    target_index,
                    lm,
                    fen=fen,
                    move_uci=best_uci,
                    puzzle_id=(puzzle_id or f"row_{seen}"),
                )
        else:
            if not _cap_reached(train_cap_samples, train_writer):
                train_writer.add(state, target_index, lm)

        if seen % 10000 == 0:
            elapsed = max(1e-6, time.time() - t0)
            print(f"processed={seen} ({seen/elapsed:.1f} samples/s)")

        if limit is not None and seen >= limit:
            return True

        train_full = _cap_reached(train_cap_samples, train_writer)
        val_full = _cap_reached(val_cap_samples, val_writer)
        # Stop only when every capped split is full; uncapped splits are not "satisfied".
        train_capped_full = train_cap_samples is not None and train_full
        val_capped_full = val_cap_samples is not None and val_full
        if train_capped_full and val_capped_full:
            print("Reached shard caps; stopping early.")
            return True
        return False

    seen = 0
    t0 = time.time()
    stop_early = False
    print(
        f"Cache build settings: workers={workers} row_batch_size={row_batch_size} "
        f"worker_chunksize={worker_chunksize} compression={args.compression}"
    )
    rows = _iter_csv_rows(args.puzzles_csv)
    if workers <= 1:
        for batch in _iter_batches(rows, row_batch_size):
            for rec in map(_process_row, batch):
                if rec is None:
                    continue
                if _consume_one(rec):
                    stop_early = True
                    break
            if stop_early:
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex_pool:
            for batch in _iter_batches(rows, row_batch_size):
                for rec in ex_pool.map(_process_row, batch, chunksize=worker_chunksize):
                    if rec is None:
                        continue
                    if _consume_one(rec):
                        stop_early = True
                        break
                if stop_early:
                    break

    # Trim spillover if final flush would exceed configured caps.
    _trim_buffer_to_cap(train_writer, train_cap_samples, with_meta=False)
    _trim_buffer_to_cap(val_writer, val_cap_samples, with_meta=True)

    train_writer.flush()
    val_writer.flush()

    manifest = {
        "puzzles_csv": args.puzzles_csv,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "val_ratio": float(val_ratio),
        "shard_size": int(args.shard_size),
        "max_train_shards": int(args.max_train_shards),
        "max_val_shards": int(args.max_val_shards),
        "workers": int(workers),
        "row_batch_size": int(row_batch_size),
        "worker_chunksize": int(worker_chunksize),
        "compression": str(args.compression),
        "processed_examples": int(seen),
        "train_samples": int(train_writer.total_samples),
        "val_samples": int(val_writer.total_samples),
        "train_shards": len(train_writer.paths),
        "val_shards": len(val_writer.paths),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Cache build complete.")
    print(f"  train: {train_writer.total_samples} samples in {len(train_writer.paths)} shards")
    print(f"  val  : {val_writer.total_samples} samples in {len(val_writer.paths)} shards")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
