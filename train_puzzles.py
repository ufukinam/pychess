from __future__ import annotations

"""
Entry-point script for puzzle-supervised training.

What it does:
- Loads data from CSV or prebuilt NPZ cache shards.
- Trains policy head with legal-move masking.
- Evaluates on validation split each epoch.
- Saves latest/best checkpoints and optional PGN examples.

External library usage:
- `argparse`: command-line options.
- `numpy`/`torch`: model training math.
- `SummaryWriter`: logs metrics for TensorBoard.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from encode import IN_CHANNELS
from net import AlphaZeroNet
from puzzle_train_data import (
    PuzzleDataset,
    filter_valid_shards,
    list_cache_shards,
    load_cached_shard,
    mask_illegal_logits,
    shard_num_samples,
    train_one_epoch,
    train_one_epoch_from_shards,
)
from puzzle_train_eval import (
    evaluate_puzzle_validation,
    evaluate_puzzle_validation_from_shards,
    save_best_validation_pgns,
    save_best_validation_pgns_from_shards,
)
from puzzles import PuzzleExample, load_puzzles, split_train_val


def _parse_int_list(csv_text: str) -> list[int]:
    """Parse comma-separated ints for auto-tune grids."""
    vals = []
    for p in str(csv_text).split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    return vals


def _benchmark_cache_mode(
    device: str,
    train_shards: list[str],
    batch_size: int,
    torch_threads: int,
    lr: float,
    max_batches: int,
) -> tuple[float, int, float]:
    """Quick throughput benchmark when training from shard cache."""
    if device == "cpu" and torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=128, num_blocks=10).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()

    bs = max(1, int(batch_size))
    max_b = max(1, int(max_batches))
    batches = 0
    samples = 0
    t0 = time.time()

    for path in train_shards:
        states, target_idx, legal_masks = load_cached_shard(path)
        n = int(states.shape[0])
        order = np.random.permutation(n)
        for start in range(0, n, bs):
            idx = order[start : start + bs]
            x = torch.from_numpy(states[idx]).float().to(device)
            t = torch.from_numpy(target_idx[idx]).long().to(device)
            mask = torch.from_numpy(legal_masks[idx]).to(device)

            logits, _ = net(x)
            masked_logits = mask_illegal_logits(logits, mask)
            loss = F.cross_entropy(masked_logits, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batches += 1
            samples += int(x.size(0))
            if batches >= max_b:
                elapsed = max(1e-6, time.time() - t0)
                return samples / elapsed, samples, elapsed

    elapsed = max(1e-6, time.time() - t0)
    return samples / elapsed, samples, elapsed


def _benchmark_csv_mode(
    device: str,
    train_examples: list[PuzzleExample],
    batch_size: int,
    torch_threads: int,
    lr: float,
    max_batches: int,
    num_workers: int,
) -> tuple[float, int, float]:
    """Quick throughput benchmark when training from CSV/DataLoader."""
    if device == "cpu" and torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=128, num_blocks=10).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    loader = DataLoader(
        PuzzleDataset(train_examples),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=max(0, int(num_workers)),
    )

    max_b = max(1, int(max_batches))
    batches = 0
    samples = 0
    t0 = time.time()
    for x, t, mask in loader:
        x = x.to(device)
        t = t.to(device)
        mask = mask.to(device)
        logits, _ = net(x)
        masked_logits = mask_illegal_logits(logits, mask)
        loss = F.cross_entropy(masked_logits, t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        batches += 1
        samples += int(x.size(0))
        if batches >= max_b:
            break
    elapsed = max(1e-6, time.time() - t0)
    return samples / elapsed, samples, elapsed


def main() -> None:
    """CLI program for puzzle pretraining and validation."""
    parser = argparse.ArgumentParser(description="Train policy on puzzle best moves.")
    parser.add_argument("--puzzles_csv", type=str, default="", help="Puzzle CSV path.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional prebuilt NPZ cache directory (train_shard_*.npz / val_shard_*.npz).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows; 0 disables.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for CSV mode (0 = main process).",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="Torch intra-op CPU threads (0 keeps backend default).",
    )
    parser.add_argument(
        "--auto_tune_cpu",
        action="store_true",
        help="Benchmark several (batch_size, torch_threads) configs and use the fastest.",
    )
    parser.add_argument(
        "--tune_batch_sizes",
        type=str,
        default="128,256,512",
        help="Comma-separated batch sizes to benchmark.",
    )
    parser.add_argument(
        "--tune_torch_threads",
        type=str,
        default="4,6,8",
        help="Comma-separated torch thread counts to benchmark.",
    )
    parser.add_argument(
        "--tune_max_batches",
        type=int,
        default=120,
        help="Benchmark train batches per config.",
    )
    parser.add_argument(
        "--tune_only",
        action="store_true",
        help="Run auto-tune benchmark and exit without full training.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--overfit_debug_n",
        type=int,
        default=0,
        help="If >0, train on a tiny subset to validate pipeline behavior quickly.",
    )
    parser.add_argument("--pgn_dir", type=str, default="puzzle_games")
    parser.add_argument("--pgn_max_games", type=int, default=25)
    parser.add_argument(
        "--progress_every_batches",
        type=int,
        default=100,
        help="Print train batch progress every N batches (0 disables batch progress).",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="checkpoint_puzzle_latest.pt",
        help="Optional puzzle checkpoint to resume from.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("runs", exist_ok=True)

    use_cache = bool(args.cache_dir)
    train_examples: list[PuzzleExample] = []
    val_examples: list[PuzzleExample] = []
    train_shards: list[str] = []
    val_shards: list[str] = []
    train_loader = None
    val_loader = None

    if use_cache:
        train_shards = list_cache_shards(args.cache_dir, "train")
        val_shards = list_cache_shards(args.cache_dir, "val")
        train_shards = filter_valid_shards(
            train_shards,
            required_keys=("states", "target_idx", "legal_masks_packed"),
            split_name="train",
        )
        val_shards = filter_valid_shards(
            val_shards,
            required_keys=("states", "target_idx", "legal_masks_packed"),
            split_name="val",
        )
        if not train_shards:
            raise RuntimeError(
                f"No train shards found in {args.cache_dir}. Build cache first with build_puzzle_cache.py."
            )
        if not val_shards:
            print("Warning: no val shards found in cache; val metrics will be zeros.")
        print(
            f"Using cache_dir={args.cache_dir} | train_shards={len(train_shards)} | val_shards={len(val_shards)}"
        )
        train_samples = sum(shard_num_samples(p) for p in train_shards)
        val_samples = sum(shard_num_samples(p) for p in val_shards) if val_shards else 0
        est_train_batches = max(1, train_samples // max(1, args.batch_size))
        print(
            f"[Setup] mode=cache train_samples={train_samples} val_samples={val_samples} "
            f"batch_size={args.batch_size} ~train_batches/epoch={est_train_batches}"
        )
    else:
        if not args.puzzles_csv:
            raise RuntimeError("Provide --puzzles_csv or --cache_dir.")
        examples = load_puzzles(args.puzzles_csv, limit=args.limit or None)
        train_examples, val_examples = split_train_val(
            examples, val_ratio=args.val_ratio, seed=args.seed
        )
        if args.overfit_debug_n > 0 and train_examples:
            n = min(args.overfit_debug_n, len(train_examples))
            debug_subset = train_examples[:n]
            train_examples = debug_subset
            val_examples = debug_subset[: max(1, min(len(debug_subset), n // 4 or 1))]
            print(
                f"[Debug] overfit mode on {len(train_examples)} train / {len(val_examples)} val examples"
            )
        if not train_examples:
            raise RuntimeError("No train examples loaded from puzzle CSV.")
        if not val_examples:
            print("Warning: empty validation split; val metrics will be zeros.")

        train_loader = DataLoader(
            PuzzleDataset(train_examples),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0, int(args.num_workers)),
        )
        val_loader = DataLoader(
            PuzzleDataset(val_examples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
        )
        print(
            f"[Setup] mode=csv train={len(train_examples)} val={len(val_examples)} "
            f"batch_size={args.batch_size} train_batches/epoch={len(train_loader)}"
        )

    if args.auto_tune_cpu and device == "cpu":
        batch_grid = sorted({max(1, v) for v in _parse_int_list(args.tune_batch_sizes)})
        thread_grid = sorted({max(1, v) for v in _parse_int_list(args.tune_torch_threads)})
        if not batch_grid:
            batch_grid = [max(1, int(args.batch_size))]
        if not thread_grid:
            thread_grid = [max(1, int(args.torch_threads or 1))]
        print(
            f"[Tune] benchmarking {len(batch_grid)*len(thread_grid)} configs "
            f"(max_batches={max(1, int(args.tune_max_batches))})..."
        )
        best = None
        for bs in batch_grid:
            for th in thread_grid:
                if use_cache:
                    sps, samples, elapsed = _benchmark_cache_mode(
                        device=device,
                        train_shards=train_shards,
                        batch_size=bs,
                        torch_threads=th,
                        lr=float(args.lr),
                        max_batches=max(1, int(args.tune_max_batches)),
                    )
                else:
                    sps, samples, elapsed = _benchmark_csv_mode(
                        device=device,
                        train_examples=train_examples,
                        batch_size=bs,
                        torch_threads=th,
                        lr=float(args.lr),
                        max_batches=max(1, int(args.tune_max_batches)),
                        num_workers=max(0, int(args.num_workers)),
                    )
                print(
                    f"[Tune] batch_size={bs} torch_threads={th} "
                    f"samples={samples} elapsed_s={elapsed:.2f} samples_per_s={sps:.1f}"
                )
                rec = (sps, bs, th)
                if best is None or rec[0] > best[0]:
                    best = rec
        assert best is not None
        args.batch_size = int(best[1])
        args.torch_threads = int(best[2])
        torch.set_num_threads(int(args.torch_threads))
        print(
            f"[Tune] best config -> batch_size={args.batch_size} "
            f"torch_threads={args.torch_threads} samples_per_s={best[0]:.1f}"
        )
        if not use_cache:
            train_loader = DataLoader(
                PuzzleDataset(train_examples),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(0, int(args.num_workers)),
            )
            val_loader = DataLoader(
                PuzzleDataset(val_examples),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(0, int(args.num_workers)),
            )
            print(
                f"[Tune] rebuilt CSV loaders with batch_size={args.batch_size} "
                f"train_batches/epoch={len(train_loader)}"
            )
        if args.tune_only:
            return

    net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=128, num_blocks=10).to(device)
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        payload = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            net.load_state_dict(payload["model_state_dict"])
            it = payload.get("iter")
            tag = f" (iter={it})" if it is not None else ""
            print(f"Loaded {args.resume_checkpoint}{tag}")
        else:
            net.load_state_dict(payload)
            print(f"Loaded {args.resume_checkpoint}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    writer = SummaryWriter(log_dir="runs/chesszero_puzzles")
    best_val_top1 = -1.0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_t0 = time.time()
            print(f"[Epoch {epoch:03d}] starting...")
            train_s = 0.0
            val_s = 0.0
            ckpt_s = 0.0
            pgn_s = 0.0
            if use_cache:
                t0 = time.time()
                train_loss = train_one_epoch_from_shards(
                    net,
                    train_shards,
                    opt,
                    device=device,
                    batch_size=args.batch_size,
                    label_smoothing=max(0.0, args.label_smoothing),
                    progress_every_batches=max(0, args.progress_every_batches),
                )
                train_s = time.time() - t0
                t0 = time.time()
                val_metrics = evaluate_puzzle_validation_from_shards(
                    net,
                    val_shards,
                    device=device,
                    batch_size=args.batch_size,
                    progress_every_shards=1,
                )
                val_s = time.time() - t0
            else:
                t0 = time.time()
                train_loss = train_one_epoch(
                    net,
                    train_loader,
                    opt,
                    device=device,
                    label_smoothing=max(0.0, args.label_smoothing),
                    progress_every_batches=max(0, args.progress_every_batches),
                )
                train_s = time.time() - t0
                t0 = time.time()
                val_metrics = evaluate_puzzle_validation(net, val_loader, device=device)
                val_s = time.time() - t0

            t0 = time.time()
            print(
                f"[Puzzle] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"top1={val_metrics['val_top1']:.4f} "
                f"top5={val_metrics['val_top5']:.4f} "
                f"raw_legality={val_metrics['raw_legality_rate']:.4f} "
                f"masked_legality={val_metrics['masked_legality_rate']:.4f} "
                f"epoch_time={time.time()-epoch_t0:.1f}s"
            )

            writer.add_scalar("puzzle/train_loss", train_loss, epoch)
            writer.add_scalar("puzzle/val_loss", val_metrics["val_loss"], epoch)
            writer.add_scalar("puzzle/val_top1", val_metrics["val_top1"], epoch)
            writer.add_scalar("puzzle/val_top5", val_metrics["val_top5"], epoch)
            writer.add_scalar("puzzle/raw_legality_rate", val_metrics["raw_legality_rate"], epoch)
            writer.add_scalar(
                "puzzle/masked_legality_rate", val_metrics["masked_legality_rate"], epoch
            )
            writer.add_scalar("puzzle/legality_rate", val_metrics["legality_rate"], epoch)
            ckpt_s += time.time() - t0

            t0 = time.time()
            ckpt_payload = {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "in_channels": IN_CHANNELS,
                "channels": 128,
                "num_blocks": 10,
            }
            torch.save(ckpt_payload, "checkpoint_puzzle_latest.pt")
            ckpt_s += time.time() - t0
            if val_metrics["val_top1"] > best_val_top1:
                best_val_top1 = val_metrics["val_top1"]
                t0 = time.time()
                torch.save(ckpt_payload, "checkpoint_puzzle_best.pt")
                ckpt_s += time.time() - t0
                print(f"[Puzzle] new best val_top1={best_val_top1:.4f} -> checkpoint_puzzle_best.pt")
                if use_cache:
                    t0 = time.time()
                    pgn_path, n_pgn = save_best_validation_pgns_from_shards(
                        net,
                        val_shards,
                        device=device,
                        out_dir=args.pgn_dir,
                        epoch=epoch,
                        max_games=args.pgn_max_games,
                    )
                    pgn_s += time.time() - t0
                else:
                    t0 = time.time()
                    pgn_path, n_pgn = save_best_validation_pgns(
                        net,
                        val_examples,
                        device=device,
                        out_dir=args.pgn_dir,
                        epoch=epoch,
                        max_games=args.pgn_max_games,
                    )
                    pgn_s += time.time() - t0
                if pgn_path:
                    print(f"[Puzzle] saved {n_pgn} best validation games to {pgn_path}")
            print(
                f"[EpochTiming] epoch={epoch:03d} "
                f"train_s={train_s:.2f} val_s={val_s:.2f} ckpt_s={ckpt_s:.2f} "
                f"pgn_s={pgn_s:.2f} total_s={time.time()-epoch_t0:.2f}"
            )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
