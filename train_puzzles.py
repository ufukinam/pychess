from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import AlphaZeroNet
from puzzle_train_data import (
    PuzzleDataset,
    filter_valid_shards,
    list_cache_shards,
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


def main() -> None:
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
            PuzzleDataset(train_examples), batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            PuzzleDataset(val_examples), batch_size=args.batch_size, shuffle=False
        )
        print(
            f"[Setup] mode=csv train={len(train_examples)} val={len(val_examples)} "
            f"batch_size={args.batch_size} train_batches/epoch={len(train_loader)}"
        )

    net = AlphaZeroNet(in_channels=18, channels=64, num_blocks=5).to(device)
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        net.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
        print(f"Loaded {args.resume_checkpoint}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir="runs/chesszero_puzzles")
    best_val_top1 = -1.0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_t0 = time.time()
            print(f"[Epoch {epoch:03d}] starting...")
            if use_cache:
                train_loss = train_one_epoch_from_shards(
                    net,
                    train_shards,
                    opt,
                    device=device,
                    batch_size=args.batch_size,
                    label_smoothing=max(0.0, args.label_smoothing),
                    progress_every_batches=max(0, args.progress_every_batches),
                )
                val_metrics = evaluate_puzzle_validation_from_shards(
                    net,
                    val_shards,
                    device=device,
                    batch_size=args.batch_size,
                    progress_every_shards=1,
                )
            else:
                train_loss = train_one_epoch(
                    net,
                    train_loader,
                    opt,
                    device=device,
                    label_smoothing=max(0.0, args.label_smoothing),
                    progress_every_batches=max(0, args.progress_every_batches),
                )
                val_metrics = evaluate_puzzle_validation(net, val_loader, device=device)

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

            torch.save(net.state_dict(), "checkpoint_puzzle_latest.pt")
            if val_metrics["val_top1"] > best_val_top1:
                best_val_top1 = val_metrics["val_top1"]
                torch.save(net.state_dict(), "checkpoint_puzzle_best.pt")
                print(f"[Puzzle] new best val_top1={best_val_top1:.4f} -> checkpoint_puzzle_best.pt")
                if use_cache:
                    pgn_path, n_pgn = save_best_validation_pgns_from_shards(
                        net,
                        val_shards,
                        device=device,
                        out_dir=args.pgn_dir,
                        epoch=epoch,
                        max_games=args.pgn_max_games,
                    )
                else:
                    pgn_path, n_pgn = save_best_validation_pgns(
                        net,
                        val_examples,
                        device=device,
                        out_dir=args.pgn_dir,
                        epoch=epoch,
                        max_games=args.pgn_max_games,
                    )
                if pgn_path:
                    print(f"[Puzzle] saved {n_pgn} best validation games to {pgn_path}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
