from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print("[Interface] Running:", " ".join(cmd))
    return subprocess.call(cmd)


def _py(script: str) -> list[str]:
    return [sys.executable, script]


def cmd_train_selfplay(args: argparse.Namespace) -> int:
    cmd = _py("train.py")
    cmd += ["--init_checkpoint", args.init_checkpoint]
    cmd += ["--puzzle_checkpoint", args.puzzle_checkpoint]
    if args.prefer_puzzle_init:
        cmd.append("--prefer_puzzle_init")
    return _run(cmd)


def cmd_train_puzzles(args: argparse.Namespace) -> int:
    cmd = _py("train_puzzles.py")
    if args.cache_dir:
        cmd += ["--cache_dir", args.cache_dir]
    else:
        cmd += ["--puzzles_csv", args.puzzles_csv]
    cmd += ["--limit", str(args.limit)]
    cmd += ["--batch_size", str(args.batch_size)]
    cmd += ["--epochs", str(args.epochs)]
    cmd += ["--lr", str(args.lr)]
    cmd += ["--val_ratio", str(args.val_ratio)]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--label_smoothing", str(args.label_smoothing)]
    cmd += ["--overfit_debug_n", str(args.overfit_debug_n)]
    cmd += ["--pgn_dir", args.pgn_dir]
    cmd += ["--pgn_max_games", str(args.pgn_max_games)]
    cmd += ["--progress_every_batches", str(args.progress_every_batches)]
    cmd += ["--resume_checkpoint", args.resume_checkpoint]
    return _run(cmd)


def cmd_build_puzzle_cache(args: argparse.Namespace) -> int:
    cmd = _py("build_puzzle_cache.py")
    cmd += ["--puzzles_csv", args.puzzles_csv]
    cmd += ["--out_dir", args.out_dir]
    cmd += ["--limit", str(args.limit)]
    cmd += ["--val_ratio", str(args.val_ratio)]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--shard_size", str(args.shard_size)]
    cmd += ["--max_train_shards", str(args.max_train_shards)]
    cmd += ["--max_val_shards", str(args.max_val_shards)]
    return _run(cmd)


def cmd_generate_puzzles(args: argparse.Namespace) -> int:
    cmd = _py("generate_puzzles.py")
    cmd += ["--out", args.out]
    cmd += ["--count", str(args.count)]
    cmd += ["--seed", str(args.seed)]
    return _run(cmd)


def cmd_play_vs_model(args: argparse.Namespace) -> int:
    cmd = _py("play_vs_model.py")
    cmd += ["--device", args.device]
    cmd += ["--checkpoint", args.checkpoint]
    cmd += ["--num_sims", str(args.num_sims)]
    if args.play_as_black:
        cmd.append("--play_as_black")
    if args.save_training_samples:
        cmd.append("--save_training_samples")
    cmd += ["--human_replay_dir", args.human_replay_dir]
    cmd += ["--human_pgn_dir", args.human_pgn_dir]
    return _run(cmd)


def cmd_pgn_viewer(args: argparse.Namespace) -> int:
    cmd = _py("pgn_viewer.py")
    if args.pgn_path:
        cmd += ["--pgn_path", args.pgn_path]
    if args.load_latest:
        cmd.append("--load_latest")
    cmd += ["--games_dir", args.games_dir]
    return _run(cmd)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Unified interface for ChessZero workflows. "
            "Use subcommands to run training, puzzle pipeline, GUI tools, and cache builders."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser(
        "train-selfplay",
        help="Run self-play training (train.py).",
        description="Train the main model with self-play and replay updates.",
    )
    sp.add_argument(
        "--init_checkpoint",
        type=str,
        default="checkpoint_latest.pt",
        help="Primary self-play checkpoint path to load first.",
    )
    sp.add_argument(
        "--puzzle_checkpoint",
        type=str,
        default="checkpoint_puzzle_best.pt",
        help="Puzzle-pretrained checkpoint used when --prefer_puzzle_init is enabled.",
    )
    sp.add_argument(
        "--prefer_puzzle_init",
        action="store_true",
        help="Prefer puzzle checkpoint over init checkpoint when available.",
    )
    sp.set_defaults(func=cmd_train_selfplay)

    sp = sub.add_parser(
        "train-puzzles",
        help="Run puzzle training (train_puzzles.py).",
        description=(
            "Train policy head using puzzle supervision. "
            "Use either --cache_dir (fast) or --puzzles_csv (direct CSV)."
        ),
    )
    sp.add_argument("--cache_dir", type=str, default="", help="NPZ cache directory (preferred for large datasets).")
    sp.add_argument("--puzzles_csv", type=str, default="", help="Puzzle CSV path when cache_dir is not used.")
    sp.add_argument("--limit", type=int, default=0, help="Max puzzles to load (0 = no limit).")
    sp.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    sp.add_argument("--epochs", type=int, default=5, help="Number of puzzle epochs.")
    sp.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    sp.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (CSV mode only).")
    sp.add_argument("--seed", type=int, default=42, help="Random seed.")
    sp.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for cross entropy.")
    sp.add_argument("--overfit_debug_n", type=int, default=0, help="Debug mode: overfit first N training samples.")
    sp.add_argument("--pgn_dir", type=str, default="puzzle_games", help="Directory for best-validation PGN exports.")
    sp.add_argument("--pgn_max_games", type=int, default=25, help="Max PGNs to export when val_top1 improves.")
    sp.add_argument("--progress_every_batches", type=int, default=100, help="Batch progress print interval (0 disables).")
    sp.add_argument(
        "--resume_checkpoint",
        type=str,
        default="checkpoint_puzzle_latest.pt",
        help="Checkpoint to resume puzzle training from.",
    )
    sp.set_defaults(func=cmd_train_puzzles)

    sp = sub.add_parser(
        "build-puzzle-cache",
        help="Build NPZ puzzle shards (build_puzzle_cache.py).",
        description="Preprocess large puzzle CSV into train/val NPZ shards for faster repeated training.",
    )
    sp.add_argument("--puzzles_csv", type=str, required=True, help="Input puzzle CSV path.")
    sp.add_argument("--out_dir", type=str, default="puzzle_cache", help="Output cache directory.")
    sp.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = no limit).")
    sp.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    sp.add_argument("--seed", type=int, default=42, help="Split seed.")
    sp.add_argument("--shard_size", type=int, default=2048, help="Samples per NPZ shard.")
    sp.add_argument("--max_train_shards", type=int, default=0, help="Cap train shard count (0 = unlimited).")
    sp.add_argument("--max_val_shards", type=int, default=0, help="Cap val shard count (0 = unlimited).")
    sp.set_defaults(func=cmd_build_puzzle_cache)

    sp = sub.add_parser(
        "generate-puzzles",
        help="Generate synthetic puzzle CSV (generate_puzzles.py).",
        description="Generate synthetic tactical puzzles for warm-up experiments.",
    )
    sp.add_argument("--out", type=str, default="puzzles_synthetic.csv", help="Output CSV file.")
    sp.add_argument("--count", type=int, default=256, help="Number of synthetic puzzles.")
    sp.add_argument("--seed", type=int, default=42, help="Random seed.")
    sp.set_defaults(func=cmd_generate_puzzles)

    sp = sub.add_parser(
        "play-vs-model",
        help="Launch human-vs-model GUI (play_vs_model.py).",
        description="Play against current model in a local Tkinter board UI.",
    )
    sp.add_argument("--device", type=str, default="cpu", help="Inference device for model (cpu/cuda).")
    sp.add_argument("--checkpoint", type=str, default="checkpoint_latest.pt", help="Model checkpoint to load.")
    sp.add_argument("--num_sims", type=int, default=50, help="MCTS simulations per model move.")
    sp.add_argument("--play_as_black", action="store_true", help="Start as Black.")
    sp.add_argument("--save_training_samples", action="store_true", help="Save game-derived training shards.")
    sp.add_argument("--human_replay_dir", type=str, default="replay_human", help="Replay shard output directory.")
    sp.add_argument("--human_pgn_dir", type=str, default="human_games", help="Human game PGN output directory.")
    sp.set_defaults(func=cmd_play_vs_model)

    sp = sub.add_parser(
        "pgn-viewer",
        help="Launch PGN viewer GUI (pgn_viewer.py).",
        description="Open and navigate PGN files with move-by-move board and captures.",
    )
    sp.add_argument("--pgn_path", type=str, default="", help="PGN file to open at startup.")
    sp.add_argument("--load_latest", action="store_true", help="Load latest PGN from games_dir.")
    sp.add_argument("--games_dir", type=str, default="games", help="Directory used by --load_latest.")
    sp.set_defaults(func=cmd_pgn_viewer)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train-puzzles" and not args.cache_dir and not args.puzzles_csv:
        parser.error("train-puzzles requires either --cache_dir or --puzzles_csv")

    if args.command == "train-puzzles" and args.cache_dir and not os.path.exists(args.cache_dir):
        print(f"[Warning] cache_dir does not exist yet: {args.cache_dir}")

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
