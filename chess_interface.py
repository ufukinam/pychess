from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print("[Interface] Running:", " ".join(cmd))
    return subprocess.call(cmd)


def _py(script: str) -> list[str]:
    # Unbuffered Python so GUI log streaming remains responsive.
    return [sys.executable, "-u", script]


def cmd_train_selfplay(args: argparse.Namespace) -> int:
    cmd = _py("train.py")
    cmd += ["--init_checkpoint", args.init_checkpoint]
    cmd += ["--puzzle_checkpoint", args.puzzle_checkpoint]
    cmd += ["--iters", str(args.iters)]
    cmd += ["--games_per_iter", str(args.games_per_iter)]
    cmd += ["--batch_size", str(args.batch_size)]
    cmd += ["--train_batches", str(args.train_batches)]
    cmd += ["--lr", str(args.lr)]
    cmd += ["--replay_dir", str(args.replay_dir)]
    cmd += ["--num_sims", str(args.num_sims)]
    cmd += ["--eval_num_sims", str(args.eval_num_sims)]
    cmd += ["--draw_penalty", str(args.draw_penalty)]
    cmd += ["--no_progress_limit", str(args.no_progress_limit)]
    cmd += ["--no_progress_penalty", str(args.no_progress_penalty)]
    cmd += ["--repeat2_penalty", str(args.repeat2_penalty)]
    cmd += ["--temp_floor", str(args.temp_floor)]
    cmd += ["--material_scale", str(args.material_scale)]
    cmd += ["--exchange_scale", str(args.exchange_scale)]
    cmd += ["--early_sims", str(args.early_sims)]
    cmd += ["--early_plies", str(args.early_plies)]
    cmd += ["--late_sims", str(args.late_sims)]
    cmd += ["--gate_games", str(args.gate_games)]
    cmd += ["--gate_min_score", str(args.gate_min_score)]
    if args.prefer_puzzle_init:
        cmd.append("--prefer_puzzle_init")
    if args.stop_on_threefold:
        cmd.append("--stop_on_threefold")
    if args.stop_on_repeat2:
        cmd.append("--stop_on_repeat2")
    if args.use_material_shaping:
        cmd.append("--use_material_shaping")
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
    cmd += ["--num_workers", str(args.num_workers)]
    cmd += ["--torch_threads", str(args.torch_threads)]
    if args.auto_tune_cpu:
        cmd += ["--auto_tune_cpu"]
    cmd += ["--tune_batch_sizes", str(args.tune_batch_sizes)]
    cmd += ["--tune_torch_threads", str(args.tune_torch_threads)]
    cmd += ["--tune_max_batches", str(args.tune_max_batches)]
    if args.tune_only:
        cmd += ["--tune_only"]
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
    cmd += ["--workers", str(args.workers)]
    cmd += ["--compression", str(args.compression)]
    if args.clean_out_dir:
        cmd += ["--clean_out_dir"]
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
    sp.add_argument("--iters", type=int, default=5, help="Training iterations.")
    sp.add_argument("--games_per_iter", type=int, default=40, help="Self-play games per iteration.")
    sp.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    sp.add_argument("--train_batches", type=int, default=32, help="Gradient batches per iteration.")
    sp.add_argument("--lr", type=float, default=1e-4, help="Learning rate for self-play updates.")
    sp.add_argument("--replay_dir", type=str, default="replay", help="Replay shard directory.")
    sp.add_argument("--num_sims", type=int, default=100, help="MCTS sims per self-play move.")
    sp.add_argument("--eval_num_sims", type=int, default=25, help="MCTS sims for evaluations.")
    sp.add_argument("--draw_penalty", type=float, default=0.0, help="Target value for draw-like outcomes.")
    sp.add_argument("--stop_on_threefold", action="store_true", help="Stop games on claimable threefold repetition.")
    sp.add_argument("--no_progress_limit", type=int, default=30, help="Halfmove cutoff for no-progress stop.")
    sp.add_argument("--no_progress_penalty", type=float, default=0.0, help="Target value for no-progress stop.")
    sp.add_argument("--repeat2_penalty", type=float, default=0.0, help="Target value when repeat2 stop is enabled.")
    sp.add_argument("--stop_on_repeat2", action="store_true", help="Stop game on second position repetition.")
    sp.add_argument("--temp_floor", type=float, default=0.1, help="Post-opening move temperature floor.")
    sp.add_argument("--use_material_shaping", action="store_true", help="Enable material/exchange shaping.")
    sp.add_argument("--material_scale", type=float, default=0.0, help="Material shaping scale.")
    sp.add_argument("--exchange_scale", type=float, default=0.0, help="Exchange shaping scale.")
    sp.add_argument("--early_sims", type=int, default=0, help="Opening sims per move (0=use num_sims).")
    sp.add_argument("--early_plies", type=int, default=16, help="Opening plies for early_sims.")
    sp.add_argument("--late_sims", type=int, default=0, help="Late-game sims per move (0=use num_sims).")
    sp.add_argument("--gate_games", type=int, default=8, help="Gating games vs previous model (0 disables).")
    sp.add_argument("--gate_min_score", type=float, default=0.55, help="Minimum gate score to accept update.")
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
    sp.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for CSV-mode puzzle training.",
    )
    sp.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="Torch intra-op CPU threads for puzzle training (0 = backend default).",
    )
    sp.add_argument(
        "--auto_tune_cpu",
        action="store_true",
        help="Benchmark (batch_size, torch_threads) pairs and pick the fastest.",
    )
    sp.add_argument(
        "--tune_batch_sizes",
        type=str,
        default="128,256,512",
        help="Comma-separated batch sizes for auto-tune benchmark.",
    )
    sp.add_argument(
        "--tune_torch_threads",
        type=str,
        default="4,6,8",
        help="Comma-separated torch thread counts for auto-tune benchmark.",
    )
    sp.add_argument(
        "--tune_max_batches",
        type=int,
        default=120,
        help="Number of benchmark batches per config.",
    )
    sp.add_argument(
        "--tune_only",
        action="store_true",
        help="Run benchmark and exit without normal training.",
    )
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
    sp.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes for CSV conversion (0 = auto, cpu_count-1).",
    )
    sp.add_argument(
        "--compression",
        type=str,
        choices=("compressed", "none"),
        default="compressed",
        help="Shard compression mode. 'none' is faster and uses more disk.",
    )
    sp.add_argument(
        "--clean_out_dir",
        action="store_true",
        help="Delete existing shard files in out_dir before writing new shards.",
    )
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
