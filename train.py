from __future__ import annotations

"""
Main self-play reinforcement learning loop.

High-level iteration:
1) Generate self-play games with MCTS.
2) Store samples in replay buffer.
3) Train policy+value net on replay mini-batches.
4) Periodically evaluate vs random baseline and checkpoint.
"""

import argparse
import copy
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from net import AlphaZeroNet
from selfplay import play_self_game
from replay_store import save_shard, load_shards_into_buffer
from selfplay_train_core import ReplayBuffer, train_step
from eval import eval_candidate_vs_baseline, eval_net_vs_random


def main():
    """CLI entry point for self-play training."""
    parser = argparse.ArgumentParser(description="Self-play training loop.")
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="checkpoint_latest.pt",
        help="Primary checkpoint to load before training.",
    )
    parser.add_argument(
        "--puzzle_checkpoint",
        type=str,
        default="checkpoint_puzzle_latest.pt",
        help="Puzzle-pretrained checkpoint used when --prefer_puzzle_init is set.",
    )
    parser.add_argument(
        "--prefer_puzzle_init",
        action="store_true",
        help="If set, prefer puzzle checkpoint over init checkpoint when available.",
    )
    parser.add_argument("--iters", type=int, default=5, help="Training iterations.")
    parser.add_argument("--games_per_iter", type=int, default=40, help="Self-play games per iteration.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--train_batches", type=int, default=32, help="Gradient batches per iteration.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--replay_dir", type=str, default="replay", help="Replay shard directory.")
    parser.add_argument("--num_sims", type=int, default=100, help="MCTS sims per self-play move.")
    parser.add_argument("--eval_num_sims", type=int, default=25, help="MCTS sims for eval games.")
    parser.add_argument("--draw_penalty", type=float, default=0.0, help="Value target for draw-like stops.")
    parser.add_argument("--stop_on_threefold", action="store_true", help="Stop game on claimable threefold.")
    parser.add_argument("--no_progress_limit", type=int, default=30, help="Halfmove cutoff for no-progress stop.")
    parser.add_argument("--no_progress_penalty", type=float, default=0.0, help="Target when no-progress stop triggers.")
    parser.add_argument("--repeat2_penalty", type=float, default=0.0, help="Target when 2x repeat stop triggers.")
    parser.add_argument("--stop_on_repeat2", action="store_true", help="Stop game on second position repetition.")
    parser.add_argument("--temp_floor", type=float, default=0.1, help="Post-opening temperature floor.")
    parser.add_argument("--use_material_shaping", action="store_true", help="Enable material/exchange shaping.")
    parser.add_argument("--material_scale", type=float, default=0.0, help="Scale for material shaping.")
    parser.add_argument("--exchange_scale", type=float, default=0.0, help="Scale for exchange shaping.")
    parser.add_argument("--early_sims", type=int, default=0, help="Opening-phase sims per move (0=use num_sims).")
    parser.add_argument("--early_plies", type=int, default=16, help="Opening plies using early_sims.")
    parser.add_argument("--late_sims", type=int, default=0, help="Mid/endgame sims per move (0=use num_sims).")
    parser.add_argument("--gate_games", type=int, default=8, help="Candidate-vs-baseline gating games per iter (0 disables).")
    parser.add_argument("--gate_min_score", type=float, default=0.55, help="Minimum gating score to accept candidate.")
    args = parser.parse_args()

    # Optional CPU threading tweak (sometimes speeds up, sometimes slows down)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AlphaZeroNet(in_channels=18, channels=64, num_blocks=5).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(args.lr))

    rb = ReplayBuffer(maxlen=50000)

    os.makedirs("games", exist_ok=True)
    os.makedirs(args.replay_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    writer = SummaryWriter(log_dir="runs/chesszero")

    # Resume model: optionally transfer from puzzle-pretrained checkpoint.
    loaded_ckpt = None

    def _load_checkpoint(path: str) -> None:
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            net.load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                opt.load_state_dict(payload["optimizer_state_dict"])
        else:
            # Backward-compat: old checkpoints with weights-only state_dict.
            net.load_state_dict(payload)
    if args.prefer_puzzle_init and args.puzzle_checkpoint and os.path.exists(args.puzzle_checkpoint):
        _load_checkpoint(args.puzzle_checkpoint)
        loaded_ckpt = args.puzzle_checkpoint
    elif args.init_checkpoint and os.path.exists(args.init_checkpoint):
        _load_checkpoint(args.init_checkpoint)
        loaded_ckpt = args.init_checkpoint

    if loaded_ckpt:
        print(f"Loaded {loaded_ckpt}")
        with torch.inference_mode():
            eval_stats = eval_net_vs_random(
                net, games=12, num_sims=int(args.eval_num_sims), device=device
            )
        print(
            f"[INIT EVAL vs Random] W={eval_stats['wins']} D={eval_stats['draws']} "
            f"L={eval_stats['losses']} score={eval_stats['score']:.2f} avg={eval_stats['avg_result']:+.2f}"
        )
        writer.add_scalar("eval_random/init_score", eval_stats["score"], 0)
        writer.add_scalar("eval_random/init_avg_result", eval_stats["avg_result"], 0)
    else:
        print(
            "No init checkpoint loaded. To start from puzzle pretraining, run: "
            "python train.py --prefer_puzzle_init"
        )

    # Resume replay
    loaded = load_shards_into_buffer(rb, out_dir=args.replay_dir, max_samples=50000)
    if loaded > 0:
        print(f"Loaded {loaded} replay samples from ./{args.replay_dir}")

    iters = int(args.iters)
    games_per_iter = int(args.games_per_iter)
    batch_size = int(args.batch_size)
    train_batches = int(args.train_batches)
    selfplay_kwargs = {
        "num_sims": int(args.num_sims),
        "device": device,
        "pgn_dir": "games",
        "verbose": False,
        "draw_penalty": float(args.draw_penalty),
        "stop_on_threefold": bool(args.stop_on_threefold),
        "no_progress_limit": int(args.no_progress_limit),
        "no_progress_penalty": float(args.no_progress_penalty),
        "repeat2_penalty": float(args.repeat2_penalty),
        "stop_on_repeat2": bool(args.stop_on_repeat2),
        "temp_floor": float(args.temp_floor),
        "use_material_shaping": bool(args.use_material_shaping),
        "material_scale": float(args.material_scale),
        "exchange_scale": float(args.exchange_scale),
        "early_sims": (None if int(args.early_sims) <= 0 else int(args.early_sims)),
        "early_plies": int(args.early_plies),
        "late_sims": (None if int(args.late_sims) <= 0 else int(args.late_sims)),
    }

    # Bootstrap only if buffer is empty
    if len(rb) == 0:
        bootstrap_games = 5
        for i in range(bootstrap_games):
            with torch.inference_mode():
                samples, stats, pgn_path = play_self_game(net, **selfplay_kwargs)
            rb.add_many(samples)
            print(f"Bootstrap {i+1}/{bootstrap_games}: result={stats['result_str']} plies={stats['plies']} pgn={pgn_path}")

    try:
        for it in range(1, iters + 1):
            # -------- Self-play --------
            results = []
            plies_list = []
            z_list = []
            pgn_paths = []
            iter_samples = []

            captures_list = []
            pawn_captures_list = []
            threefold_list = []
            drawlike_list = []
            maxplies_list = []
            no_prog_list = []
            repeat2_list = []
            halfmove_end_list = []

            for _ in range(games_per_iter):
                with torch.inference_mode():
                    samples, stats, pgn_path = play_self_game(net, **selfplay_kwargs)
                rb.add_many(samples)
                iter_samples.extend(samples)

                results.append(stats["result_str"])
                plies_list.append(stats["plies"])
                z_list.append(stats["result_z_white"])
                if pgn_path:
                    pgn_paths.append(pgn_path)

                captures_list.append(stats.get("captures", 0))
                pawn_captures_list.append(stats.get("pawn_captures", 0))
                threefold_list.append(stats.get("threefold_claimed", 0))
                drawlike_list.append(stats.get("draw_like", 0))
                maxplies_list.append(stats.get("ended_by_maxplies", 0))
                no_prog_list.append(stats.get("broke_no_progress", 0))
                repeat2_list.append(stats.get("broke_repeat2", 0))
                halfmove_end_list.append(stats.get("halfmove_clock_end", 0))

            # Persist training data
            shard_path = save_shard(iter_samples, out_dir=args.replay_dir)
            print("Saved replay shard:", shard_path)

            cnt = Counter(results)
            avg_plies = sum(plies_list) / max(1, len(plies_list))
            avg_z = float(np.mean(z_list)) if z_list else 0.0

            avg_captures = sum(captures_list) / max(1, len(captures_list))
            avg_pawn_captures = sum(pawn_captures_list) / max(1, len(pawn_captures_list))
            threefold_rate = sum(threefold_list) / max(1, len(threefold_list))
            draw_rate = sum(drawlike_list) / max(1, len(drawlike_list))
            maxplies_rate = sum(maxplies_list) / max(1, len(maxplies_list))
            no_prog_rate = sum(no_prog_list) / max(1, len(no_prog_list))
            repeat2_rate = sum(repeat2_list) / max(1, len(repeat2_list))
            avg_halfmove_end = sum(halfmove_end_list) / max(1, len(halfmove_end_list))

            print(f"LoopStops: no-progress {no_prog_rate:.2f} | repeat2 {repeat2_rate:.2f} | avg halfmove_end {avg_halfmove_end:.1f}")

            print(
                f"Self-play: W {cnt.get('1-0',0)} | D {cnt.get('1/2-1/2',0)} | "
                f"L {cnt.get('0-1',0)} | avg plies {avg_plies:.1f} | avg z {avg_z:+.2f}"
            )
            print(
                f"Extra: avg captures {avg_captures:.1f} | avg pawn captures {avg_pawn_captures:.1f} | "
                f"threefold {threefold_rate:.2f} | draw {draw_rate:.2f} | maxplies {maxplies_rate:.2f}"
            )

            if pgn_paths:
                print("Latest PGN:", pgn_paths[-1])

            # TensorBoard: self-play
            writer.add_scalar("selfplay/wins", cnt.get("1-0", 0), it)
            writer.add_scalar("selfplay/draws", cnt.get("1/2-1/2", 0), it)
            writer.add_scalar("selfplay/losses", cnt.get("0-1", 0), it)
            writer.add_scalar("selfplay/avg_plies", avg_plies, it)
            writer.add_scalar("selfplay/avg_z_white", avg_z, it)
            writer.add_scalar("selfplay/avg_captures", avg_captures, it)
            writer.add_scalar("selfplay/avg_pawn_captures", avg_pawn_captures, it)
            writer.add_scalar("selfplay/threefold_rate", threefold_rate, it)
            writer.add_scalar("selfplay/draw_rate", draw_rate, it)
            writer.add_scalar("selfplay/maxplies_rate", maxplies_rate, it)
            writer.add_scalar("selfplay/no_progress_rate", no_prog_rate, it)
            writer.add_scalar("selfplay/repeat2_rate", repeat2_rate, it)
            writer.add_scalar("selfplay/avg_halfmove_clock_end", avg_halfmove_end, it)
            writer.add_scalar("replay/size", len(rb), it)

            # -------- Train --------
            if len(rb) < batch_size:
                print("Replay buffer too small to train yet.")
                continue

            before_train_state = copy.deepcopy(net.state_dict())
            before_opt_state = copy.deepcopy(opt.state_dict())
            losses = []
            for _ in range(train_batches):
                s, pi, v = rb.sample(batch_size)
                losses.append(train_step(net, opt, s, pi, v, device=device))

            avg = np.mean(losses, axis=0)
            print(f"Iter {it:03d} | loss={avg[0]:.4f} policy={avg[1]:.4f} value={avg[2]:.4f} | rb={len(rb)}")

            writer.add_scalar("train/loss", avg[0], it)
            writer.add_scalar("train/policy_loss", avg[1], it)
            writer.add_scalar("train/value_loss", avg[2], it)

            accepted = True
            if int(args.gate_games) > 0:
                baseline = AlphaZeroNet(in_channels=18, channels=64, num_blocks=5).to(device)
                baseline.load_state_dict(before_train_state)
                baseline.eval()
                with torch.inference_mode():
                    gate_stats = eval_candidate_vs_baseline(
                        net,
                        baseline,
                        games=int(args.gate_games),
                        num_sims=int(args.eval_num_sims),
                        device=device,
                    )
                print(
                    f"[GATE vs Previous] W={gate_stats['wins']} D={gate_stats['draws']} "
                    f"L={gate_stats['losses']} score={gate_stats['score']:.2f} "
                    f"threshold={args.gate_min_score:.2f}"
                )
                writer.add_scalar("gate/score_vs_prev", gate_stats["score"], it)
                accepted = float(gate_stats["score"]) >= float(args.gate_min_score)
                if not accepted:
                    net.load_state_dict(before_train_state)
                    opt.load_state_dict(before_opt_state)
                    print("[GATE] rejected candidate; reverted to previous model/optimizer state.")

            # Checkpoints
            if accepted:
                checkpoint_payload = {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "iter": it,
                }
                torch.save(checkpoint_payload, "checkpoint_latest.pt")
                if it % 10 == 0:
                    torch.save(checkpoint_payload, f"checkpoint_{it:03d}.pt")
                    print("Saved checkpoint.")

            # ---- Evaluation vs random ----
            if it % 2 == 0:
                with torch.inference_mode():
                    eval_stats = eval_net_vs_random(
                        net, games=12, num_sims=int(args.eval_num_sims), device=device
                    )

                print(
                    f"[EVAL vs Random] games={eval_stats['games']} "
                    f"W={eval_stats['wins']} D={eval_stats['draws']} L={eval_stats['losses']} "
                    f"score={eval_stats['score']:.2f} avg={eval_stats['avg_result']:+.2f}"
                )

                writer.add_scalar("eval_random/score", eval_stats["score"], it)
                writer.add_scalar("eval_random/wins", eval_stats["wins"], it)
                writer.add_scalar("eval_random/draws", eval_stats["draws"], it)
                writer.add_scalar("eval_random/losses", eval_stats["losses"], it)
                writer.add_scalar("eval_random/avg_result", eval_stats["avg_result"], it)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
