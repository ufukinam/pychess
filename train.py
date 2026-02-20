from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from net import AlphaZeroNet
from selfplay import play_self_game
from replay_store import save_shard, load_shards_into_buffer
from selfplay_train_core import ReplayBuffer, train_step
from eval import eval_net_vs_random


def main():
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
    args = parser.parse_args()

    # Optional CPU threading tweak (sometimes speeds up, sometimes slows down)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AlphaZeroNet(in_channels=18, channels=64, num_blocks=5).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    rb = ReplayBuffer(maxlen=50000)

    os.makedirs("games", exist_ok=True)
    os.makedirs("replay", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    writer = SummaryWriter(log_dir="runs/chesszero")

    # Resume model: optionally transfer from puzzle-pretrained checkpoint.
    loaded_ckpt = None
    if args.prefer_puzzle_init and args.puzzle_checkpoint and os.path.exists(args.puzzle_checkpoint):
        net.load_state_dict(torch.load(args.puzzle_checkpoint, map_location=device))
        loaded_ckpt = args.puzzle_checkpoint
    elif args.init_checkpoint and os.path.exists(args.init_checkpoint):
        net.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        loaded_ckpt = args.init_checkpoint

    if loaded_ckpt:
        print(f"Loaded {loaded_ckpt}")
        with torch.inference_mode():
            eval_stats = eval_net_vs_random(net, games=12, num_sims=25, device=device)
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
    loaded = load_shards_into_buffer(rb, out_dir="replay", max_samples=50000)
    if loaded > 0:
        print(f"Loaded {loaded} replay samples from ./replay")

    iters = 5
    games_per_iter = 40
    batch_size = 64
    train_batches = 32

    # Bootstrap only if buffer is empty
    if len(rb) == 0:
        bootstrap_games = 5
        for i in range(bootstrap_games):
            with torch.inference_mode():
                samples, stats, pgn_path = play_self_game(
                    net, num_sims=100, device=device, pgn_dir="games", verbose=False
                )
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
                    samples, stats, pgn_path = play_self_game(
                        net, num_sims=100, device=device, pgn_dir="games", verbose=False
                    )
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
            shard_path = save_shard(iter_samples, out_dir="replay")
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

            losses = []
            for _ in range(train_batches):
                s, pi, v = rb.sample(batch_size)
                losses.append(train_step(net, opt, s, pi, v, device=device))

            avg = np.mean(losses, axis=0)
            print(f"Iter {it:03d} | loss={avg[0]:.4f} policy={avg[1]:.4f} value={avg[2]:.4f} | rb={len(rb)}")

            writer.add_scalar("train/loss", avg[0], it)
            writer.add_scalar("train/policy_loss", avg[1], it)
            writer.add_scalar("train/value_loss", avg[2], it)

            # Checkpoints
            torch.save(net.state_dict(), "checkpoint_latest.pt")
            if it % 10 == 0:
                torch.save(net.state_dict(), f"checkpoint_{it:03d}.pt")
                print("Saved checkpoint.")

            # ---- Evaluation vs random ----
            if it % 2 == 0:
                with torch.inference_mode():
                    eval_stats = eval_net_vs_random(net, games=12, num_sims=25, device=device)

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
