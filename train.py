from __future__ import annotations

"""
Main self-play reinforcement learning loop.

Iteration:
1) Generate self-play games with MCTS (position history enabled).
2) Store samples in replay buffer.
3) Train policy+value net on replay mini-batches (with optional augmentation).
4) Periodically evaluate vs random baseline and checkpoint.
"""

import argparse
import copy
import math
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from encode import IN_CHANNELS
from net import AlphaZeroNet
from selfplay import play_self_game
from replay_store import save_shard, load_shards_into_buffer
from selfplay_train_core import ReplayBuffer, train_step, train_step_with_feedback
from feedback_train_data import FeedbackBuffer, load_feedback_jsonl
from eval import eval_candidate_vs_baseline, eval_net_vs_random


def _cosine_lr(base_lr: float, step: int, total_steps: int,
               warmup_steps: int = 0, min_lr: float = 1e-6) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _set_lr(opt, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def main():
    ap = argparse.ArgumentParser(description="Self-play training loop.")
    ap.add_argument("--init_checkpoint", type=str, default="checkpoint_latest.pt")
    ap.add_argument("--puzzle_checkpoint", type=str, default="checkpoint_puzzle_latest.pt")
    ap.add_argument("--prefer_puzzle_init", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--games_per_iter", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--train_batches", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--replay_dir", type=str, default="replay")
    ap.add_argument("--replay_maxlen", type=int, default=200_000)
    ap.add_argument("--num_sims", type=int, default=400)
    ap.add_argument("--eval_num_sims", type=int, default=50)
    ap.add_argument("--draw_penalty", type=float, default=0.0)
    ap.add_argument("--stop_on_threefold", action="store_true")
    ap.add_argument("--no_progress_limit", type=int, default=30)
    ap.add_argument("--no_progress_penalty", type=float, default=0.0)
    ap.add_argument("--repeat2_penalty", type=float, default=0.0)
    ap.add_argument("--stop_on_repeat2", action="store_true")
    ap.add_argument("--temp_floor", type=float, default=0.1)
    ap.add_argument("--temp_moves", type=int, default=30)
    ap.add_argument("--use_material_shaping", action="store_true")
    ap.add_argument("--material_scale", type=float, default=0.0)
    ap.add_argument("--exchange_scale", type=float, default=0.0)
    ap.add_argument("--early_sims", type=int, default=0)
    ap.add_argument("--early_plies", type=int, default=16)
    ap.add_argument("--late_sims", type=int, default=0)
    ap.add_argument("--gate_games", type=int, default=30)
    ap.add_argument("--gate_min_score", type=float, default=0.52)
    ap.add_argument("--feedback_jsonl", type=str, default="")
    ap.add_argument("--feedback_weight", type=float, default=0.2)
    ap.add_argument("--feedback_batch_size", type=int, default=32)
    ap.add_argument("--feedback_margin", type=float, default=0.2)
    ap.add_argument("--feedback_max_samples", type=int, default=0)
    ap.add_argument("--augment", action="store_true",
                    help="Enable color-flip augmentation on replay batches.")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--num_blocks", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    channels = int(args.channels)
    num_blocks = int(args.num_blocks)
    net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=channels, num_blocks=num_blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(args.lr),
                            weight_decay=float(args.weight_decay))

    rb = ReplayBuffer(maxlen=int(args.replay_maxlen))
    fb = FeedbackBuffer()

    os.makedirs("games", exist_ok=True)
    os.makedirs(args.replay_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    writer = SummaryWriter(log_dir="runs/chesszero")

    if args.feedback_jsonl:
        fb_max = None if int(args.feedback_max_samples) <= 0 else int(args.feedback_max_samples)
        if os.path.exists(args.feedback_jsonl):
            fb_samples, fb_rejected = load_feedback_jsonl(args.feedback_jsonl, max_samples=fb_max)
            fb.add_many(fb_samples)
            print(f"Loaded feedback: kept={len(fb_samples)} rejected={fb_rejected}")
        else:
            print(f"Feedback file not found: {args.feedback_jsonl}")

    # ---- checkpoint loading ----
    loaded_ckpt = None
    loaded_ckpt_iter = None

    def _load_checkpoint(path: str) -> int | None:
        payload = torch.load(path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            net.load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                opt.load_state_dict(payload["optimizer_state_dict"])
            return int(payload["iter"]) if "iter" in payload else None
        else:
            net.load_state_dict(payload)
            return None

    if args.prefer_puzzle_init and args.puzzle_checkpoint and os.path.exists(args.puzzle_checkpoint):
        loaded_ckpt_iter = _load_checkpoint(args.puzzle_checkpoint)
        loaded_ckpt = args.puzzle_checkpoint
    elif args.init_checkpoint and os.path.exists(args.init_checkpoint):
        loaded_ckpt_iter = _load_checkpoint(args.init_checkpoint)
        loaded_ckpt = args.init_checkpoint

    init_eval_score: float | None = None
    last_eval_score: float | None = None
    best_eval_score: float = -1.0
    best_eval_iter: int = 0
    last_gate_score: float | None = None
    best_gate_score: float = -1.0
    best_gate_iter: int = 0
    accepted_count = 0
    rejected_count = 0

    if loaded_ckpt:
        tag = f" (iter={loaded_ckpt_iter})" if loaded_ckpt_iter is not None else ""
        print(f"Loaded {loaded_ckpt}{tag}")
        with torch.inference_mode():
            eval_stats = eval_net_vs_random(net, games=12, num_sims=int(args.eval_num_sims), device=device)
        init_eval_score = float(eval_stats["score"])
        last_eval_score = init_eval_score
        best_eval_score = init_eval_score
        print(f"[INIT EVAL] W={eval_stats['wins']} D={eval_stats['draws']} L={eval_stats['losses']} score={eval_stats['score']:.2f}")
        writer.add_scalar("eval_random/init_score", eval_stats["score"], 0)
    else:
        print("No checkpoint loaded. Use --prefer_puzzle_init to start from puzzle pretraining.")

    loaded = load_shards_into_buffer(rb, out_dir=args.replay_dir, max_samples=int(args.replay_maxlen))
    if loaded > 0:
        print(f"Loaded {loaded} replay samples from ./{args.replay_dir}")

    iters = int(args.iters)
    games_per_iter = int(args.games_per_iter)
    batch_size = int(args.batch_size)
    train_batches = int(args.train_batches)
    base_lr = float(args.lr)
    total_train_steps = iters * train_batches
    warmup_steps = min(2 * train_batches, total_train_steps // 5)
    global_step = 0

    selfplay_kwargs = {
        "num_sims": int(args.num_sims), "device": device, "pgn_dir": "games",
        "verbose": False, "draw_penalty": float(args.draw_penalty),
        "stop_on_threefold": bool(args.stop_on_threefold),
        "no_progress_limit": int(args.no_progress_limit),
        "no_progress_penalty": float(args.no_progress_penalty),
        "repeat2_penalty": float(args.repeat2_penalty),
        "stop_on_repeat2": bool(args.stop_on_repeat2),
        "temp_floor": float(args.temp_floor),
        "temp_moves": int(args.temp_moves),
        "use_material_shaping": bool(args.use_material_shaping),
        "material_scale": float(args.material_scale),
        "exchange_scale": float(args.exchange_scale),
        "early_sims": None if int(args.early_sims) <= 0 else int(args.early_sims),
        "early_plies": int(args.early_plies),
        "late_sims": None if int(args.late_sims) <= 0 else int(args.late_sims),
    }

    print(
        f"[Setup] iters={iters} games/iter={games_per_iter} bs={batch_size} "
        f"train_batches={train_batches} lr={base_lr:.2e} wd={args.weight_decay:.2e} "
        f"sims={args.num_sims} gate={args.gate_games} augment={args.augment} "
        f"channels={channels} blocks={num_blocks} replay_maxlen={args.replay_maxlen}"
    )

    if len(rb) == 0:
        bootstrap_games = 5
        for i in range(bootstrap_games):
            with torch.inference_mode():
                samples, stats, pgn_path = play_self_game(net, **selfplay_kwargs)
            rb.add_many(samples)
            print(f"Bootstrap {i+1}/{bootstrap_games}: {stats['result_str']} plies={stats['plies']}")

    try:
        for it in range(1, iters + 1):
            print(f"\n=== Iteration {it}/{iters} ===")

            # ---- self-play ----
            results, plies_list, z_list, pgn_paths, iter_samples = [], [], [], [], []
            no_prog_list, repeat2_list = [], []

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
                no_prog_list.append(stats.get("broke_no_progress", 0))
                repeat2_list.append(stats.get("broke_repeat2", 0))

            shard_path = save_shard(iter_samples, out_dir=args.replay_dir)
            cnt = Counter(results)
            avg_plies = sum(plies_list) / max(1, len(plies_list))
            avg_z = float(np.mean(z_list)) if z_list else 0.0
            print(
                f"Self-play: W {cnt.get('1-0',0)} D {cnt.get('1/2-1/2',0)} "
                f"L {cnt.get('0-1',0)} | plies {avg_plies:.1f} | z {avg_z:+.2f} | shard {shard_path}"
            )

            writer.add_scalar("selfplay/avg_plies", avg_plies, it)
            writer.add_scalar("selfplay/avg_z_white", avg_z, it)
            writer.add_scalar("replay/size", len(rb), it)

            # ---- train ----
            if len(rb) < batch_size:
                print("Replay buffer too small to train yet.")
                continue

            before_state = copy.deepcopy(net.state_dict())
            before_opt = copy.deepcopy(opt.state_dict())
            losses = []
            use_feedback = len(fb) >= int(args.feedback_batch_size) and float(args.feedback_weight) > 0.0

            for _ in range(train_batches):
                lr = _cosine_lr(base_lr, global_step, total_train_steps, warmup_steps)
                _set_lr(opt, lr)
                global_step += 1

                s, pi, v = rb.sample(batch_size, augment=bool(args.augment))
                if use_feedback:
                    fs, fgood, fbad, fw = fb.sample(int(args.feedback_batch_size))
                    losses.append(train_step_with_feedback(
                        net, opt, s, pi, v, fs, fgood, fbad, fw,
                        device=device,
                        feedback_weight=float(args.feedback_weight),
                        feedback_margin=float(args.feedback_margin),
                    ))
                else:
                    base_loss = train_step(net, opt, s, pi, v, device=device)
                    losses.append((base_loss[0], base_loss[1], base_loss[2], 0.0))

            avg = np.mean(losses, axis=0)
            print(
                f"Iter {it:03d} | loss={avg[0]:.4f} pol={avg[1]:.4f} val={avg[2]:.4f} "
                f"fb={avg[3]:.4f} lr={lr:.2e} | rb={len(rb)}"
            )
            writer.add_scalar("train/loss", avg[0], it)
            writer.add_scalar("train/policy_loss", avg[1], it)
            writer.add_scalar("train/value_loss", avg[2], it)
            writer.add_scalar("train/feedback_loss", avg[3], it)
            writer.add_scalar("train/lr", lr, it)

            # ---- gating ----
            accepted = True
            if int(args.gate_games) > 0:
                baseline = AlphaZeroNet(in_channels=IN_CHANNELS, channels=channels,
                                        num_blocks=num_blocks).to(device)
                baseline.load_state_dict(before_state)
                baseline.eval()
                with torch.inference_mode():
                    gate_stats = eval_candidate_vs_baseline(
                        net, baseline, games=int(args.gate_games),
                        num_sims=int(args.eval_num_sims), device=device,
                    )
                gate_score = float(gate_stats["score"])
                print(
                    f"[GATE] W={gate_stats['wins']} D={gate_stats['draws']} "
                    f"L={gate_stats['losses']} score={gate_score:.2f} thresh={args.gate_min_score}"
                )
                writer.add_scalar("gate/score_vs_prev", gate_score, it)
                accepted = gate_score >= float(args.gate_min_score)
                if gate_score > best_gate_score:
                    best_gate_score = gate_score
                    best_gate_iter = it
                last_gate_score = gate_score

                if not accepted:
                    net.load_state_dict(before_state)
                    opt.load_state_dict(before_opt)
                    rejected_count += 1
                    print("[GATE] rejected; reverted.")
                else:
                    accepted_count += 1
                    print(f"[GATE] accepted (total accepted={accepted_count} rejected={rejected_count})")
            else:
                accepted_count += 1

            if accepted:
                payload = {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "iter": it,
                    "channels": channels,
                    "num_blocks": num_blocks,
                    "in_channels": IN_CHANNELS,
                }
                torch.save(payload, "checkpoint_latest.pt")
                if it % 10 == 0:
                    torch.save(payload, f"checkpoint_{it:03d}.pt")

            # ---- eval vs random ----
            if it % 2 == 0:
                with torch.inference_mode():
                    ev = eval_net_vs_random(net, games=12, num_sims=int(args.eval_num_sims), device=device)
                eval_score = float(ev["score"])
                improved = eval_score > best_eval_score
                if improved:
                    best_eval_score = eval_score
                    best_eval_iter = it
                last_eval_score = eval_score
                print(
                    f"[EVAL] W={ev['wins']} D={ev['draws']} L={ev['losses']} "
                    f"score={eval_score:.2f} best={best_eval_score:.2f}@{best_eval_iter}"
                    + (" NEW BEST" if improved else "")
                )
                writer.add_scalar("eval_random/score", eval_score, it)

            print(f"[ITER {it:03d}] accepted={accepted_count} rejected={rejected_count}")

    finally:
        writer.close()


if __name__ == "__main__":
    main()
