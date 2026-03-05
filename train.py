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
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone

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


def _score_ci95(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    """
    Return (score, low95, high95) where score uses chess points:
    win=1.0, draw=0.5, loss=0.0.
    """
    n = max(1, int(wins) + int(draws) + int(losses))
    score = (float(wins) + 0.5 * float(draws)) / float(n)
    ex2 = (float(wins) + 0.25 * float(draws)) / float(n)
    var = max(0.0, ex2 - score * score)
    se = math.sqrt(var / float(n))
    margin = 1.96 * se
    return score, max(0.0, score - margin), min(1.0, score + margin)


def _elo_from_score(score: float, eps: float = 1e-6) -> float:
    s = min(max(float(score), eps), 1.0 - eps)
    return -400.0 * math.log10((1.0 / s) - 1.0)


def _score_for_mode(score: float, ci_low: float, mode: str) -> float:
    return float(ci_low if str(mode).lower() == "ci_low" else score)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_jsonl(path: str, row: dict) -> None:
    if not path:
        return
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Self-play training loop.")
    ap.add_argument("--init_checkpoint", type=str, default="checkpoint_latest.pt")
    ap.add_argument("--latest_checkpoint", type=str, default="checkpoint_latest.pt")
    ap.add_argument("--best_checkpoint", type=str, default="checkpoint_best.pt")
    ap.add_argument("--puzzle_checkpoint", type=str, default="checkpoint_puzzle_latest.pt")
    ap.add_argument("--prefer_puzzle_init", action="store_true")
    ap.add_argument(
        "--load_optimizer_from_puzzle_init",
        action="store_true",
        help=(
            "When using --prefer_puzzle_init, also restore optimizer state from the puzzle checkpoint. "
            "By default only model weights are transferred."
        ),
    )
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
    ap.add_argument(
        "--no_claim_draw_terminal",
        action="store_true",
        help="Do not treat claimable draws as immediate terminal states during self-play.",
    )
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
    ap.add_argument(
        "--gate_score_mode",
        type=str,
        choices=("score", "ci_low"),
        default="score",
        help="Use raw gate score or ci95 lower bound for accept/reject decisions.",
    )
    ap.add_argument(
        "--gate_random_opening_plies",
        type=int,
        default=6,
        help="For gate eval only: randomize each game start with 0..N random legal plies.",
    )
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_games", type=int, default=12)
    ap.add_argument(
        "--best_score_mode",
        type=str,
        choices=("score", "ci_low"),
        default="ci_low",
        help="Use raw eval score or ci95 lower bound when updating checkpoint_best.",
    )
    ap.add_argument(
        "--best_promotion_rule",
        type=str,
        choices=("eval_only", "and_gate_eval"),
        default="and_gate_eval",
        help="Best-checkpoint promotion: eval-only or require both gate+eval improvements.",
    )
    ap.add_argument(
        "--scoreboard_jsonl",
        type=str,
        default="training_scoreboard.jsonl",
        help="Append per-iteration metrics to JSONL (empty disables).",
    )
    ap.add_argument("--feedback_jsonl", type=str, default="")
    ap.add_argument("--feedback_weight", type=float, default=0.2)
    ap.add_argument("--feedback_batch_size", type=int, default=32)
    ap.add_argument("--feedback_margin", type=float, default=0.2)
    ap.add_argument("--feedback_max_samples", type=int, default=0)
    ap.add_argument(
        "--disable_pgn",
        action="store_true",
        help="Disable per-game PGN export during self-play for higher throughput.",
    )
    ap.add_argument(
        "--disable_replay_compression",
        action="store_true",
        help="Disable replay shard compression (faster writes, larger files).",
    )
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

    selfplay_pgn_dir = None if bool(args.disable_pgn) else "games"
    if selfplay_pgn_dir is not None:
        os.makedirs(selfplay_pgn_dir, exist_ok=True)
    os.makedirs(args.replay_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    _ensure_parent_dir(args.latest_checkpoint)
    _ensure_parent_dir(args.best_checkpoint)
    if args.scoreboard_jsonl:
        _ensure_parent_dir(args.scoreboard_jsonl)

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
    loaded_ckpt_opt = False
    init_checkpoint_path = str(args.init_checkpoint)
    if init_checkpoint_path == "checkpoint_latest.pt" and str(args.latest_checkpoint) != "checkpoint_latest.pt":
        init_checkpoint_path = str(args.latest_checkpoint)

    def _load_checkpoint(path: str, load_optimizer: bool = True) -> tuple[int | None, bool]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            net.load_state_dict(payload["model_state_dict"])
            optimizer_loaded = False
            if load_optimizer and "optimizer_state_dict" in payload:
                opt.load_state_dict(payload["optimizer_state_dict"])
                optimizer_loaded = True
            step = payload.get("iter", payload.get("epoch"))
            step_i = int(step) if step is not None else None
            return step_i, optimizer_loaded
        else:
            net.load_state_dict(payload)
            return None, False

    if args.prefer_puzzle_init and args.puzzle_checkpoint and os.path.exists(args.puzzle_checkpoint):
        loaded_ckpt_iter, loaded_ckpt_opt = _load_checkpoint(
            args.puzzle_checkpoint,
            load_optimizer=bool(args.load_optimizer_from_puzzle_init),
        )
        loaded_ckpt = args.puzzle_checkpoint
    elif init_checkpoint_path and os.path.exists(init_checkpoint_path):
        loaded_ckpt_iter, loaded_ckpt_opt = _load_checkpoint(
            init_checkpoint_path,
            load_optimizer=True,
        )
        loaded_ckpt = init_checkpoint_path

    init_eval_score: float | None = None
    last_eval_score: float | None = None
    best_eval_score: float = -1.0
    best_eval_iter: int = 0
    last_gate_score: float | None = None
    best_gate_score: float = -1.0
    best_gate_iter: int = 0
    accepted_count = 0
    rejected_count = 0
    best_checkpoint_metric: float = -1.0
    best_checkpoint_iter: int = 0
    best_checkpoint_gate_metric: float = -1.0

    if args.best_checkpoint and os.path.exists(args.best_checkpoint):
        try:
            best_payload = torch.load(args.best_checkpoint, map_location=device, weights_only=False)
            if isinstance(best_payload, dict):
                metric = best_payload.get("best_score_metric")
                mode = best_payload.get("best_score_mode")
                if metric is not None and (mode is None or str(mode) == str(args.best_score_mode)):
                    best_checkpoint_metric = float(metric)
                    best_checkpoint_iter = int(best_payload.get("best_score_iter", best_payload.get("iter", 0) or 0))
                    gate_metric = best_payload.get("best_gate_metric")
                    if gate_metric is not None:
                        best_checkpoint_gate_metric = float(gate_metric)
                    print(
                        f"[BEST] baseline loaded metric={best_checkpoint_metric:.3f} "
                        f"mode={args.best_score_mode} iter={best_checkpoint_iter}"
                    )
                elif "model_state_dict" in best_payload:
                    b_channels = int(best_payload.get("channels", channels))
                    b_blocks = int(best_payload.get("num_blocks", num_blocks))
                    b_in = int(best_payload.get("in_channels", IN_CHANNELS))
                    if b_channels == channels and b_blocks == num_blocks and b_in == IN_CHANNELS:
                        best_net = AlphaZeroNet(in_channels=IN_CHANNELS, channels=channels, num_blocks=num_blocks).to(device)
                        best_net.load_state_dict(best_payload["model_state_dict"])
                        best_net.eval()
                        with torch.inference_mode():
                            b_ev = eval_net_vs_random(
                                best_net,
                                games=int(args.eval_games),
                                num_sims=int(args.eval_num_sims),
                                device=device,
                            )
                        _, b_ci_lo, _ = _score_ci95(
                            int(b_ev["wins"]),
                            int(b_ev["draws"]),
                            int(b_ev["losses"]),
                        )
                        best_checkpoint_metric = _score_for_mode(
                            float(b_ev["score"]), b_ci_lo, str(args.best_score_mode)
                        )
                        best_checkpoint_iter = int(best_payload.get("iter", 0) or 0)
                        print(
                            f"[BEST] baseline evaluated metric={best_checkpoint_metric:.3f} "
                            f"mode={args.best_score_mode} iter={best_checkpoint_iter}"
                        )
                    else:
                        print(
                            "[BEST] existing checkpoint_best architecture mismatch; "
                            "baseline metric will be re-seeded this run."
                        )
                else:
                    print("[BEST] existing checkpoint_best payload unsupported; baseline will be re-seeded this run.")
        except Exception as exc:
            print(f"[BEST] warning: failed to read {args.best_checkpoint}: {exc}")

    if loaded_ckpt:
        tag = f" (iter={loaded_ckpt_iter})" if loaded_ckpt_iter is not None else ""
        opt_tag = "optimizer=loaded" if loaded_ckpt_opt else "optimizer=not_loaded"
        print(f"Loaded {loaded_ckpt}{tag} [{opt_tag}]")
        with torch.inference_mode():
            eval_stats = eval_net_vs_random(
                net,
                games=int(args.eval_games),
                num_sims=int(args.eval_num_sims),
                device=device,
            )
        init_eval_score = float(eval_stats["score"])
        last_eval_score = init_eval_score
        best_eval_score = init_eval_score
        _, init_eval_lo, init_eval_hi = _score_ci95(
            int(eval_stats["wins"]),
            int(eval_stats["draws"]),
            int(eval_stats["losses"]),
        )
        print(f"[INIT EVAL] W={eval_stats['wins']} D={eval_stats['draws']} L={eval_stats['losses']} score={eval_stats['score']:.2f}")
        print(f"[INIT EVAL] ci95=[{init_eval_lo:.2f},{init_eval_hi:.2f}]")
        writer.add_scalar("eval_random/init_score", eval_stats["score"], 0)
        writer.add_scalar("eval_random/init_score_ci_low", init_eval_lo, 0)
        writer.add_scalar("eval_random/init_score_ci_high", init_eval_hi, 0)
        init_metric = _score_for_mode(init_eval_score, init_eval_lo, str(args.best_score_mode))
        if best_checkpoint_metric < 0.0:
            best_checkpoint_metric = float(init_metric)
            best_checkpoint_iter = int(loaded_ckpt_iter or 0)
            print(
                f"[BEST] baseline seeded from init eval: metric={best_checkpoint_metric:.3f} "
                f"(mode={args.best_score_mode})"
            )
            if args.best_checkpoint and not os.path.exists(args.best_checkpoint):
                init_best_payload = {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "iter": int(loaded_ckpt_iter or 0),
                    "channels": channels,
                    "num_blocks": num_blocks,
                    "in_channels": IN_CHANNELS,
                    "best_score_metric": best_checkpoint_metric,
                    "best_score_mode": str(args.best_score_mode),
                    "best_score_iter": best_checkpoint_iter,
                    "best_gate_metric": float(best_checkpoint_gate_metric),
                    "best_promotion_rule": str(args.best_promotion_rule),
                    "eval_random_score": init_eval_score,
                    "eval_random_ci_low": float(init_eval_lo),
                    "eval_random_ci_high": float(init_eval_hi),
                }
                torch.save(init_best_payload, args.best_checkpoint)
                print(f"[BEST] initialized {args.best_checkpoint}")
        _append_jsonl(
            str(args.scoreboard_jsonl),
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "init_eval",
                "checkpoint_loaded": str(loaded_ckpt),
                "iter_loaded": int(loaded_ckpt_iter or 0),
                "eval_games": int(args.eval_games),
                "eval_score": float(init_eval_score),
                "eval_ci_low": float(init_eval_lo),
                "eval_ci_high": float(init_eval_hi),
                "best_score_mode": str(args.best_score_mode),
                "best_score_metric": float(best_checkpoint_metric),
                "best_score_iter": int(best_checkpoint_iter),
                "best_gate_metric": float(best_checkpoint_gate_metric),
                "best_promotion_rule": str(args.best_promotion_rule),
            },
        )
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
        "num_sims": int(args.num_sims), "device": device, "pgn_dir": selfplay_pgn_dir,
        "verbose": False, "draw_penalty": float(args.draw_penalty),
        "claim_draw_terminal": not bool(args.no_claim_draw_terminal),
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
    replay_compression = "none" if bool(args.disable_replay_compression) else "compressed"

    print(
        f"[Setup] iters={iters} games/iter={games_per_iter} bs={batch_size} "
        f"train_batches={train_batches} lr={base_lr:.2e} wd={args.weight_decay:.2e} "
        f"sims={args.num_sims} gate={args.gate_games} gate_rand_open={args.gate_random_opening_plies} "
        f"gate_mode={args.gate_score_mode} eval_every={args.eval_every} eval_games={args.eval_games} "
        f"best_mode={args.best_score_mode} best_rule={args.best_promotion_rule} "
        f"augment={args.augment} no_claim_draw_terminal={bool(args.no_claim_draw_terminal)} "
        f"channels={channels} blocks={num_blocks} replay_maxlen={args.replay_maxlen} "
        f"pgn={'off' if selfplay_pgn_dir is None else selfplay_pgn_dir} "
        f"replay_compression={replay_compression}"
    )
    print(
        f"[Checkpoints] latest={args.latest_checkpoint} best={args.best_checkpoint} "
        f"scoreboard={'off' if not args.scoreboard_jsonl else args.scoreboard_jsonl}"
    )
    best_requires_gate = str(args.best_promotion_rule) == "and_gate_eval"
    if best_requires_gate and int(args.gate_games) <= 0:
        print("[BEST] and_gate_eval requested but gate_games=0; best promotion falls back to eval-only.")

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
            iter_t0 = time.perf_counter()
            iter_gate_score = None
            iter_gate_ci_low = None
            iter_gate_ci_high = None
            iter_gate_metric = None
            iter_eval_score = None
            iter_eval_ci_low = None
            iter_eval_ci_high = None
            iter_eval_metric = None
            iter_best_promoted = False
            iter_best_eval_improved = None
            iter_best_gate_improved = None

            # ---- self-play ----
            results, plies_list, z_list, pgn_paths, iter_samples = [], [], [], [], []
            no_prog_list, repeat2_list = [], []
            game_secs = []
            selfplay_t0 = time.perf_counter()

            for _ in range(games_per_iter):
                game_t0 = time.perf_counter()
                with torch.inference_mode():
                    samples, stats, pgn_path = play_self_game(net, **selfplay_kwargs)
                game_secs.append(time.perf_counter() - game_t0)
                rb.add_many(samples)
                iter_samples.extend(samples)
                results.append(stats["result_str"])
                plies_list.append(stats["plies"])
                z_list.append(stats["result_z_white"])
                if pgn_path:
                    pgn_paths.append(pgn_path)
                no_prog_list.append(stats.get("broke_no_progress", 0))
                repeat2_list.append(stats.get("broke_repeat2", 0))

            selfplay_dt = time.perf_counter() - selfplay_t0
            shard_path = save_shard(
                iter_samples,
                out_dir=args.replay_dir,
                compression=replay_compression,
            )
            cnt = Counter(results)
            avg_plies = sum(plies_list) / max(1, len(plies_list))
            avg_z = float(np.mean(z_list)) if z_list else 0.0
            z_abs_mean = float(np.mean(np.abs(z_list))) if z_list else 0.0
            z_std = float(np.std(z_list)) if z_list else 0.0
            z_near_zero = float(np.mean(np.abs(z_list) <= 0.1)) if z_list else 0.0
            total_plies = int(sum(plies_list))
            sec_per_game = selfplay_dt / max(1, len(game_secs))
            sec_per_ply = selfplay_dt / max(1, total_plies)
            samples_per_sec = len(iter_samples) / max(1e-9, selfplay_dt)
            wins = int(cnt.get("1-0", 0))
            draws = int(cnt.get("1/2-1/2", 0))
            losses = int(cnt.get("0-1", 0))
            n_games = max(1, wins + draws + losses)
            decisive_rate = float((wins + losses) / n_games)
            draw_rate = float(draws / n_games)
            repeat2_rate = float(np.mean(repeat2_list)) if repeat2_list else 0.0
            no_prog_rate = float(np.mean(no_prog_list)) if no_prog_list else 0.0
            print(
                f"Self-play: W {cnt.get('1-0',0)} D {cnt.get('1/2-1/2',0)} "
                f"L {cnt.get('0-1',0)} | plies {avg_plies:.1f} | z {avg_z:+.2f} | shard {shard_path}"
            )
            print(
                f"[INDICATORS selfplay] decisive={decisive_rate:.2f} draw={draw_rate:.2f} "
                f"repeat2_stop={repeat2_rate:.2f} no_prog_stop={no_prog_rate:.2f} "
                f"| z_abs={z_abs_mean:.3f} z_std={z_std:.3f} z_near0={z_near_zero:.2f}"
            )
            print(
                f"[TIME selfplay] total={selfplay_dt:.2f}s sec/game={sec_per_game:.2f} "
                f"sec/ply={sec_per_ply:.3f} samples/sec={samples_per_sec:.1f}"
            )

            writer.add_scalar("selfplay/avg_plies", avg_plies, it)
            writer.add_scalar("selfplay/avg_z_white", avg_z, it)
            writer.add_scalar("selfplay/z_abs_mean", z_abs_mean, it)
            writer.add_scalar("selfplay/z_std", z_std, it)
            writer.add_scalar("selfplay/z_near_zero_rate", z_near_zero, it)
            writer.add_scalar("selfplay/draw_rate", draw_rate, it)
            writer.add_scalar("selfplay/decisive_rate", decisive_rate, it)
            writer.add_scalar("selfplay/repeat2_stop_rate", repeat2_rate, it)
            writer.add_scalar("selfplay/no_progress_stop_rate", no_prog_rate, it)
            writer.add_scalar("replay/size", len(rb), it)
            writer.add_scalar("selfplay/sec_total", selfplay_dt, it)
            writer.add_scalar("selfplay/sec_per_game", sec_per_game, it)
            writer.add_scalar("selfplay/sec_per_ply", sec_per_ply, it)
            writer.add_scalar("selfplay/samples_per_sec", samples_per_sec, it)

            # ---- train ----
            if len(rb) < batch_size:
                print("Replay buffer too small to train yet.")
                continue

            before_state = copy.deepcopy(net.state_dict())
            before_opt = copy.deepcopy(opt.state_dict())
            losses = []
            use_feedback = len(fb) >= int(args.feedback_batch_size) and float(args.feedback_weight) > 0.0
            train_t0 = time.perf_counter()
            batch_secs = []
            lr = base_lr

            for _ in range(train_batches):
                batch_t0 = time.perf_counter()
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
                batch_secs.append(time.perf_counter() - batch_t0)

            train_dt = time.perf_counter() - train_t0
            sec_per_batch = train_dt / max(1, len(batch_secs))
            avg = np.mean(losses, axis=0)
            print(
                f"Iter {it:03d} | loss={avg[0]:.4f} pol={avg[1]:.4f} val={avg[2]:.4f} "
                f"fb={avg[3]:.4f} lr={lr:.2e} | rb={len(rb)}"
            )
            print(
                f"[TIME train] total={train_dt:.2f}s sec/batch={sec_per_batch:.3f} "
                f"batches/sec={len(batch_secs) / max(1e-9, train_dt):.2f}"
            )
            writer.add_scalar("train/loss", avg[0], it)
            writer.add_scalar("train/policy_loss", avg[1], it)
            writer.add_scalar("train/value_loss", avg[2], it)
            writer.add_scalar("train/feedback_loss", avg[3], it)
            writer.add_scalar("train/lr", lr, it)
            writer.add_scalar("train/sec_total", train_dt, it)
            writer.add_scalar("train/sec_per_batch", sec_per_batch, it)
            writer.add_scalar("train/batches_per_sec", len(batch_secs) / max(1e-9, train_dt), it)

            # ---- gating ----
            accepted = True
            if int(args.gate_games) > 0:
                gate_t0 = time.perf_counter()
                baseline = AlphaZeroNet(in_channels=IN_CHANNELS, channels=channels,
                                        num_blocks=num_blocks).to(device)
                baseline.load_state_dict(before_state)
                baseline.eval()
                with torch.inference_mode():
                    gate_stats = eval_candidate_vs_baseline(
                        net, baseline, games=int(args.gate_games),
                        num_sims=int(args.eval_num_sims), device=device,
                        random_opening_plies=int(args.gate_random_opening_plies),
                    )
                gate_score = float(gate_stats["score"])
                gate_w = int(gate_stats["wins"])
                gate_d = int(gate_stats["draws"])
                gate_l = int(gate_stats["losses"])
                _, gate_ci_lo, gate_ci_hi = _score_ci95(gate_w, gate_d, gate_l)
                gate_elo = _elo_from_score(gate_score)
                gate_delta = None if last_gate_score is None else (gate_score - float(last_gate_score))
                gate_metric = _score_for_mode(gate_score, gate_ci_lo, str(args.gate_score_mode))
                iter_gate_score = gate_score
                iter_gate_ci_low = float(gate_ci_lo)
                iter_gate_ci_high = float(gate_ci_hi)
                iter_gate_metric = float(gate_metric)
                print(
                    f"[GATE] W={gate_stats['wins']} D={gate_stats['draws']} "
                    f"L={gate_stats['losses']} score={gate_score:.2f} "
                    f"metric({args.gate_score_mode})={gate_metric:.2f} thresh={args.gate_min_score}"
                )
                print(
                    f"[INDICATORS gate] ci95=[{gate_ci_lo:.2f},{gate_ci_hi:.2f}] "
                    f"elo_vs_prev={gate_elo:+.0f}"
                    + ("" if gate_delta is None else f" delta={gate_delta:+.2f}")
                )
                writer.add_scalar("gate/score_vs_prev", gate_score, it)
                writer.add_scalar("gate/score_ci_low", gate_ci_lo, it)
                writer.add_scalar("gate/score_ci_high", gate_ci_hi, it)
                writer.add_scalar("gate/elo_vs_prev", gate_elo, it)
                writer.add_scalar("gate/metric_for_accept", gate_metric, it)
                if gate_delta is not None:
                    writer.add_scalar("gate/score_delta", gate_delta, it)
                accepted = gate_metric >= float(args.gate_min_score)
                if gate_score > best_gate_score:
                    best_gate_score = gate_score
                    best_gate_iter = it
                last_gate_score = gate_score
                gate_dt = time.perf_counter() - gate_t0
                print(f"[TIME gate] total={gate_dt:.2f}s")
                writer.add_scalar("gate/sec_total", gate_dt, it)

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
                torch.save(payload, args.latest_checkpoint)
                if it % 10 == 0:
                    torch.save(payload, f"checkpoint_{it:03d}.pt")

            # ---- eval vs random ----
            eval_every = int(args.eval_every)
            if eval_every > 0 and it % eval_every == 0:
                eval_t0 = time.perf_counter()
                with torch.inference_mode():
                    ev = eval_net_vs_random(
                        net,
                        games=int(args.eval_games),
                        num_sims=int(args.eval_num_sims),
                        device=device,
                    )
                eval_score = float(ev["score"])
                eval_w = int(ev["wins"])
                eval_d = int(ev["draws"])
                eval_l = int(ev["losses"])
                _, eval_ci_lo, eval_ci_hi = _score_ci95(eval_w, eval_d, eval_l)
                eval_metric = _score_for_mode(eval_score, eval_ci_lo, str(args.best_score_mode))
                iter_eval_score = eval_score
                iter_eval_ci_low = float(eval_ci_lo)
                iter_eval_ci_high = float(eval_ci_hi)
                iter_eval_metric = float(eval_metric)
                eval_delta = None if last_eval_score is None else (eval_score - float(last_eval_score))
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
                print(
                    f"[INDICATORS eval] ci95=[{eval_ci_lo:.2f},{eval_ci_hi:.2f}]"
                    + ("" if eval_delta is None else f" delta={eval_delta:+.2f}")
                )
                writer.add_scalar("eval_random/score", eval_score, it)
                writer.add_scalar("eval_random/score_ci_low", eval_ci_lo, it)
                writer.add_scalar("eval_random/score_ci_high", eval_ci_hi, it)
                writer.add_scalar("eval_random/metric_for_best", eval_metric, it)
                if eval_delta is not None:
                    writer.add_scalar("eval_random/score_delta", eval_delta, it)
                eval_dt = time.perf_counter() - eval_t0
                print(f"[TIME eval] total={eval_dt:.2f}s")
                writer.add_scalar("eval_random/sec_total", eval_dt, it)
                best_eval_improved = bool(eval_metric > best_checkpoint_metric)
                best_gate_improved = bool(
                    iter_gate_metric is not None and float(iter_gate_metric) > float(best_checkpoint_gate_metric)
                )
                iter_best_eval_improved = best_eval_improved
                iter_best_gate_improved = best_gate_improved
                should_promote_best = bool(accepted and best_eval_improved)
                if should_promote_best and best_requires_gate and int(args.gate_games) > 0:
                    should_promote_best = bool(best_gate_improved)
                    if not should_promote_best:
                        gate_val_txt = "nan" if iter_gate_metric is None else f"{float(iter_gate_metric):.3f}"
                        print(
                            "[BEST] hold: eval improved but gate metric did not beat current best gate metric "
                            f"({gate_val_txt} "
                            f"<= {best_checkpoint_gate_metric:.3f})"
                        )
                if should_promote_best:
                    best_checkpoint_metric = float(eval_metric)
                    best_checkpoint_iter = int(it)
                    if iter_gate_metric is not None:
                        best_checkpoint_gate_metric = float(iter_gate_metric)
                    best_payload = {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iter": it,
                        "channels": channels,
                        "num_blocks": num_blocks,
                        "in_channels": IN_CHANNELS,
                        "best_score_metric": float(best_checkpoint_metric),
                        "best_score_mode": str(args.best_score_mode),
                        "best_score_iter": int(best_checkpoint_iter),
                        "best_gate_metric": float(best_checkpoint_gate_metric),
                        "best_promotion_rule": str(args.best_promotion_rule),
                        "eval_random_score": float(eval_score),
                        "eval_random_ci_low": float(eval_ci_lo),
                        "eval_random_ci_high": float(eval_ci_hi),
                    }
                    torch.save(best_payload, args.best_checkpoint)
                    iter_best_promoted = True
                    print(
                        f"[BEST] updated -> {args.best_checkpoint} "
                        f"metric({args.best_score_mode})={best_checkpoint_metric:.3f} at iter={it}"
                    )
                writer.add_scalar("best/score_metric", best_checkpoint_metric, it)
                writer.add_scalar("best/score_iter", best_checkpoint_iter, it)
                writer.add_scalar("best/gate_metric", best_checkpoint_gate_metric, it)
                writer.add_scalar("best/promoted", 1.0 if iter_best_promoted else 0.0, it)
                writer.add_scalar("best/eval_improved", 1.0 if best_eval_improved else 0.0, it)
                writer.add_scalar("best/gate_improved", 1.0 if best_gate_improved else 0.0, it)

            print(f"[ITER {it:03d}] accepted={accepted_count} rejected={rejected_count}")
            iter_dt = time.perf_counter() - iter_t0
            print(f"[TIME iter] total={iter_dt:.2f}s")
            writer.add_scalar("iter/sec_total", iter_dt, it)
            _append_jsonl(
                str(args.scoreboard_jsonl),
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "iter",
                    "iter": int(it),
                    "accepted": bool(accepted),
                    "accepted_count": int(accepted_count),
                    "rejected_count": int(rejected_count),
                    "gate_score": None if iter_gate_score is None else float(iter_gate_score),
                    "gate_ci_low": None if iter_gate_ci_low is None else float(iter_gate_ci_low),
                    "gate_ci_high": None if iter_gate_ci_high is None else float(iter_gate_ci_high),
                    "gate_metric": None if iter_gate_metric is None else float(iter_gate_metric),
                    "gate_metric_mode": str(args.gate_score_mode),
                    "eval_score": None if iter_eval_score is None else float(iter_eval_score),
                    "eval_ci_low": None if iter_eval_ci_low is None else float(iter_eval_ci_low),
                    "eval_ci_high": None if iter_eval_ci_high is None else float(iter_eval_ci_high),
                    "best_update_metric": None if iter_eval_metric is None else float(iter_eval_metric),
                    "best_score_mode": str(args.best_score_mode),
                    "best_promotion_rule": str(args.best_promotion_rule),
                    "best_promoted": bool(iter_best_promoted),
                    "best_eval_improved": None if iter_best_eval_improved is None else bool(iter_best_eval_improved),
                    "best_gate_improved": None if iter_best_gate_improved is None else bool(iter_best_gate_improved),
                    "best_score_metric": float(best_checkpoint_metric),
                    "best_score_iter": int(best_checkpoint_iter),
                    "best_gate_metric": float(best_checkpoint_gate_metric),
                    "checkpoint_latest": str(args.latest_checkpoint),
                    "checkpoint_best": str(args.best_checkpoint),
                },
            )

    finally:
        writer.close()


if __name__ == "__main__":
    main()
