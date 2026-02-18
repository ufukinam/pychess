# eval.py  (NEW)
from __future__ import annotations

import numpy as np
import chess

from env import ChessEnv
from mcts import Node, mcts_policy_and_action, reuse_root_after_action


def action_to_move(action: int) -> chess.Move:
    frm = action // (64 * 5)
    to = (action // 5) % 64
    promo_idx = action % 5
    promo = None if promo_idx == 0 else {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}[promo_idx]
    return chess.Move(frm, to, promotion=promo)


def play_vs_random(
    net,
    net_plays_white: bool,
    num_sims: int = 50,
    max_plies: int = 200,
    device: str = "cpu",
) -> float:
    """
    Returns game result from net perspective:
      +1 net win, 0 draw, -1 net loss
    """
    env = ChessEnv()
    board = env.reset()

    root = Node(board.copy(stack=False))
    ply = 0

    while (not env.is_terminal()) and ply < max_plies:
        net_to_move = (board.turn == chess.WHITE and net_plays_white) or (board.turn == chess.BLACK and not net_plays_white)

        if net_to_move:
            # Deterministic-ish for evaluation: temperature ~ 0
            pi, action = mcts_policy_and_action(
                net,
                root=root,
                num_sims=num_sims,
                temperature=1e-6,
                device=device,
            )
            mv = action_to_move(action)
            if mv not in board.legal_moves:
                mv = np.random.choice(list(board.legal_moves))
        else:
            mv = np.random.choice(list(board.legal_moves))

        env.push(mv)
        board = env.board
        ply += 1

        # If net moved, reuse tree. If random moved, easiest is to reset root.
        if net_to_move:
            root = reuse_root_after_action(root, action)
            root.board = board.copy(stack=False)
        else:
            root = Node(board.copy(stack=False))

    # Convert final result to net perspective
    if env.is_terminal():
        z_white = env.result_value()  # +1 white win, 0 draw, -1 black win
    else:
        z_white = 0.0

    if net_plays_white:
        return float(z_white)
    else:
        return float(-z_white)


def eval_net_vs_random(
    net,
    games: int = 4,
    num_sims: int = 25,
    device: str = "cpu",
) -> dict:
    """
    Plays half games as White, half as Black (roughly).
    Returns dict with win/draw/loss counts and score.
    """
    wins = draws = losses = 0
    results = []

    for i in range(games):
        net_white = (i % 2 == 0)
        r = play_vs_random(net, net_plays_white=net_white, num_sims=num_sims, device=device)
        results.append(r)
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1

    score = (wins + 0.5 * draws) / games
    avg_result = float(np.mean(results)) if results else 0.0

    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": float(score),       # 1.0 is perfect, 0.5 is all draws, 0 is all losses
        "avg_result": avg_result,     # in [-1,1]
    }
